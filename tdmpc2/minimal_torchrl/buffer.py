# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import json
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

import torch

from tensordict import is_tensorclass
from tensordict.tensordict import (
    is_tensor_collection,
    # LazyStackedTensorDict,
    TensorDict,
    # TensorDictBase,
)


from .storage import ListStorage, Storage, INT_CLASSES
from .samplers import Sampler
from .writer import Writer, RoundRobinWriter
from .transform import Compose

def _to_numpy(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data

def pin_memory_output(fun) -> Callable:
    """Calls pin_memory on outputs of decorated function if they have such method."""

    def decorated_fun(self, *args, **kwargs):
        output = fun(self, *args, **kwargs)
        if self._pin_memory:
            _tuple_out = True
            if not isinstance(output, tuple):
                _tuple_out = False
                output = (output,)
            output = tuple(_pin_memory(_output) for _output in output)
            if _tuple_out:
                return output
            return output[0]
        return output

    return decorated_fun

def _pin_memory(output: Any) -> Any:
    if hasattr(output, "pin_memory") and output.device == torch.device("cpu"):
        return output.pin_memory()
    else:
        return output


class ReplayBuffer:
    """A generic, composable replay buffer class.

    Keyword Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~minimal_torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided,
            a default :class:`~minimal_torchrl.data.replay_buffers.RandomSampler`
            will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~minimal_torchrl.data.replay_buffers.RoundRobinWriter`
            will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform, optional): Transform to be executed when
            sample() is called.
            To chain transforms use the :class:`~minimal_torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. If used with other structures, the transforms should be
            encoded with a ``"data"`` leading key that will be used to
            construct a tensordict from the non-tensordict content.
        batch_size (int, optional): the batch size to be used when sample() is
            called.
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.

    Examples:
        >>> import torch
        >>>
        >>> from minimal_torchrl.data import ReplayBuffer, ListStorage
        >>>
        >>> torch.manual_seed(0)
        >>> rb = ReplayBuffer(
        ...     storage=ListStorage(max_size=1000),
        ...     batch_size=5,
        ... )
        >>> # populate the replay buffer and get the item indices
        >>> data = range(10)
        >>> indices = rb.extend(data)
        >>> # sample will return as many elements as specified in the constructor
        >>> sample = rb.sample()
        >>> print(sample)
        tensor([4, 9, 3, 0, 3])
        >>> # Passing the batch-size to the sample method overrides the one in the constructor
        >>> sample = rb.sample(batch_size=3)
        >>> print(sample)
        tensor([9, 7, 3])
        >>> # one cans sample using the ``sample`` method or iterate over the buffer
        >>> for i, batch in enumerate(rb):
        ...     print(i, batch)
        ...     if i == 3:
        ...         break
        0 tensor([7, 3, 1, 6, 6])
        1 tensor([9, 8, 6, 6, 8])
        2 tensor([4, 3, 6, 9, 1])
        3 tensor([4, 4, 1, 9, 9])

    Replay buffers accept *any* kind of data. Not all storage types
    will work, as some expect numerical data only, but the default
    :class:`minimal_torchrl.data.ListStorage` will:

    Examples:
        >>> torch.manual_seed(0)
        >>> buffer = ReplayBuffer(storage=ListStorage(100), collate_fn=lambda x: x)
        >>> indices = buffer.extend(["a", 1, None])
        >>> buffer.sample(3)
        [None, 'a', None]
    """

    def __init__(
        self,
        *,
        storage: Storage | None = None,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "Transform" | None = None,  # noqa-F821
        batch_size: int | None = None,
    ) -> None:
        self._storage = storage if storage is not None else ListStorage(max_size=1_000)
        self._storage.attach(self)
        self._sampler = sampler
        self._writer = writer if writer is not None else RoundRobinWriter()
        self._writer.register_storage(self._storage)

        self._collate_fn = lambda x: x
        self._pin_memory = pin_memory

        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()
        # from torchrl.envs.transforms.transforms import Compose

        if transform is None:
            transform = Compose()
        elif not isinstance(transform, Compose):
            transform = Compose(transform)
        transform.eval()
        self._transform = transform

        if batch_size is None and prefetch:
            raise ValueError(
                "Dynamic batch-size specification is incompatible "
                "with multithreaded sampling. "
                "When using prefetch, the batch-size must be specified in "
                "advance. "
            )
        if (
            batch_size is None
            and hasattr(self._sampler, "drop_last")
            and self._sampler.drop_last
        ):
            raise ValueError(
                "Samplers with drop_last=True must work with a predictible batch-size. "
                "Please pass the batch-size to the ReplayBuffer constructor."
            )
        self._batch_size = batch_size

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"storage={self._storage}, "
            f"sampler={self._sampler}, "
            f"writer={self._writer}"
            ")"
        )

    @pin_memory_output
    def __getitem__(self, index: Union[int, torch.Tensor]) -> Any:
        index = _to_numpy(index)
        with self._replay_lock:
            data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        if self._transform is not None and len(self._transform):
            is_td = True
            if not is_tensor_collection(data):
                data = TensorDict({"data": data}, [])
                is_td = False
            data = self._transform(data)
            if not is_td:
                data = data["data"]

        return data

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": self._storage.state_dict(),
            "_sampler": self._sampler.state_dict(),
            "_writer": self._writer.state_dict(),
            "_transforms": self._transform.state_dict(),
            "_batch_size": self._batch_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._storage.load_state_dict(state_dict["_storage"])
        self._sampler.load_state_dict(state_dict["_sampler"])
        self._writer.load_state_dict(state_dict["_writer"])
        self._transform.load_state_dict(state_dict["_transforms"])
        self._batch_size = state_dict["_batch_size"]

    def dumps(self, path):
        """Saves the replay buffer on disk at the specified path.

        Args:
            path (Path or str): path where to save the replay buffer.

        Examples:
            >>> import tempfile
            >>> import tqdm
            >>> from minimal_torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
            >>> from minimal_torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
            >>> import torch
            >>> from tensordict import TensorDict
            >>> # Build and populate the replay buffer
            >>> S = 1_000_000
            >>> sampler = PrioritizedSampler(S, 1.1, 1.0)
            >>> # sampler = RandomSampler()
            >>> storage = LazyMemmapStorage(S)
            >>> rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)
            >>>
            >>> for _ in tqdm.tqdm(range(100)):
            ...     td = TensorDict({"obs": torch.randn(100, 3, 4), "next": {"obs": torch.randn(100, 3, 4)}, "td_error": torch.rand(100)}, [100])
            ...     rb.extend(td)
            ...     sample = rb.sample(32)
            ...     rb.update_tensordict_priority(sample)
            >>> # save and load the buffer
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     rb.dumps(tmpdir)
            ...
            ...     sampler = PrioritizedSampler(S, 1.1, 1.0)
            ...     # sampler = RandomSampler()
            ...     storage = LazyMemmapStorage(S)
            ...     rb_load = TensorDictReplayBuffer(storage=storage, sampler=sampler)
            ...     rb_load.loads(tmpdir)
            ...     assert len(rb) == len(rb_load)

        """
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        self._storage.dumps(path / "storage")
        self._sampler.dumps(path / "sampler")
        self._writer.dumps(path / "writer")
        # fall back on state_dict for transforms
        transform_sd = self._transform.state_dict()
        if transform_sd:
            torch.save(transform_sd, path / "transform.t")
        with open(path / "buffer_metadata.json", "w") as file:
            json.dump({"batch_size": self._batch_size}, file)

    def loads(self, path):
        """Loads a replay buffer state at the given path.

        The buffer should have matching components and be saved using :meth:`~.dumps`.

        Args:
            path (Path or str): path where the replay buffer was saved.

        See :meth:`~.dumps` for more info.

        """
        path = Path(path).absolute()
        self._storage.loads(path / "storage")
        self._sampler.loads(path / "sampler")
        self._writer.loads(path / "writer")
        # fall back on state_dict for transforms
        if (path / "transform.t").exists():
            self._transform.load_state_dict(torch.load(path / "transform.t"))
        with open(path / "buffer_metadata.json", "r") as file:
            metadata = json.load(file)
        self._batch_size = metadata["batch_size"]

    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        if self._transform is not None and (
            is_tensor_collection(data) or len(self._transform)
        ):
            data = self._transform.inv(data)
        return self._add(data)

    def _add(self, data):
        with self._replay_lock:
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def _extend(self, data: Sequence) -> torch.Tensor:
        with self._replay_lock:
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        If present, the inverse transforms will be called.`

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data added to the replay buffer.
        """
        if self._transform is not None and (
            is_tensor_collection(data) or len(self._transform)
        ):
            data = self._transform.inv(data)
        return self._extend(data)

    def update_priority(
        self,
        index: Union[int, torch.Tensor],
        priority: Union[int, torch.Tensor],
    ) -> None:
        with self._replay_lock:
            self._sampler.update_priority(index, priority)

    @pin_memory_output
    def _sample(self, batch_size: int) -> Tuple[Any, dict]:
        with self._replay_lock:
            index, info = self._sampler.sample(self._storage, batch_size)
            info["index"] = index
            data = self._storage.get(index)
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            is_td = True
            if not is_tensor_collection(data):
                data = TensorDict({"data": data}, [])
                is_td = False
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            data = self._transform(data)
            if is_locked:
                data.lock_()
            if not is_td:
                data = data["data"]

        return data, info

    def empty(self):
        """Empties the replay buffer and reset cursor to 0."""
        self._writer._empty()
        self._sampler._empty()
        self._storage._empty()

    def sample(self, batch_size: int | None = None, return_info: bool = False) -> Any:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A batch of data selected in the replay buffer.
            A tuple containing this batch and info if return_info flag is set to True.
        """
        if (
            batch_size is not None
            and self._batch_size is not None
            and batch_size != self._batch_size
        ):
            warnings.warn(
                f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
                f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments. "
                "The batch-size provided to the sample method "
                "will prevail."
            )
        elif batch_size is None and self._batch_size is not None:
            batch_size = self._batch_size
        elif batch_size is None:
            raise RuntimeError(
                "batch_size not specified. You can specify the batch_size when "
                "constructing the replay buffer, or pass it to the sample method. "
                "Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments."
            )
        if not self._prefetch:
            ret = self._sample(batch_size)
        else:
            with self._futures_lock:
                while len(self._prefetch_queue) < self._prefetch_cap:
                    fut = self._prefetch_executor.submit(self._sample, batch_size)
                    self._prefetch_queue.append(fut)
                ret = self._prefetch_queue.popleft().result()

        if return_info:
            return ret
        return ret[0]

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        self._sampler.mark_update(index)

    def append_transform(self, transform: "Transform") -> None:  # noqa-F821
        """Appends transform at the end.

        Transforms are applied in order when `sample` is called.

        Args:
            transform (Transform): The transform to be appended
        """
        transform.eval()
        self._transform.append(transform)

    def insert_transform(self, index: int, transform: "Transform") -> None:  # noqa-F821
        """Inserts transform.

        Transforms are executed in order when `sample` is called.

        Args:
            index (int): Position to insert the transform.
            transform (Transform): The transform to be appended
        """
        transform.eval()
        self._transform.insert(index, transform)

    def __iter__(self):
        if self._sampler.ran_out:
            self._sampler.ran_out = False
        if self._batch_size is None:
            raise RuntimeError(
                "Cannot iterate over the replay buffer. "
                "Batch_size was not specified during construction of the replay buffer."
            )
        while not self._sampler.ran_out:
            yield self.sample()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        _replay_lock = state.pop("_replay_lock", None)
        _futures_lock = state.pop("_futures_lock", None)
        if _replay_lock is not None:
            state["_replay_lock_placeholder"] = None
        if _futures_lock is not None:
            state["_futures_lock_placeholder"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        if "_replay_lock_placeholder" in state:
            state.pop("_replay_lock_placeholder")
            _replay_lock = threading.RLock()
            state["_replay_lock"] = _replay_lock
        if "_futures_lock_placeholder" in state:
            state.pop("_futures_lock_placeholder")
            _futures_lock = threading.RLock()
            state["_futures_lock"] = _futures_lock
        self.__dict__.update(state)

