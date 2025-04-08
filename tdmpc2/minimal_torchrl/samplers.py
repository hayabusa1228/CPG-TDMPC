# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
import textwrap
import warnings
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

# from tensordict import MemoryMappedTensor, TensorDict
from tensordict.utils import NestedKey

# from minimal_torchrl._extension import EXTENSION_WARNING

# from minimal_torchrl._utils import _replace_last
# from minimal_torchrl.data.replay_buffers.storages import Storage, StorageEnsemble, TensorStorage
# from minimal_torchrl.data.replay_buffers.utils import _to_numpy, INT_CLASSES

# try:
#     from minimal_torchrl._torchrl import (
#         MinSegmentTreeFp32,
#         MinSegmentTreeFp64,
#         SumSegmentTreeFp32,
#         SumSegmentTreeFp64,
#     )
# except ImportError:
#     warnings.warn(EXTENSION_WARNING)

from .storage import Storage, TensorStorage

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


def _replace_last(key: NestedKey, new_ending: str) -> NestedKey:
    if isinstance(key, str):
        return new_ending
    else:
        return key[:-1] + (new_ending,)


class Sampler(ABC):
    """A generic sampler base class for composable Replay Buffers."""

    @abstractmethod
    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        ...

    def add(self, index: int) -> None:
        return

    def extend(self, index: torch.Tensor) -> None:
        return

    def update_priority(
        self, index: Union[int, torch.Tensor], priority: Union[float, torch.Tensor]
    ) -> dict:
        return

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        return

    @property
    def default_priority(self) -> float:
        return 1.0

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @property
    def ran_out(self) -> bool:
        # by default, samplers never run out
        return False

    @abstractmethod
    def _empty(self):
        ...

    @abstractmethod
    def dumps(self, path):
        ...

    @abstractmethod
    def loads(self, path):
        ...


class RandomSampler(Sampler):
    """A uniformly random sampler for composable replay buffers.

    Args:
        batch_size (int, optional): if provided, the batch size to be used by
            the replay buffer when calling :meth:`~.ReplayBuffer.sample`.

    """

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        index = torch.randint(0, len(storage), (batch_size,))
        return index, {}

    def _empty(self):
        pass

    def dumps(self, path):
        # no op
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return


class SamplerWithoutReplacement(Sampler):
    """A data-consuming sampler that ensures that the same sample is not present in consecutive batches.

    Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept and (unlike with torch dataloaders)
            completed with other samples from a fresh indices permutation.
            Defaults to ``False``.
        shuffle (bool, optional): if ``False``, the items are not randomly
            permuted. This enables to iterate over the replay buffer in the
            order the data was collected. Defaults to ``True``.

    *Caution*: If the size of the storage changes in between two calls, the samples will be re-shuffled
    (as we can't generally keep track of which samples have been sampled before and which haven't).

    Similarly, it is expected that the storage content remains the same in between two calls,
    but this is not enforced.

    When the sampler reaches the end of the list of available indices, a new sample order
    will be generated and the resulting indices will be completed with this new draw, which
    can lead to duplicated indices, unless the :obj:`drop_last` argument is set to ``True``.

    """

    def __init__(self, drop_last: bool = False, shuffle: bool = True):
        self._sample_list = None
        self.len_storage = 0
        self.drop_last = drop_last
        self._ran_out = False
        self.shuffle = shuffle

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        with open(path / "sampler_metadata.json", "w") as file:
            json.dump(
                {
                    "len_storage": self.len_storage,
                    "_sample_list": self._sample_list,
                    "drop_last": self.drop_last,
                    "_ran_out": self._ran_out,
                },
                file,
            )

    def loads(self, path):
        with open(path / "sampler_metadata.json", "r") as file:
            metadata = json.load(file)
        self._sample_list = metadata["_sample_list"]
        self.len_storage = metadata["len_storage"]
        self.drop_last = metadata["drop_last"]
        self._ran_out = metadata["_ran_out"]

    def _get_sample_list(self, storage: Storage, len_storage: int):
        if storage is None:
            device = self._sample_list.device
        else:
            device = storage.device if hasattr(storage, "device") else None
        if self.shuffle:
            self._sample_list = torch.randperm(len_storage, device=device)
        else:
            self._sample_list = torch.arange(len_storage, device=device)

    def _single_sample(self, len_storage, batch_size):
        index = self._sample_list[:batch_size]
        self._sample_list = self._sample_list[batch_size:]

        # check if we have enough elements for one more batch, assuming same batch size
        # will be used each time sample is called
        if self._sample_list.numel() == 0 or (
            self.drop_last and len(self._sample_list) < batch_size
        ):
            self.ran_out = True
            self._get_sample_list(storage=None, len_storage=len_storage)
        else:
            self.ran_out = False
        return index

    def _storage_len(self, storage):
        return len(storage)

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        len_storage = self._storage_len(storage)
        if len_storage == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        if not len_storage:
            raise RuntimeError("An empty storage was passed")
        if self.len_storage != len_storage or self._sample_list is None:
            self._get_sample_list(storage, len_storage)
        if len_storage < batch_size and self.drop_last:
            raise ValueError(
                f"The batch size ({batch_size}) is greater than the storage capacity ({len_storage}). "
                "This makes it impossible to return a sample without repeating indices. "
                "Consider changing the sampler class or turn the 'drop_last' argument to False."
            )
        self.len_storage = len_storage
        index = self._single_sample(len_storage, batch_size)
        # we 'always' return the indices. The 'drop_last' just instructs the
        # sampler to turn to 'ran_out = True` whenever the next sample
        # will be too short. This will be read by the replay buffer
        # as a signal for an early break of the __iter__().
        return index, {}

    @property
    def ran_out(self):
        return self._ran_out

    @ran_out.setter
    def ran_out(self, value):
        self._ran_out = value

    def _empty(self):
        self._sample_list = None
        self.len_storage = 0
        self._ran_out = False

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError


class SliceSampler(Sampler):
    """Samples slices of data along the first dimension, given start and stop signals.

    This class samples sub-trajectories with replacement. For a version without
    replacement, see :class:`~minimal_torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement`.

    Keyword Args:
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        cache_values (bool, optional): to be used with static datasets.
            Will cache the start and end signal of the trajectory.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~minimal_torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~minimal_torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`minimal_torchrl.collectors.split_trajectories`. Defaults to ``True``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~minimal_torchrl.data.replay_buffers.samplers.SliceSampler` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.
    """

    def __init__(
        self,
        *,
        num_slices: int = None,
        slice_len: int = None,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
    ) -> object:
        if end_key is None:
            end_key = ("next", "done")
        if traj_key is None:
            traj_key = "episode"
        if not ((num_slices is None) ^ (slice_len is None)):
            raise TypeError(
                "Either num_slices or slice_len must be not None, and not both. "
                f"Got num_slices={num_slices} and slice_len={slice_len}."
            )
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.end_key = end_key
        self.traj_key = traj_key
        self.truncated_key = truncated_key
        self.cache_values = cache_values
        self._fetch_traj = True
        self._uses_data_prefix = False
        self.strict_length = strict_length
        self._cache = {}

    @staticmethod
    def _find_start_stop_traj(*, trajectory=None, end=None):
        if trajectory is not None:
            # slower
            # _, stop_idx = torch.unique_consecutive(trajectory, return_counts=True)
            # stop_idx = stop_idx.cumsum(0) - 1

            # even slower
            # t = trajectory.unsqueeze(0)
            # w = torch.tensor([1, -1], dtype=torch.int).view(1, 1, 2)
            # stop_idx = torch.conv1d(t, w).nonzero()

            # faster
            end = trajectory[:-1] != trajectory[1:]
            end = torch.cat([end, torch.ones_like(end[:1])], 0)
        else:
            end = torch.index_fill(
                end,
                index=torch.tensor(-1, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if end.ndim != 1:
            raise RuntimeError(
                f"Expected the end-of-trajectory signal to be 1-dimensional. Got a {end.ndim} tensor instead."
            )
        stop_idx = end.view(-1).nonzero().view(-1)
        start_idx = torch.cat([torch.zeros_like(stop_idx[:1]), stop_idx[:-1] + 1])
        lengths = stop_idx - start_idx + 1
        return start_idx, stop_idx, lengths

    def _tensor_slices_from_startend(self, seq_length, start):
        if isinstance(seq_length, int):
            return (
                torch.arange(
                    seq_length, device=start.device, dtype=start.dtype
                ).unsqueeze(0)
                + start.unsqueeze(1)
            ).view(-1)
        else:
            # when padding is needed
            return torch.cat(
                [
                    _start
                    + torch.arange(_seq_len, device=start.device, dtype=start.dtype)
                    for _start, _seq_len in zip(start, seq_length)
                ]
            )

    def _get_stop_and_length(self, storage, fallback=True):
        if self.cache_values and "stop-and-length" in self._cache:
            return self._cache.get("stop-and-length")

        if self._fetch_traj:
            # We first try with the traj_key
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                try:
                    trajectory = storage._storage.get(self._used_traj_key)
                except KeyError:
                    trajectory = storage._storage.get(("_data", self.traj_key))
                    # cache that value for future use
                    self._used_traj_key = ("_data", self.traj_key)
                self._uses_data_prefix = (
                    isinstance(self._used_traj_key, tuple)
                    and self._used_traj_key[0] == "_data"
                )
                vals = self._find_start_stop_traj(trajectory=trajectory[: len(storage)])
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = False
                    return self._get_stop_and_length(storage, fallback=False)
                raise

        else:
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                try:
                    done = storage._storage.get(self._used_end_key)
                except KeyError:
                    done = storage._storage.get(("_data", self.end_key))
                    # cache that value for future use
                    self._used_end_key = ("_data", self.end_key)
                self._uses_data_prefix = (
                    isinstance(self._used_end_key, tuple)
                    and self._used_end_key[0] == "_data"
                )
                vals = self._find_start_stop_traj(end=done.squeeze())[: len(storage)]
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = True
                    return self._get_stop_and_length(storage, fallback=False)
                raise

    def _adjusted_batch_size(self, batch_size):
        if self.num_slices is not None:
            if batch_size % self.num_slices != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the number of slices, got batch_size={batch_size} and num_slices={self.num_slices}."
                )
            seq_length = batch_size // self.num_slices
            num_slices = self.num_slices
        else:
            if batch_size % self.slice_len != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the slice length, got batch_size={batch_size} and slice_len={self.slice_len}."
                )
            seq_length = self.slice_len
            num_slices = batch_size // self.slice_len
        return seq_length, num_slices

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        if not isinstance(storage, TensorStorage):
            raise RuntimeError(
                f"{type(self)} can only sample from TensorStorage subclasses, got {type(storage)} instead."
            )

        # pick up as many trajs as we need
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        return self._sample_slices(lengths, start_idx, stop_idx, seq_length, num_slices)

    def _sample_slices(
        self, lengths, start_idx, stop_idx, seq_length, num_slices, traj_idx=None
    ) -> Tuple[torch.Tensor, dict]:
        if traj_idx is None:
            traj_idx = torch.randint(
                lengths.shape[0], (num_slices,), device=lengths.device
            )
        else:
            num_slices = traj_idx.shape[0]

        if (lengths < seq_length).any():
            if self.strict_length:
                raise RuntimeError(
                    f"Some stored trajectories have a length shorter than the slice that was asked for. "
                    "Create the sampler with `strict_length=False` to allow shorter trajectories to appear "
                    f"in you batch: {lengths}"
                )
            # make seq_length a tensor with values clamped by lengths
            seq_length = lengths[traj_idx].clamp_max(seq_length)

        relative_starts = (
            (
                torch.rand(num_slices, device=lengths.device)
                * (lengths[traj_idx] - seq_length + 1)
            )
            .floor()
            .to(start_idx.dtype)
        )
        starts = start_idx[traj_idx] + relative_starts
        index = self._tensor_slices_from_startend(seq_length, starts)
        if self.truncated_key is not None:
            truncated_key = self.truncated_key
            done_key = _replace_last(truncated_key, "done")
            terminated_key = _replace_last(truncated_key, "terminated")

            truncated = torch.zeros(
                (*index.shape, 1), dtype=torch.bool, device=index.device
            )
            if isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            traj_terminated = stop_idx[traj_idx] == start_idx[traj_idx] + seq_length - 1
            terminated = torch.zeros_like(truncated)
            if terminated.any():
                if isinstance(seq_length, int):
                    truncated.view(num_slices, -1)[traj_terminated] = 1
                else:
                    truncated[(seq_length.cumsum(0) - 1)[traj_terminated]] = 1
            truncated = truncated & ~terminated
            done = terminated | truncated
            return index.to(torch.long), {
                truncated_key: truncated,
                done_key: done,
                terminated_key: terminated,
            }
        return index.to(torch.long), {}

    @property
    def _used_traj_key(self):
        return self.__dict__.get("__used_traj_key", self.traj_key)

    @_used_traj_key.setter
    def _used_traj_key(self, value):
        self.__dict__["__used_traj_key"] = value

    @property
    def _used_end_key(self):
        return self.__dict__.get("__used_end_key", self.end_key)

    @_used_end_key.setter
    def _used_end_key(self, value):
        self.__dict__["__used_end_key"] = value

    def _empty(self):
        pass

    def dumps(self, path):
        # no op - cache does not need to be saved
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_cache"] = {}
        return state