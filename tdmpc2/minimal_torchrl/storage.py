# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
import logging
import os
from distutils.util import strtobool
import textwrap
import warnings
from collections import OrderedDict
from copy import copy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
import typing
from typing import Any, Dict, List, Sequence, Union, overload

import numpy as np
import torch
from tensordict import is_tensorclass
from tensordict.memmap import MemmapTensor
from tensordict.tensordict import is_tensor_collection, TensorDict, TensorDictBase
from tensordict.utils import expand_right
from torch import multiprocessing as mp


import os

import tempfile
from pathlib import Path

import numpy as np
import torch

from tensordict.utils import implement_for

from torch import distributed as dist

# from torch.multiprocessing.reductions import ForkingPickler

try:
    if dist.is_available():
        from torch.distributed._tensor.api import DTensor
    else:
        raise ImportError
except ImportError:

    class DTensor(torch.Tensor):  # noqa: D101
        ...

try:
    from torchsnapshot.serialization import tensor_from_memoryview

    _has_ts = True
except ImportError:
    _has_ts = False

VERBOSE = strtobool(os.environ.get("VERBOSE", "0"))

INT_CLASSES_TYPING = Union[int, np.integer]
if hasattr(typing, "get_args"):
    INT_CLASSES = typing.get_args(INT_CLASSES_TYPING)
else:
    # python 3.7
    INT_CLASSES = (int, np.integer)

_STRDTYPE2DTYPE = {
    str(dtype): dtype
    for dtype in (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.quint4x2,
    )
}


def _reset_batch_size(x):
    """Resets the batch size of a tensordict.

    In some cases we save the original shape of the tensordict as a tensor (or memmap tensor).

    This function will read that tensor, extract its items and reset the shape
    of the tensordict to it. If items have an incompatible shape (e.g. "index")
    they will be expanded to the right to match it.

    """
    shape = x.get("_rb_batch_size", None)
    if shape is not None:
        warnings.warn(
            "Reshaping nested tensordicts will be deprecated soon.",
            category=DeprecationWarning,
        )
        data = x.get("_data")
        # we need to reset the batch-size
        if isinstance(shape, MemmapTensor):
            shape = shape.as_tensor()
        locked = data.is_locked
        if locked:
            data.unlock_()
        shape = [s.item() for s in shape[0]]
        shape = torch.Size([x.shape[0], *shape])
        # we may need to update some values in the data
        for key, value in x.items():
            if value.ndim >= len(shape):
                continue
            value = expand_right(value, shape)
            data.set(key, value)
        if locked:
            data.lock_()
        return data
    data = x.get("_data", None)
    if data is not None:
        return data
    return x


def _proc_args_const(*args, **kwargs):
    if len(args) > 0:
        # then the first (or the N first) args are the shape
        if len(args) == 1 and not isinstance(args[0], int):
            shape = torch.Size(args[0])
        else:
            shape = torch.Size(args)
    else:
        # we should have a "shape" keyword arg
        shape = kwargs.pop("shape", None)
        if shape is None:
            raise TypeError("Could not find the shape argument in the arguments.")
        shape = torch.Size(shape)
    return (
        shape,
        kwargs.pop("device", None),
        kwargs.pop("dtype", None),
        kwargs.pop("fill_value", None),
        kwargs.pop("filename", None),
    )

class MemoryMappedTensor(torch.Tensor):
    """A Memory-mapped Tensor.

    Supports filenames or file handlers.

    The main advantage of MemoryMappedTensor resides in its serialization methods,
    which ensure that the tensor is passed through queues or RPC remote calls without
    any copy.

    .. note::
      When used within RPC settings, the filepath should be accessible to both nodes.
      If it isn't the behaviour of passing a MemoryMappedTensor from one worker
      to another is undefined.

    MemoryMappedTensor supports multiple construction methods.

    Examples:
          >>> # from an existing tensor
          >>> tensor = torch.randn(3)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=file.name)
          ...     assert memmap_tensor.filename is not None
          >>> # if no filename is passed, a handler is used
          >>> tensor = torch.randn(3)
          >>> memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=file.name)
          >>> assert memmap_tensor.filename is None
          >>> # one can create an empty tensor too
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor_empty = MemoryMappedTensor.empty_like(tensor, filename=file.name)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor_zero = MemoryMappedTensor.zeros_like(tensor, filename=file.name)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor = MemoryMappedTensor.ones_like(tensor, filename=file.name)
    """

    # _filename: str | Path
    # _handler: _FileHandler
    _filename: None
    _handler: None
    _clear: bool
    index: Any
    parent_shape: torch.Size

    def __new__(
        cls,
        tensor_or_file,
        *,
        dtype=None,
        shape=None,
        index=None,
        device=None,
        handler=None,
    ):
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError(f"{cls} device must be cpu!")
        if isinstance(tensor_or_file, str):
            return cls.from_filename(
                tensor_or_file,
                dtype,
                shape,
                index,
            )
        elif handler is not None:
            return cls.from_handler(
                handler,
                dtype,
                shape,
                index,
            )
        return super().__new__(cls, tensor_or_file)

    def __init__(
        self, tensor_or_file, handler=None, dtype=None, shape=None, device=None
    ):
        ...

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def from_tensor(
        cls,
        input,
        *,
        filename=None,
        existsok=False,
        copy_existing=False,
        copy_data=True,
    ):
        """Creates a MemoryMappedTensor with the same content as another tensor.

        If the tensor is already a MemoryMappedTensor the original tensor is
        returned if the `filename` argument is `None` or if the two paths match.
        In all other cases, a new :class:`MemoryMappedTensor` is produced.

        Args:
            input (torch.Tensor): the tensor which content must be copied onto
                the MemoryMappedTensor.
            filename (path to a file): the path to the file where the tensor
                should be stored. If none is provided, a file handler is used
                instead.
            existsok (bool, optional): if ``True``, the file will overwrite
                an existing file. Defaults to ``False``.
            copy_existing (bool, optional): if ``True`` and the provided input
                is a MemoryMappedTensor with an associated filename, copying
                the content to the new location is permitted. Otherwise an
                exception is thown. This behaviour exists to prevent
                unadvertedly duplicating data on disk.
            copy_data (bool, optional): if ``True``, the content of the tensor
                will be copied on the storage. Defaults to ``True``.

        """
        if isinstance(input, MemoryMappedTensor):
            if (filename is None and input._filename is None) or (
                input._filename is not None
                and filename is not None
                and Path(filename).absolute() == Path(input.filename).absolute()
            ):
                # either location was not specified, or memmap is already in the
                # correct location, so just return the MemmapTensor unmodified
                return input
            elif not copy_existing and (
                input._filename is not None
                and filename is not None
                and Path(filename).absolute() != Path(input.filename).absolute()
            ):
                raise RuntimeError(
                    f"A filename was provided but the tensor already has a file associated "
                    f"({input.filename}). "
                    f"To copy the tensor onto the new location, pass copy_existing=True."
                )
        elif isinstance(input, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemoryMappedTensor.from_tensor."
            )
        if input.requires_grad:
            raise RuntimeError(
                "MemoryMappedTensor.from_tensor is incompatible with tensor.requires_grad."
            )
        shape = input.shape
        if filename is None:
            if input.dtype.is_floating_point:
                size = torch.finfo(input.dtype).bits // 8 * shape.numel()
            elif input.dtype.is_complex:
                raise ValueError(
                    "Complex-valued tensors are not supported by MemoryMappedTensor."
                )
            elif input.dtype == torch.bool:
                size = shape.numel()
            else:
                # assume integer
                size = torch.iinfo(input.dtype).bits // 8 * shape.numel()
            # handler = _FileHandler(size)
            handler = None
            out = torch.frombuffer(memoryview(handler.buffer), dtype=input.dtype)
            out = out.view(shape)
            out = cls(out)
        else:
            handler = None
            if not existsok and os.path.exists(str(filename)):
                raise RuntimeError(f"The file {filename} already exists.")
            out = cls(
                torch.from_file(
                    str(filename), shared=True, dtype=input.dtype, size=shape.numel()
                ).view(input.shape)
            )
        out._handler = handler
        out._filename = filename
        out.index = None
        out.parent_shape = input.shape
        if copy_data:
            if isinstance(input, DTensor):
                input = input.full_tensor()
            out.copy_(input)
        return out

    @property
    def filename(self):
        """The filename of the tensor, if it has one.

        Raises an exception otherwise.
        """
        filename = self._filename
        if filename is None:
            raise RuntimeError("The MemoryMappedTensor has no file associated.")
        return filename

    @classmethod
    def empty_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with no content but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        )

    @classmethod
    def full_like(cls, input, fill_value, *, filename=None):
        # noqa: D417
        """Creates a tensor with a single content indicated by the `fill_value` argument, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.
            fill_value (float or equivalent): content of the tensor.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(fill_value)

    @classmethod
    def zeros_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with a 0-filled content, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(0.0)

    @classmethod
    def ones_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with a 1-filled content, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.ones((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(1.0)

    @classmethod
    @overload
    def ones(cls, *size, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def ones(cls, shape, *, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def ones(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a 1-filled content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.ones((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        return cls.from_tensor(
            result,
            filename=filename,
        )

    @classmethod
    @overload
    def zeros(cls, *size, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def zeros(cls, shape, *, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def zeros(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a 0-filled content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(
            result,
            filename=filename,
        )
        return result

    @classmethod
    @overload
    def empty(cls, *size, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def empty(cls, shape, *, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def empty(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with empty content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(result, filename=filename)
        return result

    @classmethod
    @overload
    def full(cls, *size, fill_value, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def full(cls, shape, *, fill_value, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def full(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a single content specified by `fill_value`, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            fill_value (float or equivalent): content of the tensor.
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, fill_value, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device).fill_(fill_value)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        return cls.from_tensor(result, filename=filename)

    @classmethod
    def from_filename(cls, filename, dtype, shape, index=None):
        # noqa: D417
        """Loads a MemoryMappedTensor from a given filename.

        Args:
            filename (path or equivalent): the path to the file.
            dtype (torch.dtype): the dtype of the tensor.
            shape (integers or torch.Size): the shape of the tensor.
            index (torch-compatible index type): an index to use to build the
                tensor.

        """
        shape = torch.Size(shape)
        tensor = torch.from_file(
            str(filename), shared=True, dtype=dtype, size=shape.numel()
        ).view(shape)
        if index is not None:
            tensor = tensor[index]
        out = cls(tensor)
        out._filename = filename
        out._handler = None
        out.index = index
        out.parent_shape = shape
        return out

    @classmethod
    def from_handler(cls, handler, dtype, shape, index):
        # noqa: D417
        """Loads a MemoryMappedTensor from a given handler.

        Args:
            handler (compatible file handler): the handler for the tensor.
            dtype (torch.dtype): the dtype of the tensor.
            shape (integers or torch.Size): the shape of the tensor.
            index (torch-compatible index type): an index to use to build the
                tensor.

        """
        shape = torch.Size(shape)
        out = torch.frombuffer(memoryview(handler.buffer), dtype=dtype)
        out = torch.reshape(out, shape)
        if index is not None:
            out = out[index]
        out = cls(out)
        out._filename = None
        out._handler = handler
        out.index = index
        out.parent_shape = shape
        return out

    @property
    def _tensor(self):
        # for bc-compatibility with MemmapTensor, to be deprecated in v0.4
        return self

    def __setstate__(self, state):
        if "filename" in state:
            self.__dict__ = type(self).from_filename(**state).__dict__
        else:
            self.__dict__ = type(self).from_handler(**state).__dict__

    def __getstate__(self):
        if getattr(self, "_handler", None) is not None:
            return {
                "handler": self._handler,
                "dtype": self.dtype,
                "shape": self.parent_shape,
                "index": self.index,
            }
        elif getattr(self, "_filename", None) is not None:
            return {
                "filename": self._filename,
                "dtype": self.dtype,
                "shape": self.parent_shape,
                "index": self.index,
            }
        else:
            raise RuntimeError("Could not find handler or filename.")

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
        if getattr(self, "_handler", None) is not None:
            return type(self).from_handler, (
                self._handler,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        elif getattr(self, "_filename", None) is not None:
            return type(self).from_filename, (
                self._filename,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        else:
            raise RuntimeError("Could not find handler or filename.")

    @implement_for("torch", "2.0", None)
    def __getitem__(self, item):
        try:
            out = super().__getitem__(item)
        except ValueError as err:
            if "is unbound" in str(err):
                raise ValueError(
                    "Using first class dimension indices with MemoryMappedTensor "
                    "isn't supported at the moment."
                ) from err
            raise
        if out.untyped_storage().data_ptr() == self.untyped_storage().data_ptr():
            out = MemoryMappedTensor(out)
            out._handler = self._handler
            out._filename = self._filename
            out.index = item
            out.parent_shape = self.parent_shape
        return out

    @implement_for("torch", None, "2.0")
    def __getitem__(self, item):  # noqa: F811
        try:
            out = super().__getitem__(item)
        except ValueError as err:
            if "is unbound" in str(err):
                raise ValueError(
                    "Using first class dimension indices with MemoryMappedTensor "
                    "isn't supported at the moment."
                ) from err
            raise
        if out.storage().data_ptr() == self.storage().data_ptr():
            out = MemoryMappedTensor(out)
            out._handler = self._handler
            out._filename = self._filename
            out.index = item
            out.parent_shape = self.parent_shape
        return out


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    def __init__(self, max_size: int) -> None:
        self.max_size = int(max_size)

    @property
    def _attached_entities(self):
        # RBs that use a given instance of Storage should add
        # themselves to this set.
        _attached_entities = self.__dict__.get("_attached_entities_set", None)
        if _attached_entities is None:
            _attached_entities = set()
            self.__dict__["_attached_entities_set"] = _attached_entities
        return _attached_entities

    @abc.abstractmethod
    def set(self, cursor: int, data: Any):
        ...

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        ...

    @abc.abstractmethod
    def dumps(self, path):
        ...

    @abc.abstractmethod
    def loads(self, path):
        ...

    def attach(self, buffer: Any) -> None:
        """This function attaches a sampler to this storage.

        Buffers that read from this storage must be included as an attached
        entity by calling this method. This guarantees that when data
        in the storage changes, components are made aware of changes even if the storage
        is shared with other buffers (eg. Priority Samplers).

        Args:
            buffer: the object that reads from this storage.
        """
        self._attached_entities.add(buffer)

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, index, value):
        ret = self.set(index, value)
        for ent in self._attached_entities:
            ent.mark_update(index)
        return ret

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __len__(self):
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def _empty(self):
        ...


class ListStorage(Storage):
    """A storage stored in a list.

    Args:
        max_size (int): the maximum number of elements stored in the storage.

    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._storage = []

    def dumps(self, path):
        raise NotImplementedError(
            "ListStorage doesn't support serialization via `dumps` - `loads` API."
        )

    def loads(self, path):
        raise NotImplementedError(
            "ListStorage doesn't support serialization via `dumps` - `loads` API."
        )

    def set(self, cursor: Union[int, Sequence[int], slice], data: Any):
        if not isinstance(cursor, INT_CLASSES):
            if (isinstance(cursor, torch.Tensor) and cursor.numel() <= 1) or (
                isinstance(cursor, np.ndarray) and cursor.size <= 1
            ):
                self.set(int(cursor), data)
                return
            if isinstance(cursor, slice):
                self._storage[cursor] = data
                return
            for _cursor, _data in zip(cursor, data):
                self.set(_cursor, _data)
            return
        else:
            if cursor > len(self._storage):
                raise RuntimeError(
                    "Cannot append data located more than one item away from "
                    f"the storage size: the storage size is {len(self)} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor >= self.max_size:
                raise RuntimeError(
                    f"Cannot append data to the list storage: "
                    f"maximum capacity is {self.max_size} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor == len(self._storage):
                self._storage.append(data)
            else:
                self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        if isinstance(index, (INT_CLASSES, slice)):
            return self._storage[index]
        else:
            return [self._storage[i] for i in index]

    def __len__(self):
        return len(self._storage)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": [
                elt if not hasattr(elt, "state_dict") else elt.state_dict()
                for elt in self._storage
            ]
        }

    def load_state_dict(self, state_dict):
        _storage = state_dict["_storage"]
        self._storage = []
        for elt in _storage:
            if isinstance(elt, torch.Tensor):
                self._storage.append(elt)
            elif isinstance(elt, (dict, OrderedDict)):
                self._storage.append(TensorDict({}, []).load_state_dict(elt))
            else:
                raise TypeError(
                    f"Objects of type {type(elt)} are not supported by ListStorage.load_state_dict"
                )

    def _empty(self):
        self._storage = []

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Cannot share a storage of type {type(self)} between processes."
            )
        state = copy(self.__dict__)
        return state


class TensorStorage(Storage):
    """A storage for tensors and tensordicts.

    Args:
        storage (tensor or TensorDict): the data buffer to be used.
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
            If "auto" is passed, the device is automatically gathered from the
            first batch of data passed. This is not enabled by default to avoid
            data placed on GPU by mistake, causing OOM issues.

    Examples:
        >>> data = TensorDict({
        ...     "some data": torch.randn(10, 11),
        ...     ("some", "nested", "data"): torch.randn(10, 11, 12),
        ... }, batch_size=[10, 11])
        >>> storage = TensorStorage(data)
        >>> len(storage)  # only the first dimension is considered as indexable
        10
        >>> storage.get(0)
        TensorDict(
            fields={
                some data: Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
                some: TensorDict(
                    fields={
                        nested: TensorDict(
                            fields={
                                data: Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([11]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([11]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([11]),
            device=None,
            is_shared=False)
        >>> storage.set(0, storage.get(0).zero_()) # zeros the data along index ``0``

    This class also supports tensorclass data.

    Examples:
        >>> from tensordict import tensorclass
        >>> @tensorclass
        ... class MyClass:
        ...     foo: torch.Tensor
        ...     bar: torch.Tensor
        >>> data = MyClass(foo=torch.randn(10, 11), bar=torch.randn(10, 11, 12), batch_size=[10, 11])
        >>> storage = TensorStorage(data)
        >>> storage.get(0)
        MyClass(
            bar=Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
            foo=Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
            batch_size=torch.Size([11]),
            device=None,
            is_shared=False)

    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._storage = None
        return super().__new__(cls)

    def __init__(self, storage, max_size=None, device="cpu"):
        if not ((storage is None) ^ (max_size is None)):
            if storage is None:
                raise ValueError("Expected storage to be non-null.")
            if max_size != storage.shape[0]:
                raise ValueError(
                    "The max-size and the storage shape mismatch: got "
                    f"max_size={max_size} for a storage of shape {storage.shape}."
                )
        elif storage is not None:
            max_size = storage.shape[0]
        super().__init__(max_size)
        self.initialized = storage is not None
        if self.initialized:
            self._len = max_size
        else:
            self._len = 0
        self.device = (
            torch.device(device)
            if device != "auto"
            else storage.device
            if storage is not None
            else "auto"
        )
        self._storage = storage

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        if not self.initialized:
            raise RuntimeError("Cannot save a non-initialized storage.")
        if isinstance(self._storage, torch.Tensor):
            try:
                MemoryMappedTensor.from_filename(
                    shape=self._storage.shape,
                    filename=path / "storage.memmap",
                    dtype=self._storage.dtype,
                ).copy_(self._storage)
            except FileNotFoundError:
                MemoryMappedTensor.from_tensor(
                    self._storage, filename=path / "storage.memmap", copy_existing=True
                )
            is_tensor = True
            dtype = str(self._storage.dtype)
            shape = list(self._storage.shape)
        else:
            # try to load the path and overwrite.
            self._storage.memmap(
                path, copy_existing=True, num_threads=torch.get_num_threads()
            )
            is_tensor = False
            dtype = None
            shape = None

        with open(path / "storage_metadata.json", "w") as file:
            json.dump(
                {
                    "is_tensor": is_tensor,
                    "dtype": dtype,
                    "shape": shape,
                    "len": self._len,
                },
                file,
            )

    def loads(self, path):
        with open(path / "storage_metadata.json", "r") as file:
            metadata = json.load(file)
        is_tensor = metadata["is_tensor"]
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        _len = metadata["len"]
        if dtype is not None:
            shape = torch.Size(shape)
            dtype = _STRDTYPE2DTYPE[dtype]
        if is_tensor:
            _storage = MemoryMappedTensor.from_filename(
                path / "storage.memmap", shape=shape, dtype=dtype
            ).clone()
        else:
            _storage = TensorDict.load_memmap(path)
        if not self.initialized:
            self._storage = _storage
            self.initialized = True
        else:
            self._storage.copy_(_storage)
        self._len = _len

    @property
    def _len(self):
        _len_value = self.__dict__.get("_len_value", None)
        if _len_value is None:
            _len_value = self._len_value = mp.Value("i", 0)
        return _len_value.value

    @_len.setter
    def _len(self, value):
        _len_value = self.__dict__.get("_len_value", None)
        if _len_value is None:
            _len_value = self._len_value = mp.Value("i", 0)
        _len_value.value = value

    def __getstate__(self):
        state = copy(self.__dict__)
        if get_spawning_popen() is None:
            len = self._len
            del state["_len_value"]
            state["len__context"] = len
        elif not self.initialized:
            # check that the storage is initialized
            raise RuntimeError(
                f"Cannot share a storage of type {type(self)} between processed if "
                f"it has not been initialized yet. Populate the buffer with "
                f"some data in the main process before passing it to the other "
                f"subprocesses (or create the buffer explicitely with a TensorStorage)."
            )
        else:
            # check that the content is shared, otherwise tell the user we can't help
            storage = self._storage
            STORAGE_ERR = "The storage must be place in shared memory or memmapped before being shared between processes."
            if is_tensor_collection(storage):
                if not storage.is_memmap() and not storage.is_shared():
                    raise RuntimeError(STORAGE_ERR)
            else:
                if (
                    not isinstance(storage, MemoryMappedTensor)
                    and not storage.is_shared()
                ):
                    raise RuntimeError(STORAGE_ERR)

        return state

    def __setstate__(self, state):
        len = state.pop("len__context", None)
        if len is not None:
            _len_value = mp.Value("i", len)
            state["_len_value"] = _len_value
        self.__dict__.update(state)

    def state_dict(self) -> Dict[str, Any]:
        _storage = self._storage
        if isinstance(_storage, torch.Tensor):
            pass
        elif is_tensor_collection(_storage):
            _storage = _storage.state_dict()
        elif _storage is None:
            _storage = {}
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by {type(self)}.state_dict"
            )
        return {
            "_storage": _storage,
            "initialized": self.initialized,
            "_len": self._len,
        }

    def load_state_dict(self, state_dict):
        _storage = copy(state_dict["_storage"])
        if isinstance(_storage, torch.Tensor):
            if isinstance(self._storage, torch.Tensor):
                self._storage.copy_(_storage)
            elif self._storage is None:
                self._storage = _storage
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        elif isinstance(_storage, (dict, OrderedDict)):
            if is_tensor_collection(self._storage):
                self._storage.load_state_dict(_storage)
            elif self._storage is None:
                self._storage = TensorDict({}, []).load_state_dict(_storage)
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by ListStorage.load_state_dict"
            )
        self.initialized = state_dict["initialized"]
        self._len = state_dict["_len"]

    # @implement_for("torch", "2.0", None)
    # def set(
    #     self,
    #     cursor: Union[int, Sequence[int], slice],
    #     data: Union[TensorDictBase, torch.Tensor],
    # ):
    #     if isinstance(cursor, INT_CLASSES):
    #         self._len = max(self._len, cursor + 1)
    #     else:
    #         self._len = max(self._len, max(cursor) + 1)
    #
    #     if not self.initialized:
    #         if not isinstance(cursor, INT_CLASSES):
    #             self._init(data[0])
    #         else:
    #             self._init(data)
    #     self._storage[cursor] = data
    #
    # @implement_for("torch", None, "2.0")
    # def set(  # noqa: F811
    #     self,
    #     cursor: Union[int, Sequence[int], slice],
    #     data: Union[TensorDictBase, torch.Tensor],
    # ):
    #     if isinstance(cursor, INT_CLASSES):
    #         self._len = max(self._len, cursor + 1)
    #     else:
    #         self._len = max(self._len, max(cursor) + 1)
    #
    #     if not self.initialized:
    #         if not isinstance(cursor, INT_CLASSES):
    #             self._init(data[0])
    #         else:
    #             self._init(data)
    #     if not isinstance(cursor, (*INT_CLASSES, slice)):
    #         if not isinstance(cursor, torch.Tensor):
    #             cursor = torch.tensor(cursor, dtype=torch.long)
    #         elif cursor.dtype != torch.long:
    #             cursor = cursor.to(dtype=torch.long)
    #         if len(cursor) > len(self._storage):
    #             warnings.warn(
    #                 "A cursor of length superior to the storage capacity was provided. "
    #                 "To accomodate for this, the cursor will be truncated to its last "
    #                 "element such that its length matched the length of the storage. "
    #                 "This may **not** be the optimal behaviour for your application! "
    #                 "Make sure that the storage capacity is big enough to support the "
    #                 "batch size provided."
    #             )
    #     self._storage[cursor] = data

    # @implement_for("torch", None, "2.0")
    def set(  # noqa: F811
        self,
        cursor: Union[int, Sequence[int], slice],
        data: Union[TensorDictBase, torch.Tensor],
    ):
        if isinstance(cursor, INT_CLASSES):
            self._len = max(self._len, cursor + 1)
        else:
            self._len = max(self._len, max(cursor) + 1)

        if not self.initialized:
            if not isinstance(cursor, INT_CLASSES):
                self._init(data[0])
            else:
                self._init(data)
        if not isinstance(cursor, (*INT_CLASSES, slice)):
            if not isinstance(cursor, torch.Tensor):
                cursor = torch.tensor(cursor, dtype=torch.long, device=self.device)
            elif cursor.dtype != torch.long:
                cursor = cursor.to(dtype=torch.long, device=self.device)
            if len(cursor) > len(self._storage):
                warnings.warn(
                    "A cursor of length superior to the storage capacity was provided. "
                    "To accomodate for this, the cursor will be truncated to its last "
                    "element such that its length matched the length of the storage. "
                    "This may **not** be the optimal behaviour for your application! "
                    "Make sure that the storage capacity is big enough to support the "
                    "batch size provided."
                )
        self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        if self._len < self.max_size:
            storage = self._storage[: self._len]
        else:
            storage = self._storage
        if not self.initialized:
            raise RuntimeError(
                "Cannot get an item from an unitialized LazyMemmapStorage"
            )
        out = storage[index]
        if is_tensor_collection(out):
            out = _reset_batch_size(out)
            return out  # .unlock_()
        return out

    def __len__(self):
        return self._len

    def _empty(self):
        # assuming that the data structure is the same, we don't need to to
        # anything if the cursor is reset to 0
        self._len = 0

    def _init(self):
        raise NotImplementedError(
            f"{type(self)} must be initialized during construction."
        )


class LazyTensorStorage(TensorStorage):
    """A pre-allocated tensor storage for tensors and tensordicts.

    Args:
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
            If "auto" is passed, the device is automatically gathered from the
            first batch of data passed. This is not enabled by default to avoid
            data placed on GPU by mistake, causing OOM issues.

    Examples:
        >>> data = TensorDict({
        ...     "some data": torch.randn(10, 11),
        ...     ("some", "nested", "data"): torch.randn(10, 11, 12),
        ... }, batch_size=[10, 11])
        >>> storage = LazyTensorStorage(100)
        >>> storage.set(range(10), data)
        >>> len(storage)  # only the first dimension is considered as indexable
        10
        >>> storage.get(0)
        TensorDict(
            fields={
                some data: Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
                some: TensorDict(
                    fields={
                        nested: TensorDict(
                            fields={
                                data: Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([11]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([11]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([11]),
            device=cpu,
            is_shared=False)
        >>> storage.set(0, storage.get(0).zero_()) # zeros the data along index ``0``

    This class also supports tensorclass data.

    Examples:
        >>> from tensordict import tensorclass
        >>> @tensorclass
        ... class MyClass:
        ...     foo: torch.Tensor
        ...     bar: torch.Tensor
        >>> data = MyClass(foo=torch.randn(10, 11), bar=torch.randn(10, 11, 12), batch_size=[10, 11])
        >>> storage = LazyTensorStorage(10)
        >>> storage.set(range(10), data)
        >>> storage.get(0)
        MyClass(
            bar=Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
            foo=Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
            batch_size=torch.Size([11]),
            device=cpu,
            is_shared=False)

    """

    def __init__(self, max_size, device="cpu"):
        super().__init__(storage=None, max_size=max_size, device=device)

    def _init(self, data: Union[TensorDictBase, torch.Tensor]) -> None:
        if VERBOSE:
            logging.info("Creating a TensorStorage...")
        if self.device == "auto":
            self.device = data.device
        if isinstance(data, torch.Tensor):
            # if Tensor, we just create a MemoryMappedTensor of the desired shape, device and dtype
            out = torch.empty(
                self.max_size,
                *data.shape,
                device=self.device,
                dtype=data.dtype,
            )
        elif is_tensorclass(data):
            out = (
                data.expand(self.max_size, *data.shape).clone().zero_().to(self.device)
            )
        else:
            out = (
                data.expand(self.max_size, *data.shape)
                .to_tensordict()
                .zero_()
                .clone()
                .to(self.device)
            )

        self._storage = out
        self.initialized = True


# # # Copyright (c) Meta Platforms, Inc. and affiliates.
# # #
# # # This source code is licensed under the MIT license found in the
# # # LICENSE file in the root directory of this source tree.
# # from __future__ import annotations
# #
# # import collections
# # import json
# # import textwrap
# # import threading
# # import warnings
# # from concurrent.futures import ThreadPoolExecutor
# # from pathlib import Path
# # from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
# #
# # import numpy as np
# #
# # import torch
# #
# # from tensordict import is_tensorclass
# # from tensordict.tensordict import (
# #     is_tensor_collection,
# #     LazyStackedTensorDict,
# #     TensorDict,
# #     TensorDictBase,
# # )
# # from tensordict.utils import expand_as_right, expand_right
# # from torch import Tensor
# #
# # from minimal_torchrl._utils import accept_remote_rref_udf_invocation
# # from minimal_torchrl.data.replay_buffers.samplers import (
# #     PrioritizedSampler,
# #     RandomSampler,
# #     Sampler,
# #     SamplerEnsemble,
# # )
# # from minimal_torchrl.data.replay_buffers.storages import (
# #     _get_default_collate,
# #     ListStorage,
# #     Storage,
# #     StorageEnsemble,
# # )
# # from minimal_torchrl.data.replay_buffers.utils import (
# #     _to_numpy,
# #     _to_torch,
# #     INT_CLASSES,
# #     pin_memory_output,
# # )
# # from minimal_torchrl.data.replay_buffers.writers import (
# #     RoundRobinWriter,
# #     TensorDictRoundRobinWriter,
# #     Writer,
# #     WriterEnsemble,
# # )
# # from minimal_torchrl.data.utils import DEVICE_TYPING
#
#
# class ReplayBuffer:
#     """A generic, composable replay buffer class.
#
#     Keyword Args:
#         storage (Storage, optional): the storage to be used. If none is provided
#             a default :class:`~minimal_torchrl.data.replay_buffers.ListStorage` with
#             ``max_size`` of ``1_000`` will be created.
#         sampler (Sampler, optional): the sampler to be used. If none is provided,
#             a default :class:`~minimal_torchrl.data.replay_buffers.RandomSampler`
#             will be used.
#         writer (Writer, optional): the writer to be used. If none is provided
#             a default :class:`~minimal_torchrl.data.replay_buffers.RoundRobinWriter`
#             will be used.
#         collate_fn (callable, optional): merges a list of samples to form a
#             mini-batch of Tensor(s)/outputs.  Used when using batched
#             loading from a map-style dataset. The default value will be decided
#             based on the storage type.
#         pin_memory (bool): whether pin_memory() should be called on the rb
#             samples.
#         prefetch (int, optional): number of next batches to be prefetched
#             using multithreading. Defaults to None (no prefetching).
#         transform (Transform, optional): Transform to be executed when
#             sample() is called.
#             To chain transforms use the :class:`~minimal_torchrl.envs.Compose` class.
#             Transforms should be used with :class:`tensordict.TensorDict`
#             content. If used with other structures, the transforms should be
#             encoded with a ``"data"`` leading key that will be used to
#             construct a tensordict from the non-tensordict content.
#         batch_size (int, optional): the batch size to be used when sample() is
#             called.
#             .. note::
#               The batch-size can be specified at construction time via the
#               ``batch_size`` argument, or at sampling time. The former should
#               be preferred whenever the batch-size is consistent across the
#               experiment. If the batch-size is likely to change, it can be
#               passed to the :meth:`~.sample` method. This option is
#               incompatible with prefetching (since this requires to know the
#               batch-size in advance) as well as with samplers that have a
#               ``drop_last`` argument.
#
#     Examples:
#         >>> import torch
#         >>>
#         >>> from minimal_torchrl.data import ReplayBuffer, ListStorage
#         >>>
#         >>> torch.manual_seed(0)
#         >>> rb = ReplayBuffer(
#         ...     storage=ListStorage(max_size=1000),
#         ...     batch_size=5,
#         ... )
#         >>> # populate the replay buffer and get the item indices
#         >>> data = range(10)
#         >>> indices = rb.extend(data)
#         >>> # sample will return as many elements as specified in the constructor
#         >>> sample = rb.sample()
#         >>> print(sample)
#         tensor([4, 9, 3, 0, 3])
#         >>> # Passing the batch-size to the sample method overrides the one in the constructor
#         >>> sample = rb.sample(batch_size=3)
#         >>> print(sample)
#         tensor([9, 7, 3])
#         >>> # one cans sample using the ``sample`` method or iterate over the buffer
#         >>> for i, batch in enumerate(rb):
#         ...     print(i, batch)
#         ...     if i == 3:
#         ...         break
#         0 tensor([7, 3, 1, 6, 6])
#         1 tensor([9, 8, 6, 6, 8])
#         2 tensor([4, 3, 6, 9, 1])
#         3 tensor([4, 4, 1, 9, 9])
#
#     Replay buffers accept *any* kind of data. Not all storage types
#     will work, as some expect numerical data only, but the default
#     :class:`minimal_torchrl.data.ListStorage` will:
#
#     Examples:
#         >>> torch.manual_seed(0)
#         >>> buffer = ReplayBuffer(storage=ListStorage(100), collate_fn=lambda x: x)
#         >>> indices = buffer.extend(["a", 1, None])
#         >>> buffer.sample(3)
#         [None, 'a', None]
#     """
#
#     def __init__(
#         self,
#         *,
#         storage: Storage | None = None,
#         sampler: Sampler | None = None,
#         writer: Writer | None = None,
#         collate_fn: Callable | None = None,
#         pin_memory: bool = False,
#         prefetch: int | None = None,
#         transform: "Transform" | None = None,  # noqa-F821
#         batch_size: int | None = None,
#     ) -> None:
#         self._storage = storage if storage is not None else ListStorage(max_size=1_000)
#         self._storage.attach(self)
#         self._sampler = sampler
#         self._writer = writer if writer is not None else RoundRobinWriter()
#         self._writer.register_storage(self._storage)
#
#         self._collate_fn = lambda x: x
#         self._pin_memory = pin_memory
#
#         self._prefetch = bool(prefetch)
#         self._prefetch_cap = prefetch or 0
#         self._prefetch_queue = collections.deque()
#         if self._prefetch_cap:
#             self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)
#
#         self._replay_lock = threading.RLock()
#         self._futures_lock = threading.RLock()
#         from minimal_torchrl.envs.transforms.transforms import Compose
#
#         if transform is None:
#             transform = Compose()
#         elif not isinstance(transform, Compose):
#             transform = Compose(transform)
#         transform.eval()
#         self._transform = transform
#
#         if batch_size is None and prefetch:
#             raise ValueError(
#                 "Dynamic batch-size specification is incompatible "
#                 "with multithreaded sampling. "
#                 "When using prefetch, the batch-size must be specified in "
#                 "advance. "
#             )
#         if (
#             batch_size is None
#             and hasattr(self._sampler, "drop_last")
#             and self._sampler.drop_last
#         ):
#             raise ValueError(
#                 "Samplers with drop_last=True must work with a predictible batch-size. "
#                 "Please pass the batch-size to the ReplayBuffer constructor."
#             )
#         self._batch_size = batch_size
#
#     def __len__(self) -> int:
#         with self._replay_lock:
#             return len(self._storage)
#
#     def __repr__(self) -> str:
#         return (
#             f"{type(self).__name__}("
#             f"storage={self._storage}, "
#             f"sampler={self._sampler}, "
#             f"writer={self._writer}"
#             ")"
#         )
#
#     @pin_memory_output
#     def __getitem__(self, index: Union[int, torch.Tensor]) -> Any:
#         index = _to_numpy(index)
#         with self._replay_lock:
#             data = self._storage[index]
#
#         if not isinstance(index, INT_CLASSES):
#             data = self._collate_fn(data)
#
#         if self._transform is not None and len(self._transform):
#             is_td = True
#             if not is_tensor_collection(data):
#                 data = TensorDict({"data": data}, [])
#                 is_td = False
#             data = self._transform(data)
#             if not is_td:
#                 data = data["data"]
#
#         return data
#
#     def state_dict(self) -> Dict[str, Any]:
#         return {
#             "_storage": self._storage.state_dict(),
#             "_sampler": self._sampler.state_dict(),
#             "_writer": self._writer.state_dict(),
#             "_transforms": self._transform.state_dict(),
#             "_batch_size": self._batch_size,
#         }
#
#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         self._storage.load_state_dict(state_dict["_storage"])
#         self._sampler.load_state_dict(state_dict["_sampler"])
#         self._writer.load_state_dict(state_dict["_writer"])
#         self._transform.load_state_dict(state_dict["_transforms"])
#         self._batch_size = state_dict["_batch_size"]
#
#     def dumps(self, path):
#         """Saves the replay buffer on disk at the specified path.
#
#         Args:
#             path (Path or str): path where to save the replay buffer.
#
#         Examples:
#             >>> import tempfile
#             >>> import tqdm
#             >>> from minimal_torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
#             >>> from minimal_torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
#             >>> import torch
#             >>> from tensordict import TensorDict
#             >>> # Build and populate the replay buffer
#             >>> S = 1_000_000
#             >>> sampler = PrioritizedSampler(S, 1.1, 1.0)
#             >>> # sampler = RandomSampler()
#             >>> storage = LazyMemmapStorage(S)
#             >>> rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)
#             >>>
#             >>> for _ in tqdm.tqdm(range(100)):
#             ...     td = TensorDict({"obs": torch.randn(100, 3, 4), "next": {"obs": torch.randn(100, 3, 4)}, "td_error": torch.rand(100)}, [100])
#             ...     rb.extend(td)
#             ...     sample = rb.sample(32)
#             ...     rb.update_tensordict_priority(sample)
#             >>> # save and load the buffer
#             >>> with tempfile.TemporaryDirectory() as tmpdir:
#             ...     rb.dumps(tmpdir)
#             ...
#             ...     sampler = PrioritizedSampler(S, 1.1, 1.0)
#             ...     # sampler = RandomSampler()
#             ...     storage = LazyMemmapStorage(S)
#             ...     rb_load = TensorDictReplayBuffer(storage=storage, sampler=sampler)
#             ...     rb_load.loads(tmpdir)
#             ...     assert len(rb) == len(rb_load)
#
#         """
#         path = Path(path).absolute()
#         path.mkdir(exist_ok=True)
#         self._storage.dumps(path / "storage")
#         self._sampler.dumps(path / "sampler")
#         self._writer.dumps(path / "writer")
#         # fall back on state_dict for transforms
#         transform_sd = self._transform.state_dict()
#         if transform_sd:
#             torch.save(transform_sd, path / "transform.t")
#         with open(path / "buffer_metadata.json", "w") as file:
#             json.dump({"batch_size": self._batch_size}, file)
#
#     def loads(self, path):
#         """Loads a replay buffer state at the given path.
#
#         The buffer should have matching components and be saved using :meth:`~.dumps`.
#
#         Args:
#             path (Path or str): path where the replay buffer was saved.
#
#         See :meth:`~.dumps` for more info.
#
#         """
#         path = Path(path).absolute()
#         self._storage.loads(path / "storage")
#         self._sampler.loads(path / "sampler")
#         self._writer.loads(path / "writer")
#         # fall back on state_dict for transforms
#         if (path / "transform.t").exists():
#             self._transform.load_state_dict(torch.load(path / "transform.t"))
#         with open(path / "buffer_metadata.json", "r") as file:
#             metadata = json.load(file)
#         self._batch_size = metadata["batch_size"]
#
#     def add(self, data: Any) -> int:
#         """Add a single element to the replay buffer.
#
#         Args:
#             data (Any): data to be added to the replay buffer
#
#         Returns:
#             index where the data lives in the replay buffer.
#         """
#         if self._transform is not None and (
#             is_tensor_collection(data) or len(self._transform)
#         ):
#             data = self._transform.inv(data)
#         return self._add(data)
#
#     def _add(self, data):
#         with self._replay_lock:
#             index = self._writer.add(data)
#             self._sampler.add(index)
#         return index
#
#     def _extend(self, data: Sequence) -> torch.Tensor:
#         with self._replay_lock:
#             index = self._writer.extend(data)
#             self._sampler.extend(index)
#         return index
#
#     def extend(self, data: Sequence) -> torch.Tensor:
#         """Extends the replay buffer with one or more elements contained in an iterable.
#
#         If present, the inverse transforms will be called.`
#
#         Args:
#             data (iterable): collection of data to be added to the replay
#                 buffer.
#
#         Returns:
#             Indices of the data added to the replay buffer.
#         """
#         if self._transform is not None and (
#             is_tensor_collection(data) or len(self._transform)
#         ):
#             data = self._transform.inv(data)
#         return self._extend(data)
#
#     def update_priority(
#         self,
#         index: Union[int, torch.Tensor],
#         priority: Union[int, torch.Tensor],
#     ) -> None:
#         with self._replay_lock:
#             self._sampler.update_priority(index, priority)
#
#     @pin_memory_output
#     def _sample(self, batch_size: int) -> Tuple[Any, dict]:
#         with self._replay_lock:
#             index, info = self._sampler.sample(self._storage, batch_size)
#             info["index"] = index
#             data = self._storage.get(index)
#         if not isinstance(index, INT_CLASSES):
#             data = self._collate_fn(data)
#         if self._transform is not None and len(self._transform):
#             is_td = True
#             if not is_tensor_collection(data):
#                 data = TensorDict({"data": data}, [])
#                 is_td = False
#             is_locked = data.is_locked
#             if is_locked:
#                 data.unlock_()
#             data = self._transform(data)
#             if is_locked:
#                 data.lock_()
#             if not is_td:
#                 data = data["data"]
#
#         return data, info
#
#     def empty(self):
#         """Empties the replay buffer and reset cursor to 0."""
#         self._writer._empty()
#         self._sampler._empty()
#         self._storage._empty()
#
#     def sample(self, batch_size: int | None = None, return_info: bool = False) -> Any:
#         """Samples a batch of data from the replay buffer.
#
#         Uses Sampler to sample indices, and retrieves them from Storage.
#
#         Args:
#             batch_size (int, optional): size of data to be collected. If none
#                 is provided, this method will sample a batch-size as indicated
#                 by the sampler.
#             return_info (bool): whether to return info. If True, the result
#                 is a tuple (data, info). If False, the result is the data.
#
#         Returns:
#             A batch of data selected in the replay buffer.
#             A tuple containing this batch and info if return_info flag is set to True.
#         """
#         if (
#             batch_size is not None
#             and self._batch_size is not None
#             and batch_size != self._batch_size
#         ):
#             warnings.warn(
#                 f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
#                 f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
#                 "for a proper usage of the batch-size arguments. "
#                 "The batch-size provided to the sample method "
#                 "will prevail."
#             )
#         elif batch_size is None and self._batch_size is not None:
#             batch_size = self._batch_size
#         elif batch_size is None:
#             raise RuntimeError(
#                 "batch_size not specified. You can specify the batch_size when "
#                 "constructing the replay buffer, or pass it to the sample method. "
#                 "Refer to the ReplayBuffer documentation "
#                 "for a proper usage of the batch-size arguments."
#             )
#         if not self._prefetch:
#             ret = self._sample(batch_size)
#         else:
#             with self._futures_lock:
#                 while len(self._prefetch_queue) < self._prefetch_cap:
#                     fut = self._prefetch_executor.submit(self._sample, batch_size)
#                     self._prefetch_queue.append(fut)
#                 ret = self._prefetch_queue.popleft().result()
#
#         if return_info:
#             return ret
#         return ret[0]
#
#     def mark_update(self, index: Union[int, torch.Tensor]) -> None:
#         self._sampler.mark_update(index)
#
#     def append_transform(self, transform: "Transform") -> None:  # noqa-F821
#         """Appends transform at the end.
#
#         Transforms are applied in order when `sample` is called.
#
#         Args:
#             transform (Transform): The transform to be appended
#         """
#         transform.eval()
#         self._transform.append(transform)
#
#     def insert_transform(self, index: int, transform: "Transform") -> None:  # noqa-F821
#         """Inserts transform.
#
#         Transforms are executed in order when `sample` is called.
#
#         Args:
#             index (int): Position to insert the transform.
#             transform (Transform): The transform to be appended
#         """
#         transform.eval()
#         self._transform.insert(index, transform)
#
#     def __iter__(self):
#         if self._sampler.ran_out:
#             self._sampler.ran_out = False
#         if self._batch_size is None:
#             raise RuntimeError(
#                 "Cannot iterate over the replay buffer. "
#                 "Batch_size was not specified during construction of the replay buffer."
#             )
#         while not self._sampler.ran_out:
#             yield self.sample()
#
#     def __getstate__(self) -> Dict[str, Any]:
#         state = self.__dict__.copy()
#         _replay_lock = state.pop("_replay_lock", None)
#         _futures_lock = state.pop("_futures_lock", None)
#         if _replay_lock is not None:
#             state["_replay_lock_placeholder"] = None
#         if _futures_lock is not None:
#             state["_futures_lock_placeholder"] = None
#         return state
#
#     def __setstate__(self, state: Dict[str, Any]):
#         if "_replay_lock_placeholder" in state:
#             state.pop("_replay_lock_placeholder")
#             _replay_lock = threading.RLock()
#             state["_replay_lock"] = _replay_lock
#         if "_futures_lock_placeholder" in state:
#             state.pop("_futures_lock_placeholder")
#             _futures_lock = threading.RLock()
#             state["_futures_lock"] = _futures_lock
#         self.__dict__.update(state)
#
