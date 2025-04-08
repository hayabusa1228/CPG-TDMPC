# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import heapq
import json
# import textwrap
from abc import ABC, abstractmethod
from copy import copy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
# from tensordict import is_tensor_collection, MemoryMappedTensor
# from tensordict.utils import _STRDTYPE2DTYPE
from torch import multiprocessing as mp

from .storage import Storage


class Writer(ABC):
    """A ReplayBuffer base Writer class."""

    def __init__(self) -> None:
        self._storage = None

    def register_storage(self, storage: Storage) -> None:
        self._storage = storage

    @abstractmethod
    def add(self, data: Any) -> int:
        """Inserts one piece of data at an appropriate index, and returns that index."""
        ...

    @abstractmethod
    def extend(self, data: Sequence) -> torch.Tensor:
        """Inserts a series of data points at appropriate indices, and returns a tensor containing the indices."""
        ...

    @abstractmethod
    def _empty(self):
        ...

    @abstractmethod
    def dumps(self, path):
        ...

    @abstractmethod
    def loads(self, path):
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


class ImmutableDatasetWriter(Writer):
    """A blocking writer for immutable datasets."""

    WRITING_ERR = "This dataset doesn't allow writing."

    def add(self, data: Any) -> int:
        raise RuntimeError(self.WRITING_ERR)

    def extend(self, data: Sequence) -> torch.Tensor:
        raise RuntimeError(self.WRITING_ERR)

    def _empty(self):
        raise RuntimeError(self.WRITING_ERR)

    def dumps(self, path):
        ...

    def loads(self, path):
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return


class RoundRobinWriter(Writer):
    """A RoundRobin Writer class for composable replay buffers."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._cursor = 0

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        with open(path / "metadata.json", "w") as file:
            json.dump({"cursor": self._cursor}, file)

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "metadata.json", "r") as file:
            metadata = json.load(file)
            self._cursor = metadata["cursor"]

    def add(self, data: Any) -> int:
        ret = self._cursor
        _cursor = self._cursor
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (self._cursor + 1) % self._storage.max_size
        self._storage[_cursor] = data
        return ret

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        batch_size = len(data)
        index = np.arange(cur_size, batch_size + cur_size) % self._storage.max_size
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (batch_size + cur_size) % self._storage.max_size
        self._storage[index] = data
        return index

    def state_dict(self) -> Dict[str, Any]:
        return {"_cursor": self._cursor}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cursor = state_dict["_cursor"]

    def _empty(self):
        self._cursor = 0

    @property
    def _cursor(self):
        _cursor_value = self.__dict__.get("_cursor_value", None)
        if _cursor_value is None:
            _cursor_value = self._cursor_value = mp.Value("i", 0)
        return _cursor_value.value

    @_cursor.setter
    def _cursor(self, value):
        _cursor_value = self.__dict__.get("_cursor_value", None)
        if _cursor_value is None:
            _cursor_value = self._cursor_value = mp.Value("i", 0)
        _cursor_value.value = value

    def __getstate__(self):
        state = copy(self.__dict__)
        if get_spawning_popen() is None:
            cursor = self._cursor
            del state["_cursor_value"]
            state["cursor__context"] = cursor
        return state

    def __setstate__(self, state):
        cursor = state.pop("cursor__context", None)
        if cursor is not None:
            _cursor_value = mp.Value("i", cursor)
            state["_cursor_value"] = _cursor_value
        self.__dict__.update(state)

