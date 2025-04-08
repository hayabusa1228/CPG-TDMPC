# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import warnings
from copy import copy
from functools import wraps
from textwrap import indent
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from dataclasses import dataclass

import numpy as np

import torch

from tensordict import unravel_key, unravel_key_list
# from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.nn import dispatch, TensorDictModuleBase
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torch import nn, Tensor
# from torchrl._utils import _replace_last

# from torchrl.data.tensor_specs import (
#     CompositeSpec,
#     TensorSpec,
# )


@dataclass(repr=False)
class CompositeSpec:
    ...

@dataclass(repr=False)
class TensorSpec:
    ...


try:
    from torchvision.transforms.functional import center_crop

    try:
        from torchvision.transforms.functional import InterpolationMode, resize

        def interpolation_fn(interpolation):  # noqa: D103
            return InterpolationMode(interpolation)

    except ImportError:

        def interpolation_fn(interpolation):  # noqa: D103
            return interpolation

        from torchvision.transforms.functional_tensor import resize

    _has_tv = True
except ImportError:
    _has_tv = False

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent environment."


DEVICE_TYPING = Union[torch.device, str, int]


class _EnvPostInit(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance: EnvBase = super().__call__(*args, **kwargs)
        # we create the done spec by adding a done/terminated entry if one is missing
        instance._create_done_specs()
        # we access lazy attributed to make sure they're built properly.
        # This isn't done in `__init__` because we don't know if supre().__init__
        # will be called before or after the specs, batch size etc are set.
        _ = instance.done_spec
        _ = instance.reward_spec
        _ = instance.state_spec
        return instance


class EnvBase(nn.Module, metaclass=_EnvPostInit):
    """Abstract environment parent class.

    Methods:
        step (TensorDictBase -> TensorDictBase): step in the environment
        reset (TensorDictBase, optional -> TensorDictBase): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec
        rollout (Callable, ... -> TensorDictBase): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = GymEnv("Pendulum-v1")
        >>> env.batch_size  # how many envs are run at once
        torch.Size([])
        >>> env.input_spec
        CompositeSpec(
            full_state_spec: None,
            full_action_spec: CompositeSpec(
                action: BoundedTensorSpec(
                    shape=torch.Size([1]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
        >>> env.action_spec
        BoundedTensorSpec(
            shape=torch.Size([1]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> env.observation_spec
        CompositeSpec(
            observation: BoundedTensorSpec(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=cpu, shape=torch.Size([]))
        >>> env.reward_spec
        UnboundedContinuousTensorSpec(
            shape=torch.Size([1]),
            space=None,
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> env.done_spec
        DiscreteTensorSpec(
            shape=torch.Size([1]),
            space=DiscreteBox(n=2),
            device=cpu,
            dtype=torch.bool,
            domain=discrete)
        >>> # the output_spec contains all the expected outputs
        >>> env.output_spec
        CompositeSpec(
            full_reward_spec: CompositeSpec(
                reward: UnboundedContinuousTensorSpec(
                    shape=torch.Size([1]),
                    space=None,
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])),
            full_observation_spec: CompositeSpec(
                observation: BoundedTensorSpec(
                    shape=torch.Size([3]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])),
            full_done_spec: CompositeSpec(
                done: DiscreteTensorSpec(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

    """

    def __init__(
        self,
        device: DEVICE_TYPING = None,
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
        run_type_checks: bool = False,
        allow_done_after_reset: bool = False,
    ):
        if device is None:
            device = torch.device("cpu")
        self.__dict__.setdefault("_batch_size", None)
        if device is not None:
            self.__dict__["_device"] = torch.device(device)
            output_spec = self.__dict__.get("_output_spec", None)
            if output_spec is not None:
                self.__dict__["_output_spec"] = output_spec.to(self.device)
            input_spec = self.__dict__.get("_input_spec", None)
            if input_spec is not None:
                self.__dict__["_input_spec"] = input_spec.to(self.device)

        super().__init__()
        self.dtype = dtype_map.get(dtype, dtype)
        if "is_closed" not in self.__dir__():
            self.is_closed = True
        if batch_size is not None:
            # we want an error to be raised if we pass batch_size but
            # it's already been set
            self.batch_size = torch.Size(batch_size)
        self._run_type_checks = run_type_checks
        self._allow_done_after_reset = allow_done_after_reset

    @classmethod
    def __new__(cls, *args, _inplace_update=False, _batch_locked=True, **kwargs):
        # inplace update will write tensors in-place on the provided tensordict.
        # This is risky, especially if gradients need to be passed (in-place copy
        # for tensors that are part of computational graphs will result in an error).
        # It can also lead to inconsistencies when calling rollout.
        cls._inplace_update = _inplace_update
        cls._batch_locked = _batch_locked
        cls._device = None
        # cached in_keys to be excluded from update when calling step
        cls._cache_in_keys = None

        # We may assign _input_spec to the cls, but it must be assigned to the instance
        # we pull it off, and place it back where it belongs
        _input_spec = None
        if hasattr(cls, "_input_spec"):
            _input_spec = cls._input_spec.clone()
            delattr(cls, "_input_spec")
        _output_spec = None
        if hasattr(cls, "_output_spec"):
            _output_spec = cls._output_spec.clone()
            delattr(cls, "_output_spec")
        env = super().__new__(cls)
        if _input_spec is not None:
            env.__dict__["_input_spec"] = _input_spec
        if _output_spec is not None:
            env.__dict__["_output_spec"] = _output_spec
        return env

        return super().__new__(cls)

    def __setattr__(self, key, value):
        if key in (
            "_input_spec",
            "_observation_spec",
            "_action_spec",
            "_reward_spec",
            "_output_spec",
            "_state_spec",
            "_done_spec",
        ):
            raise AttributeError(
                "To set an environment spec, please use `env.observation_spec = obs_spec` (without the leading"
                " underscore)."
            )
        return super().__setattr__(key, value)

    @property
    def batch_locked(self) -> bool:
        """Whether the environment can be used with a batch size different from the one it was initialized with or not.

        If True, the env needs to be used with a tensordict having the same batch size as the env.
        batch_locked is an immutable property.
        """
        return self._batch_locked

    @batch_locked.setter
    def batch_locked(self, value: bool) -> None:
        raise RuntimeError("batch_locked is a read-only property")

    @property
    def run_type_checks(self) -> bool:
        return self._run_type_checks

    @run_type_checks.setter
    def run_type_checks(self, run_type_checks: bool) -> None:
        self._run_type_checks = run_type_checks

    @property
    def batch_size(self) -> torch.Size:
        _batch_size = self.__dict__["_batch_size"]
        if _batch_size is None:
            _batch_size = self._batch_size = torch.Size([])
        return _batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        self._batch_size = torch.Size(value)
        if (
            hasattr(self, "output_spec")
            and self.output_spec.shape[: len(value)] != value
        ):
            self.output_spec.unlock_()
            self.output_spec.shape = value
            self.output_spec.lock_()
        if hasattr(self, "input_spec") and self.input_spec.shape[: len(value)] != value:
            self.input_spec.unlock_()
            self.input_spec.shape = value
            self.input_spec.lock_()

    @property
    def device(self) -> torch.device:
        device = self.__dict__.get("_device", None)
        if device is None:
            device = self.__dict__["_device"] = torch.device("cpu")
        return device

    @device.setter
    def device(self, value: torch.device) -> None:
        device = self.__dict__.get("_device", None)
        if device is None:
            self.__dict__["_device"] = value
            return
        raise RuntimeError("device cannot be set. Call env.to(device) instead.")

    def ndimension(self):
        return len(self.batch_size)

    @property
    def ndim(self):
        return self.ndimension()

    # Parent specs: input and output spec.
    @property
    def input_spec(self) -> TensorSpec:
        """Input spec.

        The composite spec containing all specs for data input to the environments.

        It contains:

        - "full_action_spec": the spec of the input actions
        - "full_state_spec": the spec of all other environment inputs

        This attibute is locked and should be read-only.
        Instead, to set the specs contained in it, use the respective properties.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.input_spec
            CompositeSpec(
                full_state_spec: None,
                full_action_spec: CompositeSpec(
                    action: BoundedTensorSpec(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))


        """
        input_spec = self.__dict__.get("_input_spec", None)
        if input_spec is None:
            input_spec = CompositeSpec(
                full_state_spec=None,
                shape=self.batch_size,
                device=self.device,
            ).lock_()
            self.__dict__["_input_spec"] = input_spec
        return input_spec

    @input_spec.setter
    def input_spec(self, value: TensorSpec) -> None:
        raise RuntimeError("input_spec is protected.")

    @property
    def output_spec(self) -> TensorSpec:
        """Output spec.

        The composite spec containing all specs for data output from the environments.

        It contains:

        - "full_reward_spec": the spec of reward
        - "full_done_spec": the spec of done
        - "full_observation_spec": the spec of all other environment outputs

        This attibute is locked and should be read-only.
        Instead, to set the specs contained in it, use the respective properties.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.output_spec
            CompositeSpec(
                full_reward_spec: CompositeSpec(
                    reward: UnboundedContinuousTensorSpec(
                        shape=torch.Size([1]),
                        space=None,
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=cpu, shape=torch.Size([])),
                full_observation_spec: CompositeSpec(
                    observation: BoundedTensorSpec(
                        shape=torch.Size([3]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=cpu, shape=torch.Size([])),
                full_done_spec: CompositeSpec(
                    done: DiscreteTensorSpec(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))


        """
        output_spec = self.__dict__.get("_output_spec", None)
        if output_spec is None:
            output_spec = CompositeSpec(
                shape=self.batch_size,
                device=self.device,
            ).lock_()
            self.__dict__["_output_spec"] = output_spec
        return output_spec

    @output_spec.setter
    def output_spec(self, value: TensorSpec) -> None:
        raise RuntimeError("output_spec is protected.")

    @property
    def action_keys(self) -> List[NestedKey]:
        """The action keys of an environment.

        By default, there will only be one key named "action".

        Keys are sorted by depth in the data tree.
        """
        action_keys = self.__dict__.get("_action_keys", None)
        if action_keys is not None:
            return action_keys
        keys = self.input_spec["full_action_spec"].keys(True, True)
        if not len(keys):
            raise AttributeError("Could not find action spec")
        keys = sorted(keys, key=_repr_by_depth)
        self.__dict__["_action_keys"] = keys
        return keys

    @property
    def action_key(self) -> NestedKey:
        """The action key of an environment.

        By default, this will be "action".

        If there is more than one action key in the environment, this function will raise an exception.
        """
        if len(self.action_keys) > 1:
            raise KeyError(
                "action_key requested but more than one key present in the environment"
            )
        return self.action_keys[0]

    # Action spec: action specs belong to input_spec
    @property
    def action_spec(self) -> TensorSpec:
        """The ``action`` spec.

        The ``action_spec`` is always stored as a composite spec.

        If the action spec is provided as a simple spec, this will be returned.

            >>> env.action_spec = UnboundedContinuousTensorSpec(1)
            >>> env.action_spec
            UnboundedContinuousTensorSpec(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the action spec is provided as a composite spec and contains only one leaf,
        this function will return just the leaf.

            >>> env.action_spec = CompositeSpec({"nested": {"action": UnboundedContinuousTensorSpec(1)}})
            >>> env.action_spec
            UnboundedContinuousTensorSpec(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the action spec is provided as a composite spec and has more than one leaf,
        this function will return the whole spec.

            >>> env.action_spec = CompositeSpec({"nested": {"action": UnboundedContinuousTensorSpec(1), "another_action": DiscreteTensorSpec(1)}})
            >>> env.action_spec
            CompositeSpec(
                nested: CompositeSpec(
                    action: UnboundedContinuousTensorSpec(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    another_action: DiscreteTensorSpec(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=1),
                        device=cpu,
                        dtype=torch.int64,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        To retrieve the full spec passed, use:

            >>> env.input_spec["full_action_spec"]

        This property is mutable.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.action_spec
            BoundedTensorSpec(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)
        """
        try:
            action_spec = self.input_spec["full_action_spec"]
        except (KeyError, AttributeError):
            raise KeyError("Failed to find the action_spec.")

        if len(self.action_keys) > 1:
            out = action_spec
        else:
            try:
                out = action_spec[self.action_key]
            except KeyError:
                # the key may have changed
                raise KeyError(
                    "The action_key attribute seems to have changed. "
                    "This occurs when a action_spec is updated without "
                    "calling `env.action_spec = new_spec`. "
                    "Make sure you rely on this  type of command "
                    "to set the action and other specs."
                )

        return out

    @action_spec.setter
    def action_spec(self, value: TensorSpec) -> None:
        try:
            self.input_spec.unlock_()
            device = self.input_spec.device
            try:
                delattr(self, "_action_keys")
            except AttributeError:
                pass
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"action_spec of type {type(value)} do not have a shape attribute."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )

            if isinstance(value, CompositeSpec):
                for _ in value.values(True, True):  # noqa: B007
                    break
                else:
                    raise RuntimeError(
                        "An empty CompositeSpec was passed for the action spec. "
                        "This is currently not permitted."
                    )
            else:
                value = CompositeSpec(
                    action=value.to(device), shape=self.batch_size, device=device
                )

            self.input_spec["full_action_spec"] = value.to(device)
        finally:
            self.input_spec.lock_()

    @property
    def full_action_spec(self) -> CompositeSpec:
        """The full action spec.

        ``full_action_spec`` is a :class:`~torchrl.data.CompositeSpec`` instance
        that contains all the action entries.

        Examples:
            >>> from torchrl.envs import BraxEnv
            >>> for envname in BraxEnv.available_envs:
            ...     break
            >>> env = BraxEnv(envname)
            >>> env.full_action_spec
        CompositeSpec(
            action: BoundedTensorSpec(
                shape=torch.Size([8]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=cpu, shape=torch.Size([]))

        """
        return self.input_spec["full_action_spec"]

    @full_action_spec.setter
    def full_action_spec(self, spec: CompositeSpec) -> None:
        self.action_spec = spec

    # Reward spec
    @property
    def reward_keys(self) -> List[NestedKey]:
        """The reward keys of an environment.

        By default, there will only be one key named "reward".

        Keys are sorted by depth in the data tree.
        """
        reward_keys = self.__dict__.get("_reward_keys", None)
        if reward_keys is not None:
            return reward_keys

        reward_keys = sorted(self.full_reward_spec.keys(True, True), key=_repr_by_depth)
        self.__dict__["_reward_keys"] = reward_keys
        return reward_keys

    @property
    def reward_key(self):
        """The reward key of an environment.

        By default, this will be "reward".

        If there is more than one reward key in the environment, this function will raise an exception.
        """
        if len(self.reward_keys) > 1:
            raise KeyError(
                "reward_key requested but more than one key present in the environment"
            )
        return self.reward_keys[0]

    # Reward spec: reward specs belong to output_spec
    @property
    def reward_spec(self) -> TensorSpec:
        """The ``reward`` spec.

        The ``reward_spec`` is always stored as a composite spec.

        If the reward spec is provided as a simple spec, this will be returned.

            >>> env.reward_spec = UnboundedContinuousTensorSpec(1)
            >>> env.reward_spec
            UnboundedContinuousTensorSpec(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the reward spec is provided as a composite spec and contains only one leaf,
        this function will return just the leaf.

            >>> env.reward_spec = CompositeSpec({"nested": {"reward": UnboundedContinuousTensorSpec(1)}})
            >>> env.reward_spec
            UnboundedContinuousTensorSpec(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the reward spec is provided as a composite spec and has more than one leaf,
        this function will return the whole spec.

            >>> env.reward_spec = CompositeSpec({"nested": {"reward": UnboundedContinuousTensorSpec(1), "another_reward": DiscreteTensorSpec(1)}})
            >>> env.reward_spec
            CompositeSpec(
                nested: CompositeSpec(
                    reward: UnboundedContinuousTensorSpec(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    another_reward: DiscreteTensorSpec(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=1),
                        device=cpu,
                        dtype=torch.int64,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        To retrieve the full spec passed, use:

            >>> env.output_spec["full_reward_spec"]

        This property is mutable.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.reward_spec
            UnboundedContinuousTensorSpec(
                shape=torch.Size([1]),
                space=None,
                device=cpu,
                dtype=torch.float32,
                domain=continuous)
        """
        reward_spec = self.output_spec["full_reward_spec"]
        # try:
        #     reward_spec = self.output_spec["full_reward_spec"]
        # except (KeyError, AttributeError):
        #     # populate the "reward" entry
        #     # this will be raised if there is not full_reward_spec (unlikely) or no reward_key
        #     # Since output_spec is lazily populated with an empty composite spec for
        #     # reward_spec, the second case is much more likely to occur.
        #     self.reward_spec = UnboundedContinuousTensorSpec(
        #         shape=(*self.batch_size, 1),
        #         device=self.device,
        #     )
        #     reward_spec = self.output_spec["full_reward_spec"]

        reward_keys = self.reward_keys
        if len(reward_keys) > 1 or not len(reward_keys):
            return reward_spec
        else:
            return reward_spec[self.reward_keys[0]]

    @reward_spec.setter
    def reward_spec(self, value: TensorSpec) -> None:
        try:
            self.output_spec.unlock_()
            device = self.output_spec.device
            try:
                delattr(self, "_reward_keys")
            except AttributeError:
                pass
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"reward_spec of type {type(value)} do not have a shape "
                    f"attribute."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            if isinstance(value, CompositeSpec):
                for _ in value.values(True, True):  # noqa: B007
                    break
                else:
                    raise RuntimeError(
                        "An empty CompositeSpec was passed for the reward spec. "
                        "This is currently not permitted."
                    )
            else:
                value = CompositeSpec(
                    reward=value.to(device), shape=self.batch_size, device=device
                )
            for leaf in value.values(True, True):
                if len(leaf.shape) == 0:
                    raise RuntimeError(
                        "the reward_spec's leafs shape cannot be empty (this error"
                        " usually comes from trying to set a reward_spec"
                        " with a null number of dimensions. Try using a multidimensional"
                        " spec instead, for instance with a singleton dimension at the tail)."
                    )
            self.output_spec["full_reward_spec"] = value.to(device)
        finally:
            self.output_spec.lock_()

    @property
    def full_reward_spec(self) -> CompositeSpec:
        """The full reward spec.

        ``full_reward_spec`` is a :class:`~torchrl.data.CompositeSpec`` instance
        that contains all the reward entries.

        Examples:
            >>> import gymnasium
            >>> from torchrl.envs import GymWrapper, TransformedEnv, RenameTransform
            >>> base_env = GymWrapper(gymnasium.make("Pendulum-v1"))
            >>> env = TransformedEnv(base_env, RenameTransform("reward", ("nested", "reward")))
            >>> env.full_reward_spec
            CompositeSpec(
                nested: CompositeSpec(
                    reward: UnboundedContinuousTensorSpec(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=None, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        """
        return self.output_spec["full_reward_spec"]

    @full_reward_spec.setter
    def full_reward_spec(self, spec: CompositeSpec) -> None:
        self.reward_spec = spec

    # done spec
    @property
    def done_keys(self) -> List[NestedKey]:
        """The done keys of an environment.

        By default, there will only be one key named "done".

        Keys are sorted by depth in the data tree.
        """
        done_keys = self.__dict__.get("_done_keys", None)
        if done_keys is not None:
            return done_keys
        done_keys = sorted(self.full_done_spec.keys(True, True), key=_repr_by_depth)
        self.__dict__["_done_keys"] = done_keys
        return done_keys

    @property
    def done_key(self):
        """The done key of an environment.

        By default, this will be "done".

        If there is more than one done key in the environment, this function will raise an exception.
        """
        if len(self.done_keys) > 1:
            raise KeyError(
                "done_key requested but more than one key present in the environment"
            )
        return self.done_keys[0]

    @property
    def full_done_spec(self) -> CompositeSpec:
        """The full done spec.

        ``full_done_spec`` is a :class:`~torchrl.data.CompositeSpec`` instance
        that contains all the done entries.
        It can be used to generate fake data with a structure that mimics the
        one obtained at runtime.

        Examples:
            >>> import gymnasium
            >>> from torchrl.envs import GymWrapper
            >>> env = GymWrapper(gymnasium.make("Pendulum-v1"))
            >>> env.full_done_spec
            CompositeSpec(
                done: DiscreteTensorSpec(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete),
                truncated: DiscreteTensorSpec(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete), device=cpu, shape=torch.Size([]))

        """
        return self.output_spec["full_done_spec"]

    @full_done_spec.setter
    def full_done_spec(self, spec: CompositeSpec) -> None:
        self.done_spec = spec

    # Done spec: done specs belong to output_spec
    @property
    def done_spec(self) -> TensorSpec:
        """The ``done`` spec.

        The ``done_spec`` is always stored as a composite spec.

        If the done spec is provided as a simple spec, this will be returned.

            >>> env.done_spec = DiscreteTensorSpec(2, dtype=torch.bool)
            >>> env.done_spec
            DiscreteTensorSpec(
                shape=torch.Size([]),
                space=DiscreteBox(n=2),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)

        If the done spec is provided as a composite spec and contains only one leaf,
        this function will return just the leaf.

            >>> env.done_spec = CompositeSpec({"nested": {"done": DiscreteTensorSpec(2, dtype=torch.bool)}})
            >>> env.done_spec
            DiscreteTensorSpec(
                shape=torch.Size([]),
                space=DiscreteBox(n=2),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)

        If the done spec is provided as a composite spec and has more than one leaf,
        this function will return the whole spec.

            >>> env.done_spec = CompositeSpec({"nested": {"done": DiscreteTensorSpec(2, dtype=torch.bool), "another_done": DiscreteTensorSpec(2, dtype=torch.bool)}})
            >>> env.done_spec
            CompositeSpec(
                nested: CompositeSpec(
                    done: DiscreteTensorSpec(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    another_done: DiscreteTensorSpec(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        To always retrieve the full spec passed, use:

            >>> env.output_spec["full_done_spec"]

        This property is mutable.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.done_spec
            DiscreteTensorSpec(
                shape=torch.Size([1]),
                space=DiscreteBox(n=2),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)
        """
        done_spec = self.output_spec["full_done_spec"]
        return done_spec

    def _create_done_specs(self):
        """Reads through the done specs and makes it so that it's complete.

        If the done_specs contain only a ``"done"`` entry, a similar ``"terminated"`` entry is created.
        Same goes if only ``"terminated"`` key is present.

        If none of ``"done"`` and ``"terminated"`` can be found and the spec is not
        empty, nothing is changed.

        """
        try:
            full_done_spec = self.output_spec["full_done_spec"]
        except KeyError:
            full_done_spec = CompositeSpec(
                shape=self.output_spec.shape, device=self.output_spec.device
            )
            full_done_spec["done"] = DiscreteTensorSpec(
                n=2,
                shape=(*full_done_spec.shape, 1),
                dtype=torch.bool,
                device=self.device,
            )
            full_done_spec["terminated"] = DiscreteTensorSpec(
                n=2,
                shape=(*full_done_spec.shape, 1),
                dtype=torch.bool,
                device=self.device,
            )
            self.output_spec.unlock_()
            self.output_spec["full_done_spec"] = full_done_spec
            self.output_spec.lock_()
            return

        def check_local_done(spec):
            shape = None
            for key, item in list(
                spec.items()
            ):  # list to avoid error due to in-loop changes
                # in the case where the spec is non-empty and there is no done and no terminated, we do nothing
                if key == "done" and "terminated" not in spec.keys():
                    spec["terminated"] = item.clone()
                elif key == "terminated" and "done" not in spec.keys():
                    spec["done"] = item.clone()
                elif isinstance(item, CompositeSpec):
                    check_local_done(item)
                else:
                    if shape is None:
                        shape = item.shape
                        continue
                    # checks that all shape match
                    if shape != item.shape:
                        raise ValueError(
                            f"All shapes should match in done_spec {spec} (shape={shape}, key={key})."
                        )

            # if the spec is empty, we need to add a done and terminated manually
            if spec.is_empty():
                spec["done"] = DiscreteTensorSpec(
                    n=2, shape=(*spec.shape, 1), dtype=torch.bool, device=self.device
                )
                spec["terminated"] = DiscreteTensorSpec(
                    n=2, shape=(*spec.shape, 1), dtype=torch.bool, device=self.device
                )

        self.output_spec.unlock_()
        check_local_done(full_done_spec)
        self.output_spec["full_done_spec"] = full_done_spec
        self.output_spec.lock_()
        return

    @done_spec.setter
    def done_spec(self, value: TensorSpec) -> None:
        try:
            self.output_spec.unlock_()
            device = self.output_spec.device
            try:
                delattr(self, "_done_keys")
            except AttributeError:
                pass
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"done_spec of type {type(value)} do not have a shape "
                    f"attribute."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            if isinstance(value, CompositeSpec):
                for _ in value.values(True, True):  # noqa: B007
                    break
                else:
                    raise RuntimeError(
                        "An empty CompositeSpec was passed for the done spec. "
                        "This is currently not permitted."
                    )
            else:
                value = CompositeSpec(
                    done=value.to(device),
                    terminated=value.to(device),
                    shape=self.batch_size,
                    device=device,
                )
            for leaf in value.values(True, True):
                if len(leaf.shape) == 0:
                    raise RuntimeError(
                        "the done_spec's leafs shape cannot be empty (this error"
                        " usually comes from trying to set a reward_spec"
                        " with a null number of dimensions. Try using a multidimensional"
                        " spec instead, for instance with a singleton dimension at the tail)."
                    )
            self.output_spec["full_done_spec"] = value.to(device)
            self._create_done_specs()
        finally:
            self.output_spec.lock_()

    # observation spec: observation specs belong to output_spec
    @property
    def observation_spec(self) -> CompositeSpec:
        """Observation spec.

        Must be a :class:`torchrl.data.CompositeSpec` instance.
        The keys listed in the spec are directly accessible after reset and step.

        In TorchRL, even though they are not properly speaking "observations"
        all info, states, results of transforms etc. outputs from the environment are stored in the
        ``observation_spec``.

        Therefore, ``"observation_spec"`` should be thought as
        a generic data container for environment outputs that are not done or reward data.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.observation_spec
            CompositeSpec(
                observation: BoundedTensorSpec(
                    shape=torch.Size([3]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([]))

        """
        observation_spec = self.output_spec["full_observation_spec"]
        if observation_spec is None:
            observation_spec = CompositeSpec(shape=self.batch_size, device=self.device)
            self.output_spec.unlock_()
            self.output_spec["full_observation_spec"] = observation_spec
            self.output_spec.lock_()
        return observation_spec

    @observation_spec.setter
    def observation_spec(self, value: TensorSpec) -> None:
        try:
            self.output_spec.unlock_()
            device = self.output_spec.device
            if not isinstance(value, CompositeSpec):
                raise TypeError("The type of an observation_spec must be Composite.")
            elif value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            self.output_spec["full_observation_spec"] = value.to(device)
        finally:
            self.output_spec.lock_()

    @property
    def full_observation_spec(self) -> CompositeSpec:
        return self.observation_spec

    @full_observation_spec.setter
    def full_observation_spec(self, spec: CompositeSpec):
        self.observation_spec = spec

    # state spec: state specs belong to input_spec
    @property
    def state_spec(self) -> CompositeSpec:
        """State spec.

        Must be a :class:`torchrl.data.CompositeSpec` instance.
        The keys listed here should be provided as input alongside actions to the environment.

        In TorchRL, even though they are not properly speaking "state"
        all inputs to the environment that are not actions are stored in the
        ``state_spec``.

        Therefore, ``"state_spec"`` should be thought as
        a generic data container for environment inputs that are not action data.

        Examples:
            >>> from torchrl.envs import BraxEnv
            >>> for envname in BraxEnv.available_envs:
            ...     break
            >>> env = BraxEnv(envname)
            >>> env.state_spec
            CompositeSpec(
                state: CompositeSpec(
                    pipeline_state: CompositeSpec(
                        q: UnboundedContinuousTensorSpec(
                            shape=torch.Size([15]),
                            space=None,
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                [...], device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))


        """
        state_spec = self.input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = CompositeSpec(shape=self.batch_size, device=self.device)
            self.input_spec.unlock_()
            self.input_spec["full_state_spec"] = state_spec
            self.input_spec.lock_()
        return state_spec

    @state_spec.setter
    def state_spec(self, value: CompositeSpec) -> None:
        try:
            self.input_spec.unlock_()
            if value is None:
                self.input_spec["full_state_spec"] = CompositeSpec(
                    device=self.device, shape=self.batch_size
                )
            else:
                device = self.input_spec.device
                if not isinstance(value, CompositeSpec):
                    raise TypeError("The type of an state_spec must be Composite.")
                elif value.shape[: len(self.batch_size)] != self.batch_size:
                    raise ValueError(
                        f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                    )
                if value.shape[: len(self.batch_size)] != self.batch_size:
                    raise ValueError(
                        f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                    )
                self.input_spec["full_state_spec"] = value.to(device)
        finally:
            self.input_spec.lock_()

    @property
    def full_state_spec(self) -> CompositeSpec:
        """The full state spec.

        ``full_state_spec`` is a :class:`~torchrl.data.CompositeSpec`` instance
        that contains all the state entries (ie, the input data that is not action).

        Examples:
            >>> from torchrl.envs import BraxEnv
            >>> for envname in BraxEnv.available_envs:
            ...     break
            >>> env = BraxEnv(envname)
            >>> env.full_state_spec
            CompositeSpec(
                state: CompositeSpec(
                    pipeline_state: CompositeSpec(
                        q: UnboundedContinuousTensorSpec(
                            shape=torch.Size([15]),
                            space=None,
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                [...], device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        """
        return self.state_spec

    @full_state_spec.setter
    def full_state_spec(self, spec: CompositeSpec) -> None:
        self.state_spec = spec

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Makes a step in the environment.

        Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
        to be taken.
        Step will call an out-place private method, _step, which is the method to be re-written by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase): Tensordict containing the action to be taken.
                If the input tensordict contains a ``"next"`` entry, the values contained in it
                will prevail over the newly computed values. This gives a mechanism
                to override the underlying computations.

        Returns:
            the input tensordict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """
        # sanity check
        self._assert_tensordict_shape(tensordict)
        next_preset = tensordict.get("next", None)

        next_tensordict = self._step(tensordict)
        next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            # tensordict could already have a "next" key
            # this could be done more efficiently by not excluding but just passing
            # the necessary keys
            next_tensordict.update(
                next_preset.exclude(*next_tensordict.keys(True, True))
            )
        tensordict.set("next", next_tensordict)
        return tensordict

    @classmethod
    def _complete_done(
        cls, done_spec: CompositeSpec, data: TensorDictBase
    ) -> TensorDictBase:
        """Completes the data structure at step time to put missing done keys."""
        # by default, if a done key is missing, it is assumed that it is False
        # except in 2 cases: (1) there is a "done" but no "terminated" or (2)
        # there is a "terminated" but no "done".
        if done_spec.ndim:
            leading_dim = data.shape[: -done_spec.ndim]
        else:
            leading_dim = data.shape
        vals = {}
        i = -1
        for i, (key, item) in enumerate(done_spec.items()):  # noqa: B007
            val = data.get(key, None)
            if isinstance(item, CompositeSpec):
                cls._complete_done(item, val)
                continue
            shape = (*leading_dim, *item.shape)
            if val is not None:
                if val.shape != shape:
                    data.set(key, val.reshape(shape))
                vals[key] = val

        if len(vals) < i + 1:
            # complete missing dones: we only want to do that if we don't have enough done values
            data_keys = set(data.keys())
            done_spec_keys = set(done_spec.keys())
            for key, item in done_spec.items(False, True):
                val = vals.get(key, None)
                if (
                    key == "done"
                    and val is not None
                    and "terminated" in done_spec_keys
                    and "terminated" not in data_keys
                ):
                    if "truncated" in data_keys:
                        raise RuntimeError(
                            "Cannot infer the value of terminated when only done and truncated are present."
                        )
                    data.set("terminated", val)
                elif (
                    key == "terminated"
                    and val is not None
                    and "done" in done_spec_keys
                    and "done" not in data_keys
                ):
                    if "truncated" in data_keys:
                        done = val | data.get("truncated")
                        data.set("done", done)
                    else:
                        data.set("done", val)
                elif val is None:
                    # we must keep this here: we only want to fill with 0s if we're sure
                    # done should not be copied to terminated or terminated to done
                    # in this case, just fill with 0s
                    data.set(key, item.zero(leading_dim))
        return data

    def _step_proc_data(self, next_tensordict_out):
        batch_size = self.batch_size
        dims = len(batch_size)
        leading_batch_size = (
            next_tensordict_out.batch_size[:-dims]
            if dims
            else next_tensordict_out.shape
        )
        for reward_key in self.reward_keys:
            reward = next_tensordict_out.get(reward_key)
            expected_reward_shape = torch.Size(
                [
                    *leading_batch_size,
                    *self.output_spec["full_reward_spec"][reward_key].shape,
                ]
            )
            actual_reward_shape = reward.shape
            if actual_reward_shape != expected_reward_shape:
                reward = reward.view(expected_reward_shape)
                next_tensordict_out.set(reward_key, reward)

        self._complete_done(self.full_done_spec, next_tensordict_out)

        if self.run_type_checks:
            for key, spec in self.observation_spec.items():
                obs = next_tensordict_out.get(key)
                spec.type_check(obs)

            for reward_key in self.reward_keys:
                if (
                    next_tensordict_out.get(reward_key).dtype
                    is not self.output_spec[
                        unravel_key(("full_reward_spec", reward_key))
                    ].dtype
                ):
                    raise TypeError(
                        f"expected reward.dtype to be {self.output_spec[unravel_key(('full_reward_spec',reward_key))]} "
                        f"but got {next_tensordict_out.get(reward_key).dtype}"
                    )

            for done_key in self.done_keys:
                if (
                    next_tensordict_out.get(done_key).dtype
                    is not self.output_spec["full_done_spec", done_key].dtype
                ):
                    raise TypeError(
                        f"expected done.dtype to be {self.output_spec['full_done_spec', done_key].dtype} but got {next_tensordict_out.get(done_key).dtype}"
                    )
        return next_tensordict_out

    def _get_in_keys_to_exclude(self, tensordict):
        if self._cache_in_keys is None:
            self._cache_in_keys = list(
                set(self.input_spec.keys(True)).intersection(
                    tensordict.keys(True, True)
                )
            )
        return self._cache_in_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError("EnvBase.forward is not implemented")

    @abc.abstractmethod
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        raise NotImplementedError

    def reset(
        self,
        tensordict: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        """Resets the environment.

        As for step and _step, only the private method :obj:`_reset` should be overwritten by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase, optional): tensordict to be used to contain the resulting new observation.
                In some cases, this input can also be used to pass argument to the reset function.
            kwargs (optional): other arguments to be passed to the native
                reset function.

        Returns:
            a tensordict (or the input tensordict, if any), modified in place with the resulting observations.

        """
        if tensordict is not None:
            self._assert_tensordict_shape(tensordict)

        tensordict_reset = self._reset(tensordict, **kwargs)
        #        We assume that this is done properly
        #        if tensordict_reset.device != self.device:
        #            tensordict_reset = tensordict_reset.to(self.device, non_blocking=True)
        if tensordict_reset is tensordict:
            raise RuntimeError(
                "EnvBase._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _reset before writing new tensors onto this new instance."
            )
        if not isinstance(tensordict_reset, TensorDictBase):
            raise RuntimeError(
                f"env._reset returned an object of type {type(tensordict_reset)} but a TensorDict was expected."
            )

        return self._reset_proc_data(tensordict, tensordict_reset)

    def _reset_proc_data(self, tensordict, tensordict_reset):
        self._complete_done(self.full_done_spec, tensordict_reset)
        self._reset_check_done(tensordict, tensordict_reset)
        if tensordict is not None:
            return _update_during_reset(tensordict_reset, tensordict, self.reset_keys)
        return tensordict_reset

    def _reset_check_done(self, tensordict, tensordict_reset):
        """Checks the done status after reset.

        If _reset signals were passed, we check that the env is not done for these
        indices.

        We also check that the input tensordict contained ``"done"``s if the
        reset is partial and incomplete.

        """
        # we iterate over (reset_key, (done_key, truncated_key)) and check that all
        # values where reset was true now have a done set to False.
        # If no reset was present, all done and truncated must be False
        for reset_key, done_key_group in zip(self.reset_keys, self.done_keys_groups):
            reset_value = (
                tensordict.get(reset_key, default=None)
                if tensordict is not None
                else None
            )
            if reset_value is not None:
                for done_key in done_key_group:
                    done_val = tensordict_reset.get(done_key)
                    if done_val[reset_value].any() and not self._allow_done_after_reset:
                        raise RuntimeError(
                            f"Env done entry '{done_key}' was (partially) True after reset on specified '_reset' dimensions. This is not allowed."
                        )
                    if (
                        done_key not in tensordict.keys(True)
                        and done_val[~reset_value].any()
                    ):
                        warnings.warn(
                            f"A partial `'_reset'` key has been passed to `reset` ({reset_key}), "
                            f"but the corresponding done_key ({done_key}) was not present in the input "
                            f"tensordict. "
                            f"This is discouraged, since the input tensordict should contain "
                            f"all the data not being reset."
                        )
                        # we set the done val to tensordict, to make sure that
                        # _update_during_reset does not pad the value
                        tensordict.set(done_key, done_val)
            elif not self._allow_done_after_reset:
                for done_key in done_key_group:
                    if tensordict_reset.get(done_key).any():
                        raise RuntimeError(
                            f"The done entry '{done_key}' was (partially) True after a call to reset() in env {self}."
                        )

    def numel(self) -> int:
        return prod(self.batch_size)

    def set_seed(
        self, seed: Optional[int] = None, static_seed: bool = False
    ) -> Optional[int]:
        """Sets the seed of the environment and returns the next seed to be used (which is the input seed if a single environment is present).

        Args:
            seed (int): seed to be set
            static_seed (bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            integer representing the "next seed": i.e. the seed that should be
            used for another environment if created concomittently to this environment.

        """
        if seed is not None:
            torch.manual_seed(seed)
        self._set_seed(seed)
        if seed is not None and not static_seed:
            new_seed = seed_generator(seed)
            seed = new_seed
        return seed

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def _assert_tensordict_shape(self, tensordict: TensorDictBase) -> None:
        if (
            self.batch_locked or self.batch_size != ()
        ) and tensordict.batch_size != self.batch_size:
            raise RuntimeError(
                f"Expected a tensordict with shape==env.shape, "
                f"got {tensordict.batch_size} and {self.batch_size}"
            )

    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        """Performs a random action given the action_spec attribute.

        Args:
            tensordict (TensorDictBase, optional): tensordict where the resulting action should be written.

        Returns:
            a tensordict object with the "action" entry updated with a random
            sample from the action-spec.

        """
        shape = torch.Size([])
        if not self.batch_locked:
            if not self.batch_size and tensordict is not None:
                # if we can't infer the batch-size from the env, take it from tensordict
                shape = tensordict.shape
            elif not self.batch_size:
                # if tensordict wasn't provided, we assume empty batch size
                shape = torch.Size([])
            elif tensordict.shape != self.batch_size:
                # if tensordict is not None and the env has a batch size, their shape must match
                raise RuntimeError(
                    "The input tensordict and the env have a different batch size: "
                    f"env.batch_size={self.batch_size} and tensordict.batch_size={tensordict.shape}. "
                    f"Non batch-locked environment require the env batch-size to be either empty or to"
                    f" match the tensordict one."
                )
        # We generate the action from the full_action_spec
        r = self.input_spec["full_action_spec"].rand(shape)
        if tensordict is None:
            return r
        tensordict.update(r)
        return tensordict

    def rand_step(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """Performs a random step in the environment given the action_spec attribute.

        Args:
            tensordict (TensorDictBase, optional): tensordict where the resulting info should be written.

        Returns:
            a tensordict object with the new observation after a random step in the environment. The action will
            be stored with the "action" key.

        """
        tensordict = self.rand_action(tensordict)
        return self.step(tensordict)

    @property
    def specs(self) -> CompositeSpec:
        """Returns a Composite container where all the environment are present.

        This feature allows one to create an environment, retrieve all of the specs in a single data container and then
        erase the environment from the workspace.

        """
        return CompositeSpec(
            output_spec=self.output_spec,
            input_spec=self.input_spec,
            shape=self.batch_size,
        ).lock_()

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase, ...], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        out=None,
    ):
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if ``True``, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from torchrl.envs.transforms import TransformedEnv, StepCounter
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(max_steps=20))
            >>> rollout = env.rollout(max_steps=1000)
            >>> print(rollout)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                    done: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            step_count: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                            truncated: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([20]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                    step_count: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    truncated: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([20]),
                device=cpu,
                is_shared=False)
            >>> print(rollout.names)
            ['time']
            >>> # with envs that contain more dimensions
            >>> from torchrl.envs import SerialEnv
            >>> env = SerialEnv(3, lambda: TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(max_steps=20)))
            >>> rollout = env.rollout(max_steps=1000)
            >>> print(rollout)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                    done: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([3, 20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            step_count: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                            truncated: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([3, 20]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([3, 20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                    step_count: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    truncated: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([3, 20]),
                device=cpu,
                is_shared=False)
            >>> print(rollout.names)
            [None, 'time']

        In some instances, contiguous tensordict cannot be obtained because
        they cannot be stacked. This can happen when the data returned at
        each step may have a different shape, or when different environments
        are executed together. In that case, ``return_contiguous=False``
        will cause the returned tensordict to be a lazy stack of tensordicts:

        Examples:
            >>> rollout = env.rollout(4, return_contiguous=False)
            >>> print(rollout)
        LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3, 4]),
            device=cpu,
            is_shared=False)
            >>> print(rollout.names)
            [None, 'time']

        """
        try:
            policy_device = next(policy.parameters()).device
        except (StopIteration, AttributeError):
            policy_device = self.device

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        if policy is None:

            policy = self.rand_action

        kwargs = {
            "tensordict": tensordict,
            "auto_cast_to_device": auto_cast_to_device,
            "max_steps": max_steps,
            "policy": policy,
            "policy_device": policy_device,
            "env_device": env_device,
            "callback": callback,
        }
        if break_when_any_done:
            tensordicts = self._rollout_stop_early(**kwargs)
        else:
            tensordicts = self._rollout_nonstop(**kwargs)
        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        out_td = torch.stack(tensordicts, len(batch_size), out=out)
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")
        return out_td

    def _rollout_stop_early(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
    ):
        tensordicts = []
        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)
            tensordict = policy(tensordict)
            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)
            tensordicts.append(tensordict.clone(False))

            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.reward_keys,
                action_keys=self.action_keys,
                done_keys=self.done_keys,
            )
            # done and truncated are in done_keys
            # We read if any key is done.
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key=None,
            )
            if any_done:
                break

            if callback is not None:
                callback(self, tensordict)
        return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
    ):
        tensordicts = []
        tensordict_ = tensordict
        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(policy_device, non_blocking=True)
            tensordict_ = policy(tensordict_)
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(env_device, non_blocking=True)
            tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            if callback is not None:
                callback(self, tensordict)

        return tensordicts

    def step_and_maybe_reset(
        self, tensordict: TensorDictBase
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        """Runs a step in the environment and (partially) resets it if needed.

        Args:
            tensordict (TensorDictBase): an input data structure for the :meth:`~.step`
                method.

        This method allows to easily code non-stopping rollout functions.

        Examples:
            >>> from torchrl.envs import ParallelEnv, GymEnv
            >>> def rollout(env, n):
            ...     data_ = env.reset()
            ...     result = []
            ...     for i in range(n):
            ...         data, data_ = env.step_and_maybe_reset(data_)
            ...         result.append(data)
            ...     return torch.stack(result).contiguous()
            >>> env = ParallelEnv(2, lambda: GymEnv("CartPole-v1"))
            >>> print(rollout(env, 2))
            TensorDict(
                fields={
                    done: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([2, 2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            terminated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            truncated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([2, 2]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([2, 2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    terminated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([2, 2]),
                device=cpu,
                is_shared=False)
        """
        tensordict = self.step(tensordict)
        # done and truncated are in done_keys
        # We read if any key is done.
        tensordict_ = step_mdp(
            tensordict,
            keep_other=True,
            exclude_action=False,
            exclude_reward=True,
            reward_keys=self.reward_keys,
            action_keys=self.action_keys,
            done_keys=self.done_keys,
        )
        any_done = _terminated_or_truncated(
            tensordict_,
            full_done_spec=self.output_spec["full_done_spec"],
            key="_reset",
        )
        if any_done:
            tensordict_ = self.reset(tensordict_)
        return tensordict, tensordict_

    def empty_cache(self):
        """Erases all the cached values.

        For regular envs, the key lists (reward, done etc) are cached, but in some cases
        they may change during the execution of the code (eg, when adding a transform).

        """
        self.__dict__["_reward_keys"] = None
        self.__dict__["_done_keys"] = None
        self.__dict__["_action_keys"] = None
        self.__dict__["_done_keys_group"] = None

    @property
    def reset_keys(self) -> List[NestedKey]:
        """Returns a list of reset keys.

        Reset keys are keys that indicate partial reset, in batched, multitask or multiagent
        settings. They are structured as ``(*prefix, "_reset")`` where ``prefix`` is
        a (possibly empty) tuple of strings pointing to a tensordict location
        where a done state can be found.

        Keys are sorted by depth in the data tree.
        """
        reset_keys = self.__dict__.get("_reset_keys", None)
        if reset_keys is not None:
            return reset_keys

        reset_keys = sorted(
            (
                _replace_last(done_key, "_reset")
                for (done_key, *_) in self.done_keys_groups
            ),
            key=_repr_by_depth,
        )
        self.__dict__["_reset_keys"] = reset_keys
        return reset_keys

    @property
    def _filtered_reset_keys(self):
        """Returns the only the effective reset keys, discarding nested resets if they're not being used."""
        reset_keys = self.reset_keys
        result = []

        def _root(key):
            if isinstance(key, str):
                return ()
            return key[:-1]

        roots = []
        for reset_key in reset_keys:
            cur_root = _root(reset_key)
            for root in roots:
                if cur_root[: len(root)] == root:
                    break
            else:
                roots.append(cur_root)
                result.append(reset_key)
        return result

    @property
    def done_keys_groups(self):
        """A list of done keys, grouped as the reset keys.

        This is a list of lists. The outer list has the length of reset keys, the
        inner lists contain the done keys (eg, done and truncated) that can
        be read to determine a reset when it is absent.
        """
        done_keys_group = self.__dict__.get("_done_keys_group", None)
        if done_keys_group is not None:
            return done_keys_group

        # done keys, sorted as reset keys
        done_keys_group = []
        roots = set()
        fds = self.full_done_spec
        for done_key in self.done_keys:
            root_name = done_key[:-1] if isinstance(done_key, tuple) else ()
            root = fds[root_name] if root_name else fds
            n = len(roots)
            roots.add(root_name)
            if len(roots) - n:
                done_keys_group.append(
                    [
                        unravel_key(root_name + (key,))
                        for key in root.keys(include_nested=False, leaves_only=True)
                    ]
                )
        self.__dict__["_done_keys_group"] = done_keys_group
        return done_keys_group

    def _select_observation_keys(self, tensordict: TensorDictBase) -> Iterator[str]:
        for key in tensordict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def close(self):
        self.is_closed = True

    def __del__(self):
        # if del occurs before env has been set up, we don't want a recursion
        # error
        if "is_closed" in self.__dict__ and not self.is_closed:
            try:
                self.close()
            except Exception:
                # a TypeError will typically be raised if the env is deleted when the program ends.
                # In the future, insignificant changes to the close method may change the error type.
                # We excplicitely assume that any error raised during closure in
                # __del__ will not affect the program.
                pass

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        device = torch.device(device)
        if device == self.device:
            return self
        self.__dict__["_input_spec"] = self.input_spec.to(device).lock_()
        self.__dict__["_output_spec"] = self.output_spec.to(device).lock_()
        self._device = device
        return super().to(device)

    def fake_tensordict(self) -> TensorDictBase:
        """Returns a fake tensordict with key-value pairs that match in shape, device and dtype what can be expected during an environment rollout."""
        state_spec = self.state_spec
        observation_spec = self.observation_spec
        action_spec = self.input_spec["full_action_spec"]
        # instantiates reward_spec if needed
        _ = self.reward_spec
        reward_spec = self.output_spec["full_reward_spec"]
        full_done_spec = self.output_spec["full_done_spec"]

        fake_obs = observation_spec.zero()

        fake_state = state_spec.zero()
        fake_action = action_spec.zero()
        fake_input = fake_state.update(fake_action)

        # the input and output key may match, but the output prevails
        # Hence we generate the input, and override using the output
        fake_in_out = fake_input.update(fake_obs)

        fake_reward = reward_spec.zero()
        fake_done = full_done_spec.zero()

        next_output = fake_obs.clone()
        next_output.update(fake_reward)
        next_output.update(fake_done)
        fake_in_out.update(fake_done.clone())
        if "next" not in fake_in_out.keys():
            fake_in_out.set("next", next_output)
        else:
            fake_in_out.get("next").update(next_output)

        fake_in_out.batch_size = self.batch_size
        fake_in_out = fake_in_out.to(self.device)
        return fake_in_out



class Transform(nn.Module):
    """Environment transform parent class.

    In principle, a transform receives a tensordict as input and returns (
    the same or another) tensordict as output, where a series of values have
    been modified or created with a new key. When instantiating a new
    transform, the keys that are to be read from are passed to the
    constructor via the :obj:`keys` argument.

    Transforms are to be combined with their target environments with the
    TransformedEnv class, which takes as arguments an :obj:`EnvBase` instance
    and a transform. If multiple transforms are to be used, they can be
    concatenated using the :obj:`Compose` class.
    A transform can be stateless or stateful (e.g. CatTransform). Because of
    this, Transforms support the :obj:`reset` operation, which should reset the
    transform to its initial state (such that successive trajectories are kept
    independent).

    Notably, :obj:`Transform` subclasses take care of transforming the affected
    specs from an environment: when querying
    `transformed_env.observation_spec`, the resulting objects will describe
    the specs of the transformed_in tensors.

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.in_keys_inv = in_keys_inv
        self.out_keys_inv = out_keys_inv
        self._missing_tolerance = False
        # we use __dict__ to avoid having nn.Module placing these objects in the module list
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    @property
    def in_keys(self):
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is None:
            return []
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self):
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is None:
            return []
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def in_keys_inv(self):
        in_keys_inv = self.__dict__.get("_in_keys_inv", None)
        if in_keys_inv is None:
            return []
        return in_keys_inv

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys_inv = value

    @property
    def out_keys_inv(self):
        out_keys_inv = self.__dict__.get("_out_keys_inv", None)
        if out_keys_inv is None:
            return []
        return out_keys_inv

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys_inv = value

    def reset(self, tensordict):
        warnings.warn("Transform.reset public method will be derpecated in v0.4.0.")
        return self._reset(tensordict, tensordict_reset=tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets a transform if it is stateful."""
        return tensordict_reset

    def init(self, tensordict) -> None:
        pass

    def _apply_transform(self, obs: torch.Tensor) -> None:
        """Applies the transform to a tensor.

        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform).

        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._apply_transform is not coded. If the transform is coded in "
            "transform._call, make sure that this method is called instead of"
            "transform.forward, which is reserved for usage inside nn.Modules"
            "or appended to a replay buffer."
        )

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform.

        For any operation that relates exclusively to the parent env (e.g. FrameSkip),
        modify the _step method instead. :meth:`~._call` should only be overwritten
        if a modification of the input tensordict is needed.

        :meth:`~._call` will be called by :meth:`TransformedEnv.step` and
        :meth:`TransformedEnv.reset`.

        """
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            value = tensordict.get(in_key, default=None)
            if value is not None:
                observation = self._apply_transform(value)
                tensordict.set(
                    out_key,
                    observation,
                )
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {tensordict}"
                )
        return tensordict

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            data = tensordict.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                tensordict.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """The parent method of a transform during the ``env.step`` execution.

        This method should be overwritten whenever the :meth:`~._step` needs to be
        adapted. Unlike :meth:`~._call`, it is assumed that :meth:`~._step`
        will execute some operation with the parent env or that it requires
        access to the content of the tensordict at time ``t`` and not only
        ``t+1`` (the ``"next"`` entry in the input tensordict).

        :meth:`~._step` will only be called by :meth:`TransformedEnv.step` and
        not by :meth:`TransformedEnv.reset`.

        Args:
            tensordict (TensorDictBase): data at time t
            next_tensordict (TensorDictBase): data at time t+1

        Returns: the data at t+1
        """
        next_tensordict = self._call(next_tensordict)
        return next_tensordict

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if self.invertible:
            raise NotImplementedError
        else:
            return state

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # # We create a shallow copy of the tensordict to avoid that changes are
        # # exposed to the user: we'd like that the input keys remain unchanged
        # # in the originating script if they're being transformed.
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            data = tensordict.get(in_key, None)
            if data is not None:
                item = self._inv_apply_transform(data)
                tensordict.set(out_key, item)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")

        return tensordict

    @dispatch(source="in_keys_inv", dest="out_keys_inv")
    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = self._inv_call(tensordict.clone(False))
        return out

    def transform_env_device(self, device: torch.device):
        """Transforms the device of the parent env."""
        return device

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        """Transforms the output spec such that the resulting spec matches transform mapping.

        This method should generally be left untouched. Changes should be implemented using
        :meth:`~.transform_observation_spec`, :meth:`~.transform_reward_spec` and :meth:`~.transformfull_done_spec`.
        Args:
            output_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        output_spec = output_spec.clone()
        output_spec["full_observation_spec"] = self.transform_observation_spec(
            output_spec["full_observation_spec"]
        )
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self.transform_reward_spec(
                output_spec["full_reward_spec"]
            )
        if "full_done_spec" in output_spec.keys():
            output_spec["full_done_spec"] = self.transform_done_spec(
                output_spec["full_done_spec"]
            )
        return output_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        """Transforms the input spec such that the resulting spec matches transform mapping.

        Args:
            input_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec such that the resulting spec matches transform mapping.

        Args:
            observation_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        """Transforms the reward spec such that the resulting spec matches transform mapping.

        Args:
            reward_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return reward_spec

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        """Transforms the done spec such that the resulting spec matches transform mapping.

        Args:
            done_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return done_spec

    def dump(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.in_keys})"

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        if self.parent is not None:
            raise AttributeError(
                f"parent of transform {type(self)} already set. "
                "Call `transform.clone()` to get a similar transform with no parent set."
            )
        self.__dict__["_container"] = container
        self.__dict__["_parent"] = None

    def reset_parent(self) -> None:
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def clone(self):
        self_copy = copy(self)
        state = copy(self.__dict__)
        state["_container"] = None
        state["_parent"] = None
        self_copy.__dict__.update(state)
        return self_copy

    @property
    def container(self):
        """Returns the env containing the transform.

        Examples:
            >>> from torchrl.envs import TransformedEnv, Compose, RewardSum, StepCounter
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(RewardSum(), StepCounter()))
            >>> env.transform[0].container is env
            True
        """
        if "_container" not in self.__dict__:
            raise AttributeError("transform parent uninitialized")
        container = self.__dict__["_container"]
        if container is None:
            return container
        while not isinstance(container, EnvBase):
            # if it's not an env, it should be a Compose transform
            if not isinstance(container, Compose):
                raise ValueError(
                    "A transform parent must be either another Compose transform or an environment object."
                )
            compose = container
            container = compose.__dict__.get("_container", None)
        return container

    @property
    def parent(self) -> Optional[EnvBase]:
        """Returns the parent env of the transform.

        The parent env is the env that contains all the transforms up until the current one.

        Examples:
            >>> from torchrl.envs import TransformedEnv, Compose, RewardSum, StepCounter
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(RewardSum(), StepCounter()))
            >>> env.transform[1].parent
            TransformedEnv(
                env=GymEnv(env=Pendulum-v1, batch_size=torch.Size([]), device=cpu),
                transform=Compose(
                        RewardSum(keys=['reward'])))

        """
        if self.__dict__.get("_parent", None) is None:
            if "_container" not in self.__dict__:
                raise AttributeError("transform parent uninitialized")
            container = self.__dict__["_container"]
            if container is None:
                return container
            out = None
            if not isinstance(container, EnvBase):
                # if it's not an env, it should be a Compose transform
                if not isinstance(container, Compose):
                    raise ValueError(
                        "A transform parent must be either another Compose transform or an environment object."
                    )
                out, _ = container._rebuild_up_to(self)
            # elif isinstance(container, TransformedEnv):
            #     out = TransformedEnv(container.base_env)
            else:
                raise ValueError(f"container is of type {type(container)}")
            self.__dict__["_parent"] = out
        return self.__dict__["_parent"]

    def empty_cache(self):
        self.__dict__["_parent"] = None

    def set_missing_tolerance(self, mode=False):
        self._missing_tolerance = mode

    @property
    def missing_tolerance(self):
        return self._missing_tolerance

    def to(self, *args, **kwargs):
        # remove the parent, because it could have the wrong device associated
        self.empty_cache()
        return super().to(*args, **kwargs)


class Compose(Transform):
    """Composes a chain of transforms.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transforms = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
        >>> transforms = Compose(*transforms)
        >>> transformed_env = TransformedEnv(env, transforms)

    """

    def __init__(self, *transforms: Transform):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        for t in transforms:
            t.set_container(self)

    def to(self, *args, **kwargs):
        # because Module.to(...) does not call to(...) on sub-modules, we have
        # manually call it:
        self.transforms = nn.ModuleList(
            [t.to(*args, **kwargs) for t in self.transforms]
        )
        return super().to(*args, **kwargs)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t._call(tensordict)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t(tensordict)
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for t in self.transforms:
            next_tensordict = t._step(tensordict, next_tensordict)
        return next_tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in reversed(self.transforms):
            tensordict = t._inv_call(tensordict)
        return tensordict

    def transform_env_device(self, device: torch.device):
        for t in self.transforms:
            device = t.transform_env_device(device)
        return device

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms[::-1]:
            input_spec = t.transform_input_spec(input_spec)
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            observation_spec = t.transform_observation_spec(observation_spec)
        return observation_spec

    def transform_output_spec(self, output_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            output_spec = t.transform_output_spec(output_spec)
        return output_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            reward_spec = t.transform_reward_spec(reward_spec)
        return reward_spec

    def __getitem__(self, item: Union[int, slice, List]) -> Union:
        transform = self.transforms
        transform = transform[item]
        if not isinstance(transform, Transform):
            out = Compose(*(t.clone() for t in self.transforms[item]))
            out.set_container(self.parent)
            return out
        return transform

    def dump(self, **kwargs) -> None:
        for t in self:
            t.dump(**kwargs)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        for t in self.transforms:
            tensordict_reset = t._reset(tensordict, tensordict_reset)
        return tensordict_reset

    def init(self, tensordict: TensorDictBase) -> None:
        for t in self.transforms:
            t.init(tensordict)

    def append(self, transform):
        self.empty_cache()
        if not isinstance(transform, Transform):
            raise ValueError(
                "Compose.append expected a transform but received an object of "
                f"type {type(transform)} instead."
            )
        transform.eval()
        if type(self) == type(transform) == Compose:
            for t in transform:
                self.append(t)
        else:
            self.transforms.append(transform)
        transform.set_container(self)

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        self.reset_parent()
        super().set_container(container)
        for t in self.transforms:
            t.set_container(self)

    def insert(self, index: int, transform: Transform) -> None:
        if not isinstance(transform, Transform):
            raise ValueError(
                "Compose.append expected a transform but received an object of "
                f"type {type(transform)} instead."
            )

        if abs(index) > len(self.transforms):
            raise ValueError(
                f"Index expected to be between [-{len(self.transforms)}, {len(self.transforms)}] got index={index}"
            )

        # empty cache of all transforms to reset parents and specs
        self.empty_cache()
        if index < 0:
            index = index + len(self.transforms)
        transform.eval()
        self.transforms.insert(index, transform)
        transform.set_container(self)

    def __iter__(self):
        yield from self.transforms

    def __len__(self):
        return len(self.transforms)

    def __repr__(self) -> str:
        layers_str = ",\n".join(
            [indent(str(trsf), 4 * " ") for trsf in self.transforms]
        )
        return f"{self.__class__.__name__}(\n{indent(layers_str, 4 * ' ')})"

    def empty_cache(self):
        for t in self.transforms:
            t.empty_cache()
        super().empty_cache()

    def reset_parent(self):
        for t in self.transforms:
            t.reset_parent()
        super().reset_parent()

    def clone(self):
        transforms = []
        for t in self.transforms:
            transforms.append(t.clone())
        return Compose(*transforms)

    def set_missing_tolerance(self, mode=False):
        for t in self.transforms:
            t.set_missing_tolerance(mode)
        super().set_missing_tolerance(mode)

    def _rebuild_up_to(self, final_transform):
        container = self.__dict__["_container"]

        if isinstance(container, Compose):
            out, parent_compose = container._rebuild_up_to(self)
            if out is None:
                # returns None if there is no parent env
                return None, None
        # elif isinstance(container, TransformedEnv):
        #     out = TransformedEnv(container.base_env)
        elif container is None:
            # returns None if there is no parent env
            return None, None
        else:
            raise ValueError(f"Container of type {type(container)} isn't supported.")

        if final_transform not in self.transforms:
            raise ValueError(f"Cannot rebuild with transform {final_transform}.")
        list_of_transforms = []
        for orig_trans in self.transforms:
            if orig_trans is final_transform:
                break
            transform = orig_trans.clone()
            transform.reset_parent()
            list_of_transforms.append(transform)
        if isinstance(container, Compose):
            parent_compose.append(Compose(*list_of_transforms))
            return out, parent_compose[-1]
        elif isinstance(container, TransformedEnv):
            for t in list_of_transforms:
                out.append_transform(t)
            return out, out.transform
        else:
            raise NotImplementedError(f"{container}")

