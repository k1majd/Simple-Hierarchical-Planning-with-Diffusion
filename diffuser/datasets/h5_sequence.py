"""
Goal-conditioned dataset loaded from HDF5 files.
Standalone module — does NOT import d4rl, gym, or mujoco.
"""

from collections import namedtuple
import numpy as np
import torch

from .h5_dataset import sequence_dataset_h5
from .normalization import DatasetNormalizer

Batch = namedtuple("Batch", "trajectories conditions")


def _atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


class _H5ReplayBuffer:
    """Minimal replay buffer that avoids importing d4rl."""

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            "path_lengths": np.zeros(max_n_episodes, dtype=np.int32),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return "[ H5ReplayBuffer ] Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    def items(self):
        return {
            k: v for k, v in self._dict.items() if k != "path_lengths"
        }.items()

    def _add_attributes(self):
        for key, val in self._dict.items():
            setattr(self, key, val)

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)

    def add_path(self, path):
        if self._count >= self.max_n_episodes:
            return
        path_length = len(path["observations"])
        assert (
            path_length <= self.max_path_length
        ), f"Episode length {path_length} exceeds max_path_length {self.max_path_length}"

        if not hasattr(self, "keys"):
            self.keys = list(path.keys())

        for key in self.keys:
            array = _atleast_2d(path[key])
            if key not in self._dict:
                self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        if path["terminals"].any() and self.termination_penalty is not None:
            assert not path["timeouts"].any()
            self._dict["rewards"][
                self._count, path_length - 1
            ] += self.termination_penalty

        self._dict["path_lengths"][self._count] = path_length
        self._count += 1

    def finalize(self):
        for key in list(self._dict.keys()):
            self._dict[key] = self._dict[key][: self._count]
        self._add_attributes()


class H5GoalDataset(torch.utils.data.Dataset):
    """
    Goal-conditioned dataset loaded from an HDF5 file.
    Bypasses gym/d4rl entirely.
    """

    def __init__(
        self,
        h5_path,
        final_goal,
        horizon=64,
        normalizer="LimitsNormalizer",
        max_path_length=310,
        max_n_episodes=2000,
        termination_penalty=0,
        use_padding=True,
        jump=1,
        jump_action=False,
        # accept and ignore extra kwargs from config
        env=None,
        preprocess_fns=None,
    ):
        from .preprocessing import navigation_set_terminals

        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.jump = jump
        self.jump_action = jump_action

        preprocess_fn = navigation_set_terminals(final_goal)
        itr = sequence_dataset_h5(h5_path, preprocess_fn)

        fields = _H5ReplayBuffer(
            max_n_episodes, max_path_length, termination_penalty
        )
        for episode in itr:
            fields.add_path(episode)
        fields.finalize()
        self.fields = fields

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def normalize(self, keys=["observations", "actions"]):
        for key in keys:
            array = self.fields[key].reshape(
                self.n_episodes * self.max_path_length, -1
            )
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def make_indices(self, path_lengths, horizon):
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        return {
            0: observations[0],
            self.horizon // self.jump - 1: observations[-1],
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end][
            :: self.jump
        ]
        actions = self.fields.normed_actions[path_ind, start:end].reshape(
            -1, self.jump * self.action_dim
        )

        conditions = self.get_conditions(observations)
        if self.jump_action == "none":
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
