import collections
import numpy as np
import h5py


def load_h5_dataset(h5_path):
    """Load a flat dataset from an HDF5 file."""
    dataset = {}
    with h5py.File(h5_path, "r") as f:
        for key in [
            "observations",
            "actions",
            "rewards",
            "terminals",
            "timeouts",
        ]:
            dataset[key] = f[key][:]
        if "infos" in f and "goal" in f["infos"]:
            dataset["infos/goal"] = f["infos"]["goal"][:]
    return dataset


def sequence_dataset_h5(h5_path, preprocess_fn):
    """
    Yields episode dicts from an HDF5 dataset, splitting at terminal boundaries.
    Mirrors the d4rl.sequence_dataset interface but reads from HDF5 directly.
    """
    dataset = load_h5_dataset(h5_path)
    dataset = preprocess_fn(dataset)

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        final_timestep = bool(dataset["timeouts"][i])

        for k in dataset:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            episode_data = _process_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)


def _process_episode(episode):
    """
    Adds next_observations and trims last step,
    same as process_maze2d_episode in d4rl.py.
    """
    assert "next_observations" not in episode
    length = len(episode["observations"])
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode
