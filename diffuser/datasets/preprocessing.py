import numpy as np
import einops
from scipy.spatial.transform import Rotation as R
import pdb


def _load_environment(env):
    from .d4rl import load_environment

    return load_environment(env)


# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#


def compose(*fns):
    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn


def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)


def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)


# -----------------------------------------------------------------------------#
# -------------------------- preprocessing functions --------------------------#
# -----------------------------------------------------------------------------#

# ------------------------ @TODO: remove some of these ------------------------#


def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset["actions"]
        assert (
            actions.min() >= -1 and actions.max() <= 1
        ), f"applying arctanh to actions in range [{actions.min()}, {actions.max()}]"
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset["actions"] = np.arctanh(actions)
        return dataset

    return _fn


def arccos_actions(*args, **kwargs):
    def _fn(dataset):
        actions = dataset["actions"]
        dataset["actions"] = np.arctanh(actions)
        return dataset

    return _fn


def add_deltas(env):
    def _fn(dataset):
        deltas = dataset["next_observations"] - dataset["observations"]
        dataset["deltas"] = deltas
        return dataset

    return _fn


def maze2d_set_terminals(env):
    env = _load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset["observations"][:, :2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset["rewards"])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
            f"min length: {path_lengths.min()} | max length: {path_lengths.max()} | mean length: {path_lengths.mean()}"
        )

        dataset["timeouts"] = timeouts
        dataset["terminals"] = np.zeros_like(dataset["terminals"])
        return dataset

    return _fn


def navigation_set_terminals(final_goal):
    """
    Preprocessing function for the custom navigation dataset.
    Re-segments episodes so that terminal=True only when the current segment's
    goal matches `final_goal`. This concatenates the short 51-step segments
    into longer multi-goal episodes (~255-306 steps).
    """
    final_goal = np.array(final_goal)

    def _fn(dataset):
        goals = dataset["infos/goal"]
        old_terminals = dataset["terminals"].astype(bool)
        old_timeouts = dataset["timeouts"].astype(bool)

        # find where original episodes end
        term_indices = np.where(old_terminals | old_timeouts)[0]

        # only keep terminals where the goal matches the final goal
        new_terminals = np.zeros_like(dataset["terminals"], dtype=bool)
        new_timeouts = np.zeros_like(dataset["timeouts"], dtype=bool)

        for ti in term_indices:
            if np.allclose(goals[ti], final_goal, atol=0.01):
                new_terminals[ti] = True
                new_timeouts[ti] = True

        new_term_indices = np.where(new_terminals)[0]
        ep_lens = np.diff(np.concatenate([[-1], new_term_indices])).astype(int)

        print(
            f"[ preprocessing ] Re-segmented navigation dataset | "
            f"{len(new_term_indices)} episodes | "
            f"min length: {ep_lens.min()} | max length: {ep_lens.max()} | "
            f"mean length: {ep_lens.mean():.1f}"
        )

        dataset["terminals"] = new_terminals
        dataset["timeouts"] = new_timeouts
        return dataset

    return _fn


def navigation_angles_to_sincos(
    env=None, obs_angle_indices=(2, 3, 5), action_angle_indices=(1,)
):
    """
    Replace angle-like channels with sin/cos pairs.

    For navigation states (default):
      observations [x, y, theta1, theta2, v, delta]
      -> [x, y, sin(theta1), cos(theta1), sin(theta2), cos(theta2), v, sin(delta), cos(delta)]

    For actions (default):
      actions [a, omega]
      -> [a, sin(omega), cos(omega)]
    """

    def _transform(arr, angle_indices):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            return arr

        dim = arr.shape[1]
        valid_angle_indices = sorted([i for i in angle_indices if 0 <= i < dim])
        if not valid_angle_indices:
            return arr

        transformed = []
        angle_set = set(valid_angle_indices)
        for i in range(dim):
            col = arr[:, i : i + 1]
            if i in angle_set:
                transformed.append(np.sin(col))
                transformed.append(np.cos(col))
            else:
                transformed.append(col)
        return np.concatenate(transformed, axis=1)

    def _fn(dataset):
        obs_before = dataset["observations"].shape[1]
        act_before = dataset["actions"].shape[1]

        dataset["observations"] = _transform(
            dataset["observations"], obs_angle_indices
        )
        if "next_observations" in dataset:
            dataset["next_observations"] = _transform(
                dataset["next_observations"], obs_angle_indices
            )
        dataset["actions"] = _transform(
            dataset["actions"], action_angle_indices
        )

        obs_after = dataset["observations"].shape[1]
        act_after = dataset["actions"].shape[1]
        print(
            f"[ preprocessing ] navigation_angles_to_sincos | obs_dim: {obs_before} -> {obs_after} | "
            f"action_dim: {act_before} -> {act_after}"
        )
        return dataset

    return _fn


# -------------------------- block-stacking --------------------------#


def blocks_quat_to_euler(observations):
    """
    input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
        xyz: 3
        quat: 4
        contact: 1

    returns : [ N x robot_dim + n_blocks * 10] = [ N x 47 ]
        xyz: 3
        sin: 3
        cos: 3
        contact: 1
    """
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        quat = block_info[:, 3:-1]
        contact = block_info[:, -1:]

        euler = R.from_quat(quat).as_euler("xyz")
        sin = np.sin(euler)
        cos = np.cos(euler)

        X = np.concatenate(
            [
                X,
                xpos,
                sin,
                cos,
                contact,
            ],
            axis=-1,
        )

    return X


def blocks_euler_to_quat_2d(observations):
    robot_dim = 7
    block_dim = 10
    n_blocks = 4

    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        sin = block_info[:, 3:6]
        cos = block_info[:, 6:9]
        contact = block_info[:, 9:]

        euler = np.arctan2(sin, cos)
        quat = R.from_euler("xyz", euler, degrees=False).as_quat()

        X = np.concatenate(
            [
                X,
                xpos,
                quat,
                contact,
            ],
            axis=-1,
        )

    return X


def blocks_euler_to_quat(paths):
    return np.stack([blocks_euler_to_quat_2d(path) for path in paths], axis=0)


def blocks_process_cubes(env):
    def _fn(dataset):
        for key in ["observations", "next_observations"]:
            dataset[key] = blocks_quat_to_euler(dataset[key])
        return dataset

    return _fn


def blocks_remove_kuka(env):
    def _fn(dataset):
        for key in ["observations", "next_observations"]:
            dataset[key] = dataset[key][:, 7:]
        return dataset

    return _fn


def blocks_add_kuka(observations):
    """
    observations : [ batch_size x horizon x 32 ]
    """
    robot_dim = 7
    batch_size, horizon, _ = observations.shape
    observations = np.concatenate(
        [
            np.zeros((batch_size, horizon, 7)),
            observations,
        ],
        axis=-1,
    )
    return observations


def blocks_cumsum_quat(deltas):
    """
    deltas : [ batch_size x horizon x transition_dim ]
    """
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert deltas.shape[-1] == robot_dim + n_blocks * block_dim

    batch_size, horizon, _ = deltas.shape

    cumsum = deltas.cumsum(axis=1)
    for i in range(n_blocks):
        start = robot_dim + i * block_dim + 3
        end = start + 4

        quat = deltas[:, :, start:end].copy()

        quat = einops.rearrange(quat, "b h q -> (b h) q")
        euler = R.from_quat(quat).as_euler("xyz")
        euler = einops.rearrange(euler, "(b h) e -> b h e", b=batch_size)
        cumsum_euler = euler.cumsum(axis=1)

        cumsum_euler = einops.rearrange(cumsum_euler, "b h e -> (b h) e")
        cumsum_quat = R.from_euler("xyz", cumsum_euler).as_quat()
        cumsum_quat = einops.rearrange(
            cumsum_quat, "(b h) q -> b h q", b=batch_size
        )

        cumsum[:, :, start:end] = cumsum_quat.copy()

    return cumsum


def blocks_delta_quat_helper(observations, next_observations):
    """
    input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
        xyz: 3
        quat: 4
        contact: 1
    """
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert (
        observations.shape[-1]
        == next_observations.shape[-1]
        == robot_dim + n_blocks * block_dim
    )

    deltas = (next_observations - observations)[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]
        next_block_info = next_observations[:, start:end]

        xpos = block_info[:, :3]
        next_xpos = next_block_info[:, :3]

        quat = block_info[:, 3:-1]
        next_quat = next_block_info[:, 3:-1]

        contact = block_info[:, -1:]
        next_contact = next_block_info[:, -1:]

        delta_xpos = next_xpos - xpos
        delta_contact = next_contact - contact

        rot = R.from_quat(quat)
        next_rot = R.from_quat(next_quat)

        delta_quat = (next_rot * rot.inv()).as_quat()
        w = delta_quat[:, -1:]

        ## make w positive to avoid [0, 0, 0, -1]
        delta_quat = delta_quat * np.sign(w)

        ## apply rot then delta to ensure we end at next_rot
        ## delta * rot = next_rot * rot' * rot = next_rot
        next_euler = next_rot.as_euler("xyz")
        next_euler_check = (R.from_quat(delta_quat) * rot).as_euler("xyz")
        assert np.allclose(next_euler, next_euler_check)

        deltas = np.concatenate(
            [
                deltas,
                delta_xpos,
                delta_quat,
                delta_contact,
            ],
            axis=-1,
        )

    return deltas


def blocks_add_deltas(env):
    def _fn(dataset):
        deltas = blocks_delta_quat_helper(
            dataset["observations"], dataset["next_observations"]
        )
        # deltas = dataset['next_observations'] - dataset['observations']
        dataset["deltas"] = deltas
        return dataset

    return _fn
