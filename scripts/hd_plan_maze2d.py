import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import os
import matplotlib.pyplot as plt


MAZE_BOUNDS = {
    "maze2d-umaze-v1": (0, 5, 0, 5),
    "maze2d-medium-v1": (0, 8, 0, 8),
    "maze2d-large-v1": (0, 9, 0, 12),
}


def to_maze_coords(observations, env_name):
    """Convert raw observations to normalized (0-1) maze plotting coords."""
    obs = observations[:, :2] + 0.5
    bounds = MAZE_BOUNDS[env_name]
    if len(bounds) == 2:
        _, scale = bounds
        obs = obs / scale
    elif len(bounds) == 4:
        _, iscale, _, jscale = bounds
        obs[:, 0] = obs[:, 0] / iscale
        obs[:, 1] = obs[:, 1] / jscale
    return obs


def plot_sample(
    env_name,
    background,
    rollout,
    hl_plan,
    target,
    init_obs,
    score,
    idx,
    savepath,
):
    """Render a single sample: maze background + HL waypoints + executed rollout."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = (0, 1, 1, 0)

    # --- Left: HL plan (diffusion output) ---
    ax = axes[0]
    ax.imshow(
        background * 0.5, extent=extent, cmap=plt.cm.binary, vmin=0, vmax=1
    )
    hl_coords = to_maze_coords(hl_plan, env_name)
    colors_hl = plt.cm.magma(np.linspace(0, 1, len(hl_coords)))
    ax.plot(
        hl_coords[:, 1],
        hl_coords[:, 0],
        c="magenta",
        linewidth=1.5,
        zorder=10,
        label="HL plan",
    )
    ax.scatter(hl_coords[:, 1], hl_coords[:, 0], c=colors_hl, s=20, zorder=20)
    # Mark start/goal
    start_c = to_maze_coords(init_obs[np.newaxis, :], env_name)[0]
    goal_c = to_maze_coords(np.array([[*target, 0, 0]]), env_name)[0]
    ax.scatter(
        start_c[1],
        start_c[0],
        color="green",
        marker="o",
        s=100,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        goal_c[1],
        goal_c[0],
        color="red",
        marker="X",
        s=100,
        zorder=30,
        label="Goal",
    )
    ax.set_title(f"Sample {idx}: HL waypoints")
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    # --- Right: Executed rollout ---
    ax = axes[1]
    ax.imshow(
        background * 0.5, extent=extent, cmap=plt.cm.binary, vmin=0, vmax=1
    )
    roll_arr = np.array(rollout)
    roll_coords = to_maze_coords(roll_arr, env_name)
    colors_roll = plt.cm.jet(np.linspace(0, 1, len(roll_coords)))
    ax.plot(
        roll_coords[:, 1],
        roll_coords[:, 0],
        c="black",
        linewidth=1.0,
        zorder=10,
        label="Rollout",
    )
    ax.scatter(
        roll_coords[:, 1], roll_coords[:, 0], c=colors_roll, s=8, zorder=20
    )
    # Also overlay HL waypoints faintly
    ax.scatter(
        hl_coords[:, 1],
        hl_coords[:, 0],
        c="magenta",
        marker="D",
        s=15,
        alpha=0.4,
        zorder=15,
        label="HL waypoints",
    )
    ax.scatter(
        start_c[1],
        start_c[0],
        color="green",
        marker="o",
        s=100,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        goal_c[1],
        goal_c[0],
        color="red",
        marker="X",
        s=100,
        zorder=30,
        label="Goal",
    )
    ax.set_title(f"Sample {idx}: Rollout (score={score:.3f})")
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    plt.tight_layout()
    fig_path = join(savepath, f"idx{idx}_render.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved render → {fig_path}")


def plot_summary(
    env_name, background, all_rollouts, all_scores, target, savepath
):
    """Render all rollouts overlaid on the maze."""
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = (0, 1, 1, 0)
    ax.imshow(
        background * 0.5, extent=extent, cmap=plt.cm.binary, vmin=0, vmax=1
    )

    cmap = plt.cm.tab10
    for idx, rollout in enumerate(all_rollouts):
        roll_arr = np.array(rollout)
        roll_coords = to_maze_coords(roll_arr, env_name)
        color = cmap(idx / max(len(all_rollouts) - 1, 1))
        ax.plot(
            roll_coords[:, 1],
            roll_coords[:, 0],
            color=color,
            linewidth=1.2,
            alpha=0.7,
            label=f"S{idx} ({all_scores[idx]:.2f})" if idx < 10 else None,
        )

    goal_c = to_maze_coords(np.array([[*target, 0, 0]]), env_name)[0]
    ax.scatter(
        goal_c[1],
        goal_c[0],
        color="red",
        marker="X",
        s=120,
        zorder=30,
        label="Goal",
    )
    ax.set_title(
        f"All rollouts ({len(all_rollouts)} samples, "
        f"mean score={np.mean(all_scores):.3f})"
    )
    ax.legend(loc="upper right", fontsize=6)
    ax.axis("off")

    plt.tight_layout()
    fig_path = join(savepath, "summary_render.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary → {fig_path}")


class HLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d_hl"


hl_args = HLParser().parse_args("plan")


class LLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d_ll"


ll_args = LLParser().parse_args("plan")
# ---------------------------------- setup ----------------------------------#

# ---------------------------------- loading ----------------------------------#


n_samples = 10

loadpath = (hl_args.logbase, hl_args.dataset, hl_args.diffusion_loadpath)


hl_diffusion_experiment = utils.load_diffusion(
    hl_args.logbase,
    hl_args.dataset,
    hl_args.diffusion_loadpath,
    epoch=hl_args.diffusion_epoch,
)
hl_diffusion = hl_diffusion_experiment.ema
dataset = hl_diffusion_experiment.dataset
hl_policy = Policy(hl_diffusion, dataset.normalizer)

ll_diffusion_experiment = utils.load_diffusion(
    ll_args.logbase,
    ll_args.dataset,
    ll_args.diffusion_loadpath,
    epoch=ll_args.diffusion_epoch,
)
ll_diffusion = ll_diffusion_experiment.ema
ll_policy = Policy(ll_diffusion, dataset.normalizer)

env_eval = datasets.load_environment(hl_args.dataset)

target = env_eval._target
hl_cond = {
    hl_diffusion.horizon - 1: np.array([*target, 0, 0]),
}

total_rewards = []
scores = []
rollouts = []
plans = []
track_action = []

# Get maze background for rendering
maze_background = env_eval.maze_arr == 10


for i in range(n_samples):
    observation = env_eval.reset()
    init_obs = observation.copy()
    observation = env_eval._get_obs()
    rollout = [observation.copy()]

    hl_cond[0] = observation
    action, samples = hl_policy(hl_cond, batch_size=hl_args.batch_size)
    hl_plan = samples.observations

    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0],
        ll_args.horizon - 1: ll_cond_[:, -1],
    }

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_samples = ll_samples.observations
    ll_samples = ll_samples.reshape(B, (M - 1), ll_args.horizon, -1)
    ll_samples = np.concatenate(
        [
            ll_samples[:, 0, :1],
            ll_samples[:, :, 1:].reshape(B, (M - 1) * hl_args.jump, -1),
        ],
        axis=1,
    )
    ll_sequence = ll_samples[0]
    total_reward = []
    action_list = []

    max_episode_steps = env_eval.max_episode_steps
    finished = False
    t = 0
    while t < max_episode_steps:
        if finished:
            break
        else:
            if t < len(ll_sequence) - 1:
                next_waypoint = ll_sequence[t]
            else:
                next_waypoint = ll_sequence[-1].copy()
                next_waypoint[2:] = 0

            state = observation.copy()
            action = (
                next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            )

            next_observation, reward, terminal, _ = env_eval.step(action)
            t += 1
            total_reward.append(reward)
            score = env_eval.get_normalized_score(sum(total_reward))

            ## update rollout observations
            rollout.append(next_observation.copy())
            if terminal or t >= max_episode_steps:
                finished = True
                print(
                    f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {sum(total_reward):.2f} | score: {score:.4f} | "
                )
                break
            observation = next_observation

    rollouts.append(rollout)
    total_rewards.append(total_reward)
    scores.append(env_eval.get_normalized_score(sum(total_reward)))
    plans.append(hl_plan[0])  # (M, obs_dim)

    ## save result as a json file
    json_path = join(hl_args.savepath, f"idx{i}_rollout.json")
    json_data = {
        "score": score,
        "step": t,
        "return": total_reward,
        "term": terminal,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, sort_keys=True)

    ## render per-sample visualization
    plot_sample(
        hl_args.dataset,
        maze_background,
        rollout,
        hl_plan[0],
        target,
        init_obs,
        score,
        i,
        hl_args.savepath,
    )

## render summary of all rollouts
plot_summary(
    hl_args.dataset,
    maze_background,
    rollouts,
    scores,
    target,
    hl_args.savepath,
)
print(f"\nDone. Mean score: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}")
