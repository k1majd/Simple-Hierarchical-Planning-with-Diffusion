import argparse
import json
import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


MAZE_BOUNDS = {
    "maze2d-umaze-v1": (0, 5, 0, 5),
    "maze2d-medium-v1": (0, 8, 0, 8),
    "maze2d-large-v1": (0, 9, 0, 12),
}


def to_maze_coords(observations, env_name):
    """Convert raw [x, y, ...] observations to normalized (0-1) maze plotting coords."""
    obs = np.asarray(observations)[:, :2] + 0.5
    bounds = MAZE_BOUNDS[env_name]
    if len(bounds) == 4:
        _, iscale, _, jscale = bounds
        obs[:, 0] /= iscale
        obs[:, 1] /= jscale
    else:
        _, scale = bounds
        obs /= scale
    return obs


def _draw_maze_traj(
    ax,
    background,
    env_name,
    hl_plan,
    ll_sequence,
    rollout,
    rollout_color,
    title,
    target,
    init_obs,
):
    """Draw a single maze subplot with planned HL/LL trajectory and executed rollout."""
    extent = (0, 1, 1, 0)
    ax.imshow(
        background * 0.5, extent=extent, cmap=plt.cm.binary, vmin=0, vmax=1
    )

    # HL waypoints
    hl_arr = np.array(hl_plan)
    hl_coords = to_maze_coords(hl_arr, env_name)
    ax.plot(
        hl_coords[:, 1],
        hl_coords[:, 0],
        c="magenta",
        linewidth=1.5,
        linestyle="--",
        zorder=15,
        alpha=0.8,
        label="HL waypoints",
    )
    ax.scatter(
        hl_coords[:, 1],
        hl_coords[:, 0],
        c="magenta",
        s=25,
        zorder=16,
        alpha=0.8,
    )

    # LL planned state sequence
    ll_arr = np.array(ll_sequence)
    ll_coords = to_maze_coords(ll_arr, env_name)
    ax.plot(
        ll_coords[:, 1],
        ll_coords[:, 0],
        c="gray",
        linewidth=1.0,
        linestyle="-",
        zorder=10,
        alpha=0.55,
        label="LL plan (states)",
    )

    # Executed rollout (color-coded by timestep)
    roll_arr = np.array(rollout)
    roll_coords = to_maze_coords(roll_arr, env_name)
    ts_colors = plt.cm.jet(np.linspace(0, 1, len(roll_coords)))
    ax.plot(
        roll_coords[:, 1],
        roll_coords[:, 0],
        c=rollout_color,
        linewidth=1.6,
        zorder=20,
        label="Executed rollout",
    )
    ax.scatter(
        roll_coords[:, 1], roll_coords[:, 0], c=ts_colors, s=7, zorder=21
    )

    # Start / Goal markers
    start_c = to_maze_coords(np.array([init_obs]), env_name)[0]
    goal_c = to_maze_coords(np.array([[*target, 0.0, 0.0]]), env_name)[0]
    ax.scatter(
        start_c[1],
        start_c[0],
        color="lime",
        marker="o",
        s=90,
        edgecolors="black",
        linewidth=0.8,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        goal_c[1],
        goal_c[0],
        color="red",
        marker="X",
        s=90,
        zorder=30,
        label="Goal",
    )

    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=6)
    ax.axis("off")


def plot_episode_trajectories(
    env_name, background, episode_data, output_dir, dataset, epi
):
    """3-panel figure for one episode: P-ctrl | Open-loop | Closed-loop replan."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Episode {epi} – planned vs executed trajectories ({dataset})",
        fontsize=10,
    )

    modes_cfg = [
        ("p_controller", "#1f77b4", "P Controller"),
        ("open_loop_actions", "#ff7f0e", "Open-loop LL Actions"),
        ("closed_loop_replan", "#2ca02c", "Closed-loop Replan (1st action)"),
    ]

    for ax, (mode, color, label) in zip(axes, modes_cfg):
        score = episode_data["scores"][mode]
        _draw_maze_traj(
            ax=ax,
            background=background,
            env_name=env_name,
            hl_plan=episode_data["hl_plan"],
            ll_sequence=episode_data["ll_sequence"],
            rollout=episode_data["rollouts"][mode],
            rollout_color=color,
            title=f"{label}\nscore={score:.3f}",
            target=episode_data["target"],
            init_obs=episode_data["init_obs"],
        )

    plt.tight_layout()
    fig_path = join(output_dir, f"episode_{epi:02d}_{dataset}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare maze2d execution modes for hierarchical diffusion plans"
    )
    parser.add_argument("--dataset", default="maze2d-umaze-v1")
    parser.add_argument("--logbase", default="logs")
    parser.add_argument("--hl_loadpath", default="diffusion/H120_T64_J15")
    parser.add_argument("--ll_loadpath", default="diffusion/H16_T128_J1")
    parser.add_argument("--hl_epoch", default="latest")
    parser.add_argument("--ll_epoch", default="latest")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--output_dir", default="results/maze2d_exec_compare")
    return parser


def set_env_from_obs(env, obs4):
    """Reset env state to a given [x, y, vx, vy] observation."""
    env.set_state(obs4[:2], obs4[2:])
    return env._get_obs()


def build_hl_plan(hl_policy, hl_horizon, start_obs, target):
    hl_cond = {
        0: start_obs,
        hl_horizon - 1: np.array([*target, 0.0, 0.0], dtype=np.float32),
    }
    _, hl_samples = hl_policy(hl_cond, batch_size=1)
    return hl_samples.observations


def build_ll_outputs(ll_policy, ll_horizon, ll_normalizer, hl_plan):
    """Create dense LL states and dense LL actions for one HL plan."""
    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)

    ll_cond = {
        0: ll_cond_[:, 0],
        ll_horizon - 1: ll_cond_[:, -1],
    }

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)

    # Observations
    ll_obs = ll_samples.observations
    ll_obs = ll_obs.reshape(B, M - 1, ll_horizon, -1)
    ll_sequence = np.concatenate(
        [
            ll_obs[:, 0, :1],
            ll_obs[:, :, 1:].reshape(B, (M - 1) * (ll_horizon - 1), -1),
        ],
        axis=1,
    )[0]

    # Actions (LL model in this setup has action_dim > 0)
    ll_actions = None
    if ll_samples.actions is not None:
        ll_actions = ll_samples.actions.reshape(B, M - 1, ll_horizon, -1)
        ll_actions = ll_actions.reshape(B, (M - 1) * ll_horizon, -1)[0]
        # Policy returns normalized actions; unnormalize back to env action scale.
        ll_actions = ll_normalizer.unnormalize(ll_actions, "actions")

    return ll_sequence, ll_actions


def rollout_p_controller(env, init_obs, ll_sequence, max_steps):
    observation = set_env_from_obs(env, init_obs.copy())
    rollout = [observation.copy()]
    rewards = []

    for t in range(max_steps):
        if t < len(ll_sequence) - 1:
            next_waypoint = ll_sequence[t]
        else:
            next_waypoint = ll_sequence[-1].copy()
            next_waypoint[2:] = 0.0

        action = (
            next_waypoint[:2]
            - observation[:2]
            + (next_waypoint[2:] - observation[2:])
        )
        next_observation, reward, terminal, _ = env.step(action)

        rewards.append(float(reward))
        rollout.append(next_observation.copy())
        observation = next_observation
        if terminal:
            break

    return rollout, rewards


def rollout_open_loop_actions(env, init_obs, ll_actions, max_steps):
    observation = set_env_from_obs(env, init_obs.copy())
    rollout = [observation.copy()]
    rewards = []

    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for t in range(max_steps):
        action = (
            ll_actions[t]
            if (ll_actions is not None and t < len(ll_actions))
            else zero_action
        )
        next_observation, reward, terminal, _ = env.step(action)

        rewards.append(float(reward))
        rollout.append(next_observation.copy())
        observation = next_observation
        if terminal:
            break

    return rollout, rewards


def rollout_closed_loop_replan(
    env,
    init_obs,
    target,
    hl_policy,
    ll_policy,
    hl_horizon,
    ll_horizon,
    ll_normalizer,
    max_steps,
):
    """Replan at every step: infer full LL action plan, apply first action."""
    observation = set_env_from_obs(env, init_obs.copy())
    rollout = [observation.copy()]
    rewards = []

    for _ in range(max_steps):
        hl_plan = build_hl_plan(hl_policy, hl_horizon, observation, target)
        ll_sequence, ll_actions = build_ll_outputs(
            ll_policy, ll_horizon, ll_normalizer, hl_plan
        )

        if ll_actions is not None and len(ll_actions) > 0:
            action = ll_actions[0]
        else:
            # Fallback if actions are unavailable.
            next_waypoint = ll_sequence[0]
            action = (
                next_waypoint[:2]
                - observation[:2]
                + (next_waypoint[2:] - observation[2:])
            )

        next_observation, reward, terminal, _ = env.step(action)

        rewards.append(float(reward))
        rollout.append(next_observation.copy())
        observation = next_observation
        if terminal:
            break

    return rollout, rewards


def plot_comparison(all_results, summary, output_dir, dataset):
    modes = ["p_controller", "open_loop_actions", "closed_loop_replan"]
    labels = {
        "p_controller": "P Controller",
        "open_loop_actions": "Open-loop LL Actions",
        "closed_loop_replan": "Closed-loop Replan (first LL action)",
    }
    colors = {
        "p_controller": "#1f77b4",
        "open_loop_actions": "#ff7f0e",
        "closed_loop_replan": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: per-episode score traces.
    ax = axes[0]
    for mode in modes:
        rows = all_results[mode]
        xs = [r["episode"] for r in rows]
        ys = [r["score"] for r in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            color=colors[mode],
            label=labels[mode],
        )
    ax.set_title(f"Per-episode scores ({dataset})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Normalized score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Right panel: score distributions with mean markers.
    ax = axes[1]
    score_data = [[r["score"] for r in all_results[m]] for m in modes]
    bp = ax.boxplot(
        score_data, labels=[labels[m] for m in modes], patch_artist=True
    )
    for patch, mode in zip(bp["boxes"], modes):
        patch.set_facecolor(colors[mode])
        patch.set_alpha(0.35)

    means = [summary[m]["mean_score"] for m in modes]
    ax.scatter(
        np.arange(1, len(modes) + 1),
        means,
        color="black",
        marker="D",
        s=30,
        label="Mean",
    )
    ax.set_title("Score distribution")
    ax.set_ylabel("Normalized score")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = join(output_dir, f"compare_{dataset}.png")
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hl_exp = utils.load_diffusion(
        args.logbase,
        args.dataset,
        args.hl_loadpath,
        epoch=args.hl_epoch,
    )
    ll_exp = utils.load_diffusion(
        args.logbase,
        args.dataset,
        args.ll_loadpath,
        epoch=args.ll_epoch,
    )

    hl_policy = Policy(hl_exp.ema, hl_exp.dataset.normalizer)
    ll_policy = Policy(ll_exp.ema, ll_exp.dataset.normalizer)

    hl_horizon = hl_exp.ema.horizon
    ll_horizon = ll_exp.ema.horizon

    env = datasets.load_environment(args.dataset)
    target = np.array(env._target, dtype=np.float32).copy()
    maze_background = env.maze_arr == 10

    all_results = {
        "p_controller": [],
        "open_loop_actions": [],
        "closed_loop_replan": [],
    }

    for epi in range(args.n_episodes):
        env.reset()
        init_obs = env._get_obs().copy()

        hl_plan = build_hl_plan(hl_policy, hl_horizon, init_obs, target)
        ll_sequence, ll_actions = build_ll_outputs(
            ll_policy, ll_horizon, ll_exp.dataset.normalizer, hl_plan
        )

        r_rollout, r_rewards = rollout_p_controller(
            env, init_obs, ll_sequence, args.max_episode_steps
        )
        a_rollout, a_rewards = rollout_open_loop_actions(
            env, init_obs, ll_actions, args.max_episode_steps
        )
        c_rollout, c_rewards = rollout_closed_loop_replan(
            env,
            init_obs,
            target,
            hl_policy,
            ll_policy,
            hl_horizon,
            ll_horizon,
            ll_exp.dataset.normalizer,
            args.max_episode_steps,
        )

        episode_scores = {}
        for mode, rollout, rewards in [
            ("p_controller", r_rollout, r_rewards),
            ("open_loop_actions", a_rollout, a_rewards),
            ("closed_loop_replan", c_rollout, c_rewards),
        ]:
            ret = float(np.sum(rewards))
            score = float(env.get_normalized_score(ret))
            episode_scores[mode] = score
            result = {
                "episode": epi,
                "steps": len(rewards),
                "return": ret,
                "score": score,
                "terminal": bool(len(rewards) < args.max_episode_steps),
            }
            all_results[mode].append(result)

        print(
            f"Episode {epi}: "
            f"P={all_results['p_controller'][-1]['score']:.3f}, "
            f"OpenAction={all_results['open_loop_actions'][-1]['score']:.3f}, "
            f"ClosedLoop={all_results['closed_loop_replan'][-1]['score']:.3f}"
        )

        # Per-episode trajectory figure
        episode_fig_path = plot_episode_trajectories(
            env_name=args.dataset,
            background=maze_background,
            episode_data={
                "hl_plan": hl_plan[0],  # (M, obs_dim)
                "ll_sequence": ll_sequence,  # (T, obs_dim)
                "rollouts": {
                    "p_controller": r_rollout,
                    "open_loop_actions": a_rollout,
                    "closed_loop_replan": c_rollout,
                },
                "scores": episode_scores,
                "target": target,
                "init_obs": init_obs,
            },
            output_dir=args.output_dir,
            dataset=args.dataset,
            epi=epi,
        )
        print(f"  -> Saved trajectory plot: {episode_fig_path}")

    summary = {}
    for mode, rows in all_results.items():
        scores = np.array([r["score"] for r in rows], dtype=np.float32)
        returns = np.array([r["return"] for r in rows], dtype=np.float32)
        summary[mode] = {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
        }

    out = {
        "config": vars(args),
        "summary": summary,
        "episodes": all_results,
    }

    out_path = join(args.output_dir, f"compare_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    fig_path = plot_comparison(
        all_results=all_results,
        summary=summary,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )

    print("\n=== Summary ===")
    for mode in ["p_controller", "open_loop_actions", "closed_loop_replan"]:
        s = summary[mode]
        print(
            f"{mode:>20} | score {s['mean_score']:.3f} +/- {s['std_score']:.3f}"
            f" | return {s['mean_return']:.1f} +/- {s['std_return']:.1f}"
        )
    print(f"Saved comparison to: {out_path}")
    print(f"Saved comparison plot to: {fig_path}")


if __name__ == "__main__":
    main()
