"""
Hierarchical planning inference for the navigation (tractor-trailer) dataset.

Loads the trained HL/LL diffusion models (with saved dataset/model configs),
runs hierarchical planning from a start state to a goal state, and visualizes
the planned trajectory.

The models use 9-D sin/cos observations:
  [x, y, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), v, sin(δ), cos(δ)]

Usage:
  python scripts/hd_plan_navigation.py
  python scripts/hd_plan_navigation.py --start_goal_idx 0 --end_goal_idx 3 --n_samples 10
  python scripts/hd_plan_navigation.py --device cpu
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffuser.guides.policies import Policy
from diffuser.utils.rendering import NavigationRenderer
from diffuser.utils.serialization import load_diffusion


# ---------------------------------------------------------------------------
# Constants (must match generate_data.py / train_navigation.py)
# ---------------------------------------------------------------------------

GOAL_POINTS = [
    [15.0, 27.0, 0.0, 0.0],
    [-5.0, 37.0, -np.deg2rad(45), -np.deg2rad(45)],
    [1.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
    [15.0, 3.0, 0.0, 0.0],
    [35.0, -7.0, -np.deg2rad(45), -np.deg2rad(45)],
    [31.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
]

# -- HL config --
HL_HORIZON = 255
HL_JUMP = 15

# -- LL config --
LL_HORIZON = 16

# -- Log directories (absolute, relative to project root) --
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
HL_LOG_DIR = os.path.join(
    PROJECT_ROOT, "logs", "navigation", "diffusion", "H255_T256_J15"
)
LL_LOG_DIR = os.path.join(
    PROJECT_ROOT, "logs", "navigation", "diffusion", "H16_T128_J1"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def state4d_to_obs9d(state):
    """
    [x, y, θ₁, θ₂] → 9-D [x, y, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), v, sin(δ), cos(δ)].
    Assumes v=0, δ=0.
    """
    x, y, t1, t2 = state[:4]
    return np.array(
        [x, y, np.sin(t1), np.cos(t1), np.sin(t2), np.cos(t2), 0.0, 0.0, 1.0],
        dtype=np.float32,
    )


def obs9d_to_state4d(obs):
    """9-D → [x, y, θ₁, θ₂] using atan2 on sin/cos pairs."""
    x, y = float(obs[0]), float(obs[1])
    t1 = float(np.arctan2(obs[2], obs[3]))
    t2 = float(np.arctan2(obs[4], obs[5]))
    return np.array([x, y, t1, t2])


def load_model(log_dir, epoch):
    """Load a trained navigation diffusion model using saved configs."""
    experiment = load_diffusion(log_dir, epoch=epoch)
    return experiment.ema, experiment.dataset


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_hierarchical(hl_policy, ll_policy, start_obs, goal_obs):
    """
    Run one hierarchical plan: HL generates coarse waypoints,
    LL fills in fine trajectories between each pair.

    Returns:
        trajectory: (T, obs_dim) numpy array of sin/cos observations
        hl_waypoints: (M, obs_dim) numpy array of HL sub-goals
    """
    hl_horizon = HL_HORIZON // HL_JUMP  # 17

    # HL pass
    hl_cond = {
        0: start_obs,
        hl_horizon - 1: goal_obs,
    }
    _, hl_samples = hl_policy(hl_cond, batch_size=1)
    hl_plan = hl_samples.observations  # (1, M, obs_dim)

    # LL pass: condition on consecutive HL waypoint pairs
    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0],
        LL_HORIZON - 1: ll_cond_[:, -1],
    }

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_obs = ll_samples.observations  # (B*(M-1), LL_HORIZON, obs_dim)
    ll_obs = ll_obs.reshape(B, M - 1, LL_HORIZON, -1)

    # Concatenate: first point + skip first of each LL segment to avoid duplicates
    trajectory = np.concatenate(
        [
            ll_obs[:, 0, :1],
            ll_obs[:, :, 1:].reshape(B, (M - 1) * (LL_HORIZON - 1), -1),
        ],
        axis=1,
    )[
        0
    ]  # (T, obs_dim)

    hl_waypoints = hl_plan[0]  # (M, obs_dim)
    return trajectory, hl_waypoints


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_plans(
    all_trajectories, all_hl_waypoints, start_state, goal_state, args
):
    renderer = NavigationRenderer()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: all samples overlaid ---
    ax = axes[0]
    renderer._draw_obstacles(ax)

    cmap = plt.cm.tab10
    for idx, traj in enumerate(all_trajectories):
        color = cmap(idx / max(len(all_trajectories) - 1, 1))
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linewidth=1.2,
            alpha=0.7,
            label=f"Sample {idx}" if idx < 10 else None,
        )
    ax.scatter(
        start_state[0],
        start_state[1],
        color="green",
        marker="o",
        s=120,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        goal_state[0],
        goal_state[1],
        color="red",
        marker="X",
        s=120,
        zorder=30,
        label="Goal",
    )
    for gi, gp in enumerate(GOAL_POINTS):
        ax.scatter(gp[0], gp[1], color="orange", marker="*", s=80, zorder=25)
        ax.text(gp[0] + 0.5, gp[1] + 0.5, f"G{gi}", fontsize=8)
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 40)
    ax.set_aspect("equal")
    ax.set_title(
        f"Planned trajectories: G{args.start_goal_idx} → G{args.end_goal_idx} "
        f"({len(all_trajectories)} samples)"
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --- Right: first sample with HL waypoints ---
    ax = axes[1]
    renderer._draw_obstacles(ax)

    traj = all_trajectories[0]
    hl_wp = all_hl_waypoints[0]
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color="blue",
        linewidth=1.5,
        label="LL trajectory",
    )
    ax.scatter(
        hl_wp[:, 0],
        hl_wp[:, 1],
        color="magenta",
        marker="D",
        s=50,
        zorder=25,
        label="HL waypoints",
    )

    # Draw heading arrows (θ₁) at each HL waypoint
    arrow_len = 1.5
    for i in range(len(hl_wp)):
        ax.arrow(
            hl_wp[i, 0],
            hl_wp[i, 1],
            arrow_len * np.cos(hl_wp[i, 2]),
            arrow_len * np.sin(hl_wp[i, 2]),
            head_width=0.5,
            head_length=0.3,
            fc="magenta",
            ec="magenta",
            alpha=0.5,
        )

    ax.scatter(
        start_state[0],
        start_state[1],
        color="green",
        marker="o",
        s=120,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        goal_state[0],
        goal_state[1],
        color="red",
        marker="X",
        s=120,
        zorder=30,
        label="Goal",
    )
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 40)
    ax.set_aspect("equal")
    ax.set_title("Sample 0: HL waypoints + LL trajectory")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    fig_path = os.path.join(
        args.output_dir,
        f"plan_G{args.start_goal_idx}_to_G{args.end_goal_idx}.png",
    )
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical diffusion planning for tractor-trailer navigation"
    )
    parser.add_argument(
        "--start_goal_idx",
        type=int,
        default=0,
        help="Index into GOAL_POINTS for the start position (default: 0)",
    )
    parser.add_argument(
        "--end_goal_idx",
        type=int,
        default=4,
        help="Index into GOAL_POINTS for the goal position (default: 4)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of planning samples to generate (default: 5)",
    )
    parser.add_argument("--hl_epoch", default="latest")
    parser.add_argument("--ll_epoch", default="latest")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_dir", default="results/navigation_plans")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load trained models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading HL model...")
    hl_ema, hl_dataset = load_model(HL_LOG_DIR, args.hl_epoch)

    print("Loading LL model...")
    ll_ema, ll_dataset = load_model(LL_LOG_DIR, args.ll_epoch)

    # ------------------------------------------------------------------
    # 2. Create policies
    # ------------------------------------------------------------------
    hl_policy = Policy(hl_ema, hl_dataset.normalizer)
    ll_policy = Policy(ll_ema, ll_dataset.normalizer)

    # ------------------------------------------------------------------
    # 3. Plan
    # ------------------------------------------------------------------
    start_state = GOAL_POINTS[args.start_goal_idx]
    goal_state = GOAL_POINTS[args.end_goal_idx]
    start_obs = state4d_to_obs9d(start_state)
    goal_obs = state4d_to_obs9d(goal_state)

    print("=" * 60)
    print(f"Planning: G{args.start_goal_idx} → G{args.end_goal_idx}")
    print(f"  Start (4D): {start_state}")
    print(f"  Goal  (4D): {goal_state}")
    print(f"  Samples:    {args.n_samples}")
    print(
        f"  HL horizon: {HL_HORIZON // HL_JUMP} waypoints  (H={HL_HORIZON}, J={HL_JUMP})"
    )
    print(f"  LL horizon: {LL_HORIZON} steps per segment")

    all_trajectories = []  # list of (T, 4) raw-angle trajectories
    all_hl_waypoints = []  # list of (M, 4) raw-angle HL waypoints

    for i in range(args.n_samples):
        with torch.no_grad():
            trajectory_sincos, hl_wp_sincos = plan_hierarchical(
                hl_policy, ll_policy, start_obs, goal_obs
            )

        raw_traj = np.array([obs9d_to_state4d(o) for o in trajectory_sincos])
        raw_hl_wp = np.array([obs9d_to_state4d(o) for o in hl_wp_sincos])

        all_trajectories.append(raw_traj)
        all_hl_waypoints.append(raw_hl_wp)

        print(
            f"  [{i+1}/{args.n_samples}] "
            f"{len(raw_traj)} pts | "
            f"start=({raw_traj[0,0]:.1f}, {raw_traj[0,1]:.1f}) "
            f"end=({raw_traj[-1,0]:.1f}, {raw_traj[-1,1]:.1f})"
        )

    # ------------------------------------------------------------------
    # 4. Visualize
    # ------------------------------------------------------------------
    print("=" * 60)
    plot_plans(
        all_trajectories, all_hl_waypoints, start_state, goal_state, args
    )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    json_path = os.path.join(
        args.output_dir,
        f"plan_G{args.start_goal_idx}_to_G{args.end_goal_idx}.json",
    )
    json_data = {
        "start_goal_idx": args.start_goal_idx,
        "end_goal_idx": args.end_goal_idx,
        "start_state": start_state,
        "goal_state": goal_state,
        "hl_epoch": args.hl_epoch,
        "ll_epoch": args.ll_epoch,
        "n_samples": args.n_samples,
        "trajectory_length": len(all_trajectories[0]),
        "trajectories": [t.tolist() for t in all_trajectories],
        "hl_waypoints": [w.tolist() for w in all_hl_waypoints],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved results → {json_path}")


if __name__ == "__main__":
    main()
