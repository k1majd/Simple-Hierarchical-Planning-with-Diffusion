"""
High-level planning inference for the navigation (tractor-trailer) dataset.

Loads the trained HL diffusion model from a log directory that contains all
saved configs (dataset_config.pkl, model_config.pkl, etc.) and generates
coarse waypoint plans between a start and goal state.

The model uses 9-D sin/cos observations:
  [x, y, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), v, sin(δ), cos(δ)]

Usage (from the hierarchical_diffusion directory):
  python scripts/hl_plan_navigation.py
  python scripts/hl_plan_navigation.py --start_goal_idx 0 --end_goal_idx 3 --n_samples 10
  python scripts/hl_plan_navigation.py --epoch 1960000
  python scripts/hl_plan_navigation.py --device cpu
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
# Constants
# ---------------------------------------------------------------------------

GOAL_POINTS = [
    [15.0, 27.0, 0.0, 0.0],
    [-5.0, 37.0, -np.deg2rad(45), -np.deg2rad(45)],
    [1.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
    [15.0, 3.0, 0.0, 0.0],
    [35.0, -7.0, -np.deg2rad(45), -np.deg2rad(45)],
    [31.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
]

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
HL_LOG_DIR = os.path.join(
    PROJECT_ROOT, "logs", "navigation", "diffusion", "H255_T256_J15_obs_only"
)
HL_HORIZON = 255
HL_JUMP = 15


# ---------------------------------------------------------------------------
# Helpers: 9-D sin/cos encoding
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_plans(all_waypoints, start_state, goal_state, args):
    renderer = NavigationRenderer()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: all samples overlaid ---
    ax = axes[0]
    renderer._draw_obstacles(ax)

    cmap = plt.cm.tab10
    for idx, wp in enumerate(all_waypoints):
        color = cmap(idx / max(len(all_waypoints) - 1, 1))
        ax.plot(
            wp[:, 0],
            wp[:, 1],
            "o-",
            color=color,
            linewidth=1.2,
            markersize=4,
            alpha=0.7,
            label=f"Sample {idx}" if idx < 10 else None,
        )

    ax.scatter(
        *start_state[:2],
        color="green",
        marker="o",
        s=120,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        *goal_state[:2], color="red", marker="X", s=120, zorder=30, label="Goal"
    )

    for gi, gp in enumerate(GOAL_POINTS):
        ax.scatter(gp[0], gp[1], color="orange", marker="*", s=80, zorder=25)
        ax.text(gp[0] + 0.5, gp[1] + 0.5, f"G{gi}", fontsize=8)

    ax.set_xlim(-10, 45)
    ax.set_ylim(-15, 45)
    ax.set_aspect("equal")
    ax.set_title(
        f"HL waypoints: G{args.start_goal_idx} → G{args.end_goal_idx} "
        f"({len(all_waypoints)} samples)"
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --- Right: first sample with heading arrows ---
    ax = axes[1]
    renderer._draw_obstacles(ax)

    wp = all_waypoints[0]
    ax.plot(
        wp[:, 0],
        wp[:, 1],
        "o-",
        color="blue",
        linewidth=1.5,
        markersize=5,
        label="HL waypoints",
    )

    # Draw heading arrows (θ₁) at each waypoint
    arrow_len = 1.5
    for i in range(len(wp)):
        ax.arrow(
            wp[i, 0],
            wp[i, 1],
            arrow_len * np.cos(wp[i, 2]),
            arrow_len * np.sin(wp[i, 2]),
            head_width=0.5,
            head_length=0.3,
            fc="blue",
            ec="blue",
            alpha=0.5,
        )

    ax.scatter(
        *start_state[:2],
        color="green",
        marker="o",
        s=120,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        *goal_state[:2], color="red", marker="X", s=120, zorder=30, label="Goal"
    )

    ax.set_xlim(-10, 45)
    ax.set_ylim(-15, 45)
    ax.set_aspect("equal")
    ax.set_title("Sample 0: HL waypoints with headings")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    fig_path = os.path.join(
        args.output_dir,
        f"hl_plan_G{args.start_goal_idx}_to_G{args.end_goal_idx}.png",
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
        description="High-level diffusion planning for tractor-trailer navigation"
    )
    parser.add_argument(
        "--start_goal_idx",
        type=int,
        default=5,
        help="Index into GOAL_POINTS for the start position (default: 5)",
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
        default=5,
        help="Number of planning samples to generate (default: 5)",
    )
    parser.add_argument("--epoch", default="latest")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log_dir", default=HL_LOG_DIR)
    parser.add_argument("--output_dir", default="results/navigation_plans")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load trained HL model (configs all saved in log_dir)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Loading HL model from {args.log_dir} ...")
    experiment = load_diffusion(args.log_dir, epoch=args.epoch)
    hl_ema = experiment.ema
    dataset = experiment.dataset

    hl_policy = Policy(hl_ema, dataset.normalizer)
    hl_waypoint_count = HL_HORIZON // HL_JUMP  # 17

    # ------------------------------------------------------------------
    # 2. Plan
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
    print(f"  HL waypoints: {hl_waypoint_count}  (H={HL_HORIZON}, J={HL_JUMP})")

    hl_cond_template = {
        0: start_obs,
        hl_waypoint_count - 1: goal_obs,
    }

    all_waypoints = []
    for i in range(args.n_samples):
        with torch.no_grad():
            _, samples = hl_policy(hl_cond_template, batch_size=1)

        obs9d = samples.observations[0]  # (M, 9)
        wp_4d = np.array([obs9d_to_state4d(o) for o in obs9d])
        all_waypoints.append(wp_4d)

        print(
            f"  [{i+1}/{args.n_samples}] "
            f"{len(wp_4d)} waypoints | "
            f"start=({wp_4d[0,0]:.1f}, {wp_4d[0,1]:.1f}) "
            f"end=({wp_4d[-1,0]:.1f}, {wp_4d[-1,1]:.1f})"
        )

    # ------------------------------------------------------------------
    # 3. Visualize
    # ------------------------------------------------------------------
    print("=" * 60)
    plot_plans(all_waypoints, start_state, goal_state, args)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    json_path = os.path.join(
        args.output_dir,
        f"hl_plan_G{args.start_goal_idx}_to_G{args.end_goal_idx}.json",
    )
    json_data = {
        "start_goal_idx": args.start_goal_idx,
        "end_goal_idx": args.end_goal_idx,
        "start_state": start_state,
        "goal_state": goal_state,
        "epoch": args.epoch,
        "log_dir": args.log_dir,
        "n_samples": args.n_samples,
        "hl_waypoint_count": len(all_waypoints[0]),
        "waypoints": [w.tolist() for w in all_waypoints],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved results → {json_path}")


if __name__ == "__main__":
    main()
