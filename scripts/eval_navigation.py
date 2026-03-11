"""
Evaluation script for trained hierarchical diffusion navigation models.

Runs three test suites:
  1. In-distribution:  start→goal pairs taken directly from training episodes.
  2. All goal pairs:   every combination of the 6 known GOAL_POINTS.
  3. Out-of-distribution: goals outside the training convex hull.

Metrics reported per test:
  - Goal reach error   (Euclidean distance of trajectory end to goal, xy only)
  - Start fidelity     (Euclidean distance of trajectory start to requested start)
  - Path smoothness    (mean step-to-step displacement)
  - Heading error      (angular distance of final heading to goal heading)

Usage (from the hierarchical_diffusion directory):
  python scripts/eval_navigation.py
  python scripts/eval_navigation.py --n_samples 10 --device cpu
  python scripts/eval_navigation.py --suites in_dist ood
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import diffuser.utils as utils
from diffuser.guides.policies import Policy
from diffuser.datasets.h5_sequence import H5GoalDataset
from diffuser.utils.rendering import NavigationRenderer
from diffuser.utils.serialization import (
    load_config,
    load_diffusion,
    get_latest_epoch,
)


# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
H5_PATH = os.environ.get(
    "NAV_H5_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "results",
        "planning_data_7000.h5",
    ),
)
FINAL_GOAL = [31.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)]

GOAL_POINTS = [
    [15.0, 27.0, 0.0, 0.0],
    [-5.0, 37.0, -np.deg2rad(45), -np.deg2rad(45)],
    [1.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
    [15.0, 3.0, 0.0, 0.0],
    [35.0, -7.0, -np.deg2rad(45), -np.deg2rad(45)],
    [31.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
]

HL_HORIZON = 255
HL_JUMP = 15
HL_JUMP_ACTION = "none"
LL_HORIZON = 16
LL_JUMP = 1
LL_JUMP_ACTION = False

HL_LOG_DIR = "logs/navigation/diffusion/H255_T256_J15_old"
LL_LOG_DIR = "logs/navigation/diffusion/H16_T128_J1_old"


# ---------------------------------------------------------------------------
# Helpers (must match training observation layout)
# ---------------------------------------------------------------------------
def state4d_to_obs6d(state):
    x, y, t1, t2 = state[:4]
    return np.array([x, y, t1, t2, 0.0, 0.0], dtype=np.float32)


def obs6d_to_state4d(obs):
    return np.array(
        [float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])]
    )


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def angular_distance(a, b):
    """Absolute angular distance between two angles."""
    return abs(wrap_angle(a - b))


# ---------------------------------------------------------------------------
# Reference trajectory extraction from training data
# ---------------------------------------------------------------------------
def load_reference_episodes(h5_path):
    """
    Load raw episodes from the H5 file.
    Episodes cycle: G0→G1, G1→G2, G2→G3, G3→G4, G4→G5, G5→G0, ...
    Returns a dict mapping (start_goal_idx, end_goal_idx) to a list of
    episode observation arrays (each shape (51, 6)).
    """
    with h5py.File(h5_path, "r") as f:
        obs_flat = np.array(f["observations"])  # (N*51, 6)
        terms = np.array(f["terminals"])
        timeouts = np.array(f["timeouts"])

    ep_ends = np.where(terms | timeouts)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])

    # Build mapping: consecutive goal pairs cycle with period 6
    # Episode k connects GOAL_POINTS[k%6] → GOAL_POINTS[(k+1)%6]
    n_goals = len(GOAL_POINTS)
    episodes_by_pair = {}  # (i, j) → list of obs arrays
    for ep_idx, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        src = ep_idx % n_goals
        dst = (ep_idx + 1) % n_goals
        key = (src, dst)
        if key not in episodes_by_pair:
            episodes_by_pair[key] = []
        episodes_by_pair[key].append(obs_flat[s : e + 1])

    return episodes_by_pair


def get_reference_trajectory(episodes_by_pair, start_goal_idx, end_goal_idx):
    """
    Get a single reference trajectory connecting GOAL_POINTS[start_goal_idx]
    to GOAL_POINTS[end_goal_idx] by chaining consecutive episodes.
    Returns (N, 4) array of [x, y, t1, t2] or None if not possible.
    """
    n_goals = len(GOAL_POINTS)
    if start_goal_idx == end_goal_idx:
        return None

    # Find the shortest forward chain through the cyclic goal sequence
    chain = []
    cur = start_goal_idx
    for _ in range(n_goals):
        nxt = (cur + 1) % n_goals
        chain.append((cur, nxt))
        if nxt == end_goal_idx:
            break
        cur = nxt
    else:
        return None  # shouldn't happen for valid indices

    # Pick the first available episode for each leg
    segments = []
    for pair in chain:
        if pair not in episodes_by_pair or len(episodes_by_pair[pair]) == 0:
            return None
        ep_obs = episodes_by_pair[pair][0]  # (51, 6)
        segments.append(ep_obs[:, :4])  # keep x, y, t1, t2

    # Chain: first segment fully, then skip first point of subsequent segments
    parts = [segments[0]]
    for seg in segments[1:]:
        parts.append(seg[1:])
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(log_dir, epoch, dataset=None, renderer=None):
    has_full_configs = os.path.exists(
        os.path.join(log_dir, "dataset_config.pkl")
    ) and os.path.exists(os.path.join(log_dir, "render_config.pkl"))

    if has_full_configs:
        experiment = load_diffusion(log_dir, epoch=epoch)
        return experiment.ema, experiment.dataset

    assert dataset is not None
    model_config = load_config(log_dir, "model_config.pkl")
    diffusion_config = load_config(log_dir, "diffusion_config.pkl")
    trainer_config = load_config(log_dir, "trainer_config.pkl")
    trainer_config._dict["results_folder"] = log_dir

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == "latest":
        epoch = get_latest_epoch((log_dir,))
    print(f"[ load_model ] {log_dir} | epoch {epoch}")
    if epoch != -1:
        trainer.load(epoch)

    return trainer.ema_model, dataset


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def plan_hierarchical(hl_policy, ll_policy, start_obs, goal_obs):
    hl_horizon = HL_HORIZON // HL_JUMP

    hl_cond = {0: start_obs, hl_horizon - 1: goal_obs}
    _, hl_samples = hl_policy(hl_cond, batch_size=1)
    hl_plan = hl_samples.observations

    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {0: ll_cond_[:, 0], LL_HORIZON - 1: ll_cond_[:, -1]}

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_obs = ll_samples.observations.reshape(B, M - 1, LL_HORIZON, -1)

    trajectory = np.concatenate(
        [
            ll_obs[:, 0, :1],
            ll_obs[:, :, 1:].reshape(B, (M - 1) * (LL_HORIZON - 1), -1),
        ],
        axis=1,
    )[0]
    hl_waypoints = hl_plan[0]
    return trajectory, hl_waypoints


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trajectory, start_state, goal_state):
    """Compute evaluation metrics for a single planned trajectory."""
    traj_start = trajectory[0, :4]
    traj_end = trajectory[-1, :4]

    goal_xy_err = np.linalg.norm(traj_end[:2] - np.array(goal_state[:2]))
    start_xy_err = np.linalg.norm(traj_start[:2] - np.array(start_state[:2]))

    heading_err = angular_distance(traj_end[2], goal_state[2])
    hitch_err = angular_distance(traj_end[3], goal_state[3])

    # Smoothness: mean step-to-step xy displacement
    diffs = np.diff(trajectory[:, :2], axis=0)
    step_sizes = np.linalg.norm(diffs, axis=1)
    smoothness = step_sizes.mean()
    max_step = step_sizes.max()

    return {
        "goal_xy_err": float(goal_xy_err),
        "start_xy_err": float(start_xy_err),
        "heading_err_deg": float(np.rad2deg(heading_err)),
        "hitch_err_deg": float(np.rad2deg(hitch_err)),
        "smoothness": float(smoothness),
        "max_step": float(max_step),
        "path_length": float(step_sizes.sum()),
    }


def aggregate_metrics(all_metrics):
    """Compute mean and std across a list of metric dicts."""
    keys = all_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------
def build_in_distribution_tests():
    """
    In-distribution tests: consecutive GOAL_POINT pairs as they appear in
    training data (G0→G1, G1→G2, ..., G5→G0).
    """
    tests = []
    n = len(GOAL_POINTS)
    for i in range(n):
        j = (i + 1) % n
        tests.append(
            {
                "name": f"G{i}→G{j}",
                "start": GOAL_POINTS[i],
                "goal": GOAL_POINTS[j],
            }
        )
    return tests


def build_all_pairs_tests():
    """All pairwise combinations of the 6 GOAL_POINTS (30 pairs)."""
    tests = []
    n = len(GOAL_POINTS)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tests.append(
                {
                    "name": f"G{i}→G{j}",
                    "start": GOAL_POINTS[i],
                    "goal": GOAL_POINTS[j],
                }
            )
    return tests


def build_ood_tests():
    """
    Out-of-distribution tests: goals outside the training goal set.
    Includes points near the boundary and clearly outside.

    Training spatial coverage: x ∈ [-19.7, 42.4], y ∈ [-16.5, 45.9]
    Training goal x ∈ [-5, 35], goal y ∈ [-7, 37]
    """
    tests = [
        # Near-OOD: shifted versions of known goals (outside obstacle x∈[4,28] y∈[8,22])
        {
            "name": "near_ood_1",
            "start": [15.0, 27.0, 0.0, 0.0],
            "goal": [20.0, 25.0, -np.deg2rad(45), -np.deg2rad(45)],
        },
        {
            "name": "near_ood_2",
            "start": [1.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
            "goal": [30.0, 5.0, 0.0, 0.0],
        },
        {
            "name": "near_ood_3",
            "start": [15.0, 3.0, 0.0, 0.0],
            "goal": [10.0, 30.0, -np.deg2rad(30), -np.deg2rad(30)],
        },
        # Far-OOD: outside spatial coverage of training data
        {
            "name": "far_ood_1",
            "start": [15.0, 27.0, 0.0, 0.0],
            "goal": [35.0, 37.0, np.deg2rad(45), np.deg2rad(45)],
        },
        {
            "name": "far_ood_2",
            "start": [15.0, 27.0, 0.0, 0.0],
            "goal": [-15.0, 0.0, -np.deg2rad(90), -np.deg2rad(90)],
        },
        {
            "name": "far_ood_3",
            "start": [15.0, 27.0, 0.0, 0.0],
            "goal": [40.0, 40.0, 0.0, 0.0],
        },
        # OOD start + in-distribution goal
        {
            "name": "ood_start_1",
            "start": [40.0, 40.0, 0.0, 0.0],
            "goal": [1.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
        },
        # Reverse direction (some may not appear in training)
        {
            "name": "reverse_1",
            "start": [31.0, 15.0, -np.deg2rad(90), -np.deg2rad(90)],
            "goal": [15.0, 27.0, 0.0, 0.0],
        },
    ]
    return tests


SUITES = {
    "in_dist": (
        "In-Distribution (consecutive goals)",
        build_in_distribution_tests,
    ),
    "all_pairs": ("All Goal Pairs", build_all_pairs_tests),
    "ood": ("Out-of-Distribution", build_ood_tests),
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_suite_results(
    suite_name, test_results, output_dir, ref_trajectories=None
):
    """Plot all test cases in a suite on a grid."""
    n = len(test_results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False
    )
    renderer = NavigationRenderer()

    for idx, (test, results) in enumerate(test_results):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        renderer._draw_obstacles(ax)

        # Plot reference trajectory from data (if available)
        if ref_trajectories is not None and test["name"] in ref_trajectories:
            ref = ref_trajectories[test["name"]]
            ax.plot(
                ref[:, 0],
                ref[:, 1],
                color="black",
                linewidth=2.0,
                alpha=0.8,
                linestyle="--",
                label="Data ref",
                zorder=15,
            )

        for sample_idx, (traj, _) in enumerate(results["samples"]):
            label = "Planned" if sample_idx == 0 else None
            ax.plot(
                traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.5, label=label
            )

        start = test["start"]
        goal = test["goal"]
        ax.scatter(
            start[0], start[1], color="green", marker="o", s=80, zorder=30
        )
        ax.scatter(goal[0], goal[1], color="red", marker="X", s=80, zorder=30)

        # Mark all GOAL_POINTS for reference
        for gi, gp in enumerate(GOAL_POINTS):
            ax.scatter(
                gp[0],
                gp[1],
                color="orange",
                marker="*",
                s=40,
                zorder=25,
                alpha=0.5,
            )

        m = results["agg"]
        ax.set_title(
            f"{test['name']}\n"
            f"goal_err={m['goal_xy_err']['mean']:.1f}±{m['goal_xy_err']['std']:.1f}  "
            f"head={m['heading_err_deg']['mean']:.0f}°",
            fontsize=8,
        )
        ax.set_xlim(-20, 45)
        ax.set_ylim(-20, 50)
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"eval_{suite_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {fig_path}")


def print_summary_table(suite_label, test_results):
    """Print a nicely formatted results table."""
    print(f"\n{'='*80}")
    print(f"  {suite_label}")
    print(f"{'='*80}")
    header = f"{'Test':<16} {'GoalXY':>10} {'StartXY':>10} {'Head°':>8} {'Hitch°':>8} {'PathLen':>10} {'Smooth':>8}"
    print(header)
    print("-" * len(header))

    for test, results in test_results:
        m = results["agg"]
        print(
            f"{test['name']:<16} "
            f"{m['goal_xy_err']['mean']:>6.2f}±{m['goal_xy_err']['std']:<4.1f}"
            f"{m['start_xy_err']['mean']:>6.2f}±{m['start_xy_err']['std']:<4.1f}"
            f"{m['heading_err_deg']['mean']:>5.1f}±{m['heading_err_deg']['std']:<3.0f}"
            f"{m['hitch_err_deg']['mean']:>5.1f}±{m['hitch_err_deg']['std']:<3.0f}"
            f"{m['path_length']['mean']:>7.1f}±{m['path_length']['std']:<4.0f}"
            f"{m['smoothness']['mean']:>5.2f}±{m['smoothness']['std']:<3.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical diffusion navigation models"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Samples per test case"
    )
    parser.add_argument("--hl_epoch", default="latest")
    parser.add_argument("--ll_epoch", default="latest")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", default="results/navigation_eval")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["in_dist", "all_pairs", "ood"],
        choices=list(SUITES.keys()),
        help="Which test suites to run",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load models ---
    print("Loading dataset for normalizer...")
    fallback_dataset = H5GoalDataset(
        h5_path=H5_PATH,
        final_goal=FINAL_GOAL,
        horizon=HL_HORIZON,
        normalizer="LimitsNormalizer",
        max_path_length=310,
        max_n_episodes=2000,
        termination_penalty=None,
        use_padding=False,
        jump=HL_JUMP,
        jump_action=HL_JUMP_ACTION,
    )
    fallback_renderer = NavigationRenderer()

    print("Loading HL model...")
    hl_ema, hl_dataset = load_model(
        HL_LOG_DIR, args.hl_epoch, fallback_dataset, fallback_renderer
    )
    print("Loading LL model...")
    ll_ema, _ = load_model(
        LL_LOG_DIR, args.ll_epoch, fallback_dataset, fallback_renderer
    )

    hl_policy = Policy(hl_ema, hl_dataset.normalizer)
    ll_policy = Policy(ll_ema, hl_dataset.normalizer)

    def _goal_index(state):
        """Find which GOAL_POINTS index a state matches (by xy), or None."""
        for i, gp in enumerate(GOAL_POINTS):
            if abs(state[0] - gp[0]) < 0.5 and abs(state[1] - gp[1]) < 0.5:
                return i
        return None

    # --- Load reference episodes from training data ---
    print("Loading reference trajectories from H5 data...")
    episodes_by_pair = load_reference_episodes(H5_PATH)
    print(
        f"  Loaded {sum(len(v) for v in episodes_by_pair.values())} episodes across {len(episodes_by_pair)} goal pairs"
    )

    # --- Run suites ---
    all_results = {}

    for suite_key in args.suites:
        suite_label, build_fn = SUITES[suite_key]
        tests = build_fn()
        print(f"\n{'#'*60}")
        print(
            f"  Suite: {suite_label}  ({len(tests)} tests × {args.n_samples} samples)"
        )
        print(f"{'#'*60}")

        test_results = []
        for test in tests:
            start_obs = state4d_to_obs6d(test["start"])
            goal_obs = state4d_to_obs6d(test["goal"])

            samples = []
            metrics_list = []
            for s in range(args.n_samples):
                with torch.no_grad():
                    traj_raw, hl_wp_raw = plan_hierarchical(
                        hl_policy, ll_policy, start_obs, goal_obs
                    )
                traj_4d = np.array([obs6d_to_state4d(o) for o in traj_raw])
                hl_wp_4d = np.array([obs6d_to_state4d(o) for o in hl_wp_raw])
                samples.append((traj_4d, hl_wp_4d))
                metrics_list.append(
                    compute_metrics(traj_4d, test["start"], test["goal"])
                )

            agg = aggregate_metrics(metrics_list)
            test_results.append(
                (
                    test,
                    {"samples": samples, "metrics": metrics_list, "agg": agg},
                )
            )

            print(
                f"  {test['name']:<16} "
                f"goal_err={agg['goal_xy_err']['mean']:.2f}  "
                f"start_err={agg['start_xy_err']['mean']:.2f}  "
                f"heading={agg['heading_err_deg']['mean']:.1f}°"
            )

        # Build reference trajectories for this suite
        ref_trajectories = None
        if suite_key in ("in_dist", "all_pairs"):
            ref_trajectories = {}
            for test in tests:
                si = _goal_index(test["start"])
                gi = _goal_index(test["goal"])
                if si is not None and gi is not None:
                    ref = get_reference_trajectory(episodes_by_pair, si, gi)
                    if ref is not None:
                        ref_trajectories[test["name"]] = ref

        print_summary_table(suite_label, test_results)
        plot_suite_results(
            suite_key, test_results, args.output_dir, ref_trajectories
        )
        all_results[suite_key] = {
            test["name"]: results["agg"] for test, results in test_results
        }

    # --- Save JSON summary ---
    json_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary → {json_path}")


if __name__ == "__main__":
    main()
