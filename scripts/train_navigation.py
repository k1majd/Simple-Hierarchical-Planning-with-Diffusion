"""
Training script for hierarchical diffusion on the custom navigation dataset.

Usage:
  # Train high-level planner:
  python scripts/train_navigation.py --config config.navigation_hl --dataset navigation

  # Train low-level planner:
  python scripts/train_navigation.py --config config.navigation_ll --dataset navigation
"""

import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

# Ensure CWD is on path so `config.*` modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import diffuser.utils as utils
from diffuser.datasets.h5_sequence import H5GoalDataset
from diffuser.utils.rendering import NavigationRenderer

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

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


class Parser(utils.Parser):
    dataset: str = "navigation"
    config: str = "config.navigation_hl"
    use_wandb: bool = False
    wandb_project: str = "hierarchical-diffusion-navigation"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"
    wandb_tags: str = ""


args = Parser().parse_args("diffusion")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

dataset = H5GoalDataset(
    h5_path=H5_PATH,
    final_goal=FINAL_GOAL,
    horizon=args.horizon,
    normalizer=args.normalizer,
    max_path_length=args.max_path_length,
    max_n_episodes=2000,
    termination_penalty=args.termination_penalty,
    use_padding=args.use_padding,
    jump=args.jump,
    jump_action=args.jump_action,
)

renderer = NavigationRenderer()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim * args.jump
if args.jump_action == "none":
    action_dim = 0

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon // args.jump,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim=args.dim,
    dim_mults=args.dim_mults,
    kernel_size=args.kernel_size,
    device=args.device,
    upsample_k=args.upsample_k,
    downsample_k=args.downsample_k,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon // args.jump,
    condition=args.condition,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, renderer)

# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

train_writer = SummaryWriter(log_dir=args.savepath + "-train")
wandb_run = None
if args.use_wandb:
    if wandb is None:
        raise ImportError(
            "WandB is not installed. Install with `pip install wandb` or run with --use_wandb False."
        )

    run_name = (
        args.wandb_run_name
        if args.wandb_run_name
        else os.path.basename(args.savepath)
    )
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    # Keep only simple JSON-serializable fields in WandB config.
    wb_config = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            wb_config[key] = value

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=run_name,
        mode=args.wandb_mode,
        tags=tags,
        config=wb_config,
        sync_tensorboard=True,
        dir=args.savepath,
    )
    print(f"[ train/navigation ] WandB enabled | run: {wandb_run.name}")

try:
    for i in range(n_epochs):
        print(f"Epoch {i} / {n_epochs} | {args.savepath}")
        trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer)
finally:
    train_writer.close()
    if wandb_run is not None:
        wandb.finish()
