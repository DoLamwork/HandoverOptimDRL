"""Train a PPO agent using the provided configuration."""

from datetime import datetime
import importlib.util
import os

from rl_zoo3 import linear_schedule
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import torch
import wandb

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
from ho_optim_drl.wandb_callback import WandbTrainingCallback, init_wandb
import ho_optim_drl.utils as ut

SIM_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SWEEP_NAME = "ppo_sweep"
SAVE_MODEL = True


def get_sweep_config():
    """Get the sweep configuration for WandB."""
    return {
        "name": SWEEP_NAME,
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "reward_sum_avg"},
        "parameters": {
            "ent_coef": {"values": [0.001, 0.01, 0.1]},
            "rew_const": {"values": [0.8, 0.9, 1.0]},
        },
    }


def main(
    root_path: str,
    sweep: bool = False,
    use_wandb: bool = False,
    phase: int = 1,
    load_model: str | None = None,
) -> int:
    """Main function to train or sweep PPO on the handover environment.

    Parameters
    ----------
    root_path : str
        Root path of the project.
    sweep : bool
        Whether to run a WandB sweep.
    use_wandb : bool
        Whether to use WandB logging.
    phase : int
        Curriculum learning phase:
        - Phase 1: Short HO preparation and no termination on ping-pong
        - Phase 2: Standard HO preparation and termination on ping-pong
    load_model : str or None
        Path to a pre-trained model to continue training from (for Phase 2).
    """
    if sweep:
        return sweep_ppo(root_path)
    return train_ppo(root_path, use_wandb=use_wandb, phase=phase, load_model=load_model)


def sweep_ppo(root_path: str) -> int:
    """Run a WandB sweep for hyperparameter optimization."""
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep=sweep_config, project=SWEEP_NAME)
    wandb.agent(sweep_id, lambda: train_ppo(root_path), count=1)

    return 0


def train_ppo(
    root_path: str,
    use_wandb: bool = False,
    phase: int = 1,
    load_model: str | None = None,
):
    """Train a PPO agent on the handover environment.

    Parameters
    ----------
    root_path : str
        Root path of the project.
    use_wandb : bool
        Whether to use WandB logging.
    phase : int
        Curriculum phase (1 = short preparation, 2 = standard preparation).
    load_model : str or None
        Path to pre-trained model (.zip) to load for fine-tuning.
    """
    # Load configuration
    config = Config()
    if use_wandb:
        config.use_wandb = True

    # --- Curriculum Learning ---
    if phase == 1:
        # Phase 1: Episodes terminate only on RLF or max timesteps.
        # Custom experiment: permit_ho_prep_abort=True, t_ho_prep=3.
        # Allows aborting but makes handovers easier by reducing prep time.
        config.terminate_on_rlf = True
        config.terminate_on_pp = False
        config.permit_ho_prep_abort = True
        config.t_ho_prep = 3
        config.cycle_on_reset = False
        config.random_window_reset = True
        print("[Curriculum] Phase 1: abort=True, t_ho_prep=3, terminate_on_pp=False")
        print("[Curriculum] Training uses random trace windows with prioritized failure replay.")
    elif phase == 2:
        # Phase 2: Episodes also terminate on PP events.
        # Restores real 3GPP parameters: permit_ho_prep_abort=True, t_ho_prep=5.
        config.terminate_on_rlf = True
        config.terminate_on_pp = True
        config.permit_ho_prep_abort = True
        config.t_ho_prep = 5
        config.cycle_on_reset = True
        config.random_window_reset = True
        print("[Curriculum] Phase 2: abort=True, t_ho_prep=5, terminate_on_rlf=True, terminate_on_pp=True")
        print("[Curriculum] Training uses random trace windows with prioritized failure replay.")
        if load_model is None:
            print("[WARNING] Phase 2 without --load-model: training from scratch.")
    else:
        raise ValueError(f"Invalid curriculum phase: {phase}. Use 1 or 2.")

    print(
        "[Sampler] "
        f"window={config.episode_window_steps} steps, "
        f"failure_replay={config.failure_sampling_probability:.0%}, "
        f"lookback={config.failure_lookback_min}-{config.failure_lookback_max} steps"
    )

    # Load MATLAB files
    data_dir = os.path.join(root_path, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")

    # Speed filter
    use_speed_list = [30, 50]
    rsrp_files, sinr_files, _ = ut.filenames_speed_filter(
        rsrp_files, sinr_files, use_speed_list
    )

    # Load all datasets
    rsrp_list = []
    sinr_list = []
    sinr_norm_list = []
    for rsrp_fname_i, sinr_fname_i in zip(rsrp_files, sinr_files):
        # Load dataset
        rsrp_db, sinr_db = dl.load_preprocess_dataset(
            config, data_dir, rsrp_fname_i, sinr_fname_i
        )

        # Clip and normalize SINR
        if config.clip_sinr:
            sinr_norm = ut.clipnorm(
                sinr_db, config.sinr_lower_clip, config.sinr_upper_clip
            )
        else:
            sinr_norm = sinr_db

        sinr_list.append(sinr_db)
        rsrp_list.append(rsrp_db)
        sinr_norm_list.append(sinr_norm)

    print(f"[Data] Loaded {len(rsrp_list)} datasets (speeds: {use_speed_list})")
    print(f"[Data] First dataset shape: {rsrp_list[0].shape} (time_steps, n_bs)")

    # Generate environment
    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env, warn=True)

    # WandB initialization
    callbacks = []
    if config.use_wandb:
        # If not already initialized by sweep, init a standalone run
        if wandb.run is None:
            init_wandb(
                config,
                run_name=f"ppo_phase{phase}_{SIM_ID}",
                project_name="handover-ppo",
            )
        else:
            # Sweep mode: update config from wandb sweep
            config.update(wandb.config.as_dict())

        # Log curriculum phase
        if wandb.run is not None:
            wandb.config.update({"curriculum_phase": phase}, allow_val_change=True)

        # Add the WandB training callback
        callbacks.append(WandbTrainingCallback(verbose=1))

    # Directories
    if config.use_wandb and wandb.run is not None:
        run_name = f"{wandb.run.name}_{SIM_ID}"
    else:
        run_name = SIM_ID

    model_dir = os.path.join(
        root_path,
        "results",
        "models",
        SWEEP_NAME,
        run_name,
    )
    if importlib.util.find_spec("tensorboard") is not None:
        tensorboard_log_dir = os.path.join(
            root_path,
            "results",
            "tensorboard",
            SWEEP_NAME,
            run_name,
        )
    else:
        tensorboard_log_dir = None

    # PPO model
    if load_model is not None:
        # Load pre-trained model and set new environment
        model_path = load_model
        if not model_path.endswith(".zip"):
            model_path = model_path + ".zip"
        if not os.path.isabs(model_path):
            model_path = os.path.join(root_path, model_path)

        print(f"[Model] Loading pre-trained model from: {model_path}")
        model = PPO.load(
            model_path,
            env=env,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tensorboard_log=tensorboard_log_dir,
        )
        # Update learning rate for fine-tuning (optionally use smaller LR)
        if phase == 2:
            fine_tune_lr = config.lr * 0.1  # 10x smaller LR for fine-tuning
            model.learning_rate = linear_schedule(fine_tune_lr)
            print(f"[Model] Fine-tuning LR: {fine_tune_lr}")
    else:
        # Create new model from scratch
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=config.net_arch, vf=config.net_arch),
        )
        model = PPO(
            "MlpPolicy",
            env,
            ent_coef=config.ent_coef,
            learning_rate=linear_schedule(config.lr),
            verbose=1,
            policy_kwargs=policy_kwargs,
            n_steps=config.n_steps_per_update,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            tensorboard_log=tensorboard_log_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    print(f"[Training] Starting Phase {phase} training for {config.n_steps_total} steps...")
    model.learn(
        total_timesteps=config.n_steps_total,
        progress_bar=True,
        callback=callbacks if callbacks else None,
    )

    if SAVE_MODEL:
        model.save(model_dir)
        print(f"[Model] Saved to: {model_dir}")

        # Log model as WandB artifact
        if config.use_wandb and wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"ppo-model-phase{phase}-{run_name}",
                type="model",
                description=f"PPO handover model Phase {phase} trained at {SIM_ID}",
            )
            model_artifact.add_file(f"{model_dir}.zip")
            wandb.log_artifact(model_artifact)

    if config.use_wandb:
        wandb.finish()

    return 0
