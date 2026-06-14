"""Custom WandB callback for logging PPO training metrics."""

from dataclasses import asdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class WandbTrainingCallback(BaseCallback):
    """
    Custom callback for logging detailed PPO training metrics to Weights & Biases.

    Logs per-rollout and per-episode metrics including:
    - Episode rewards and lengths
    - PPO losses (policy, value, entropy)
    - Learning rate and clip fraction
    - Domain-specific metrics (RLF, ping-pong, handover counts)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_count = 0

    def _on_training_start(self) -> None:
        """Log model architecture info at training start."""
        wandb.config.update(
            {
                "policy_class": str(type(self.model.policy).__name__),
                "device": str(self.model.device),
                "n_envs": self.model.n_envs,
            },
            allow_val_change=True,
        )

    def _on_step(self) -> bool:
        """Called at every environment step during rollout collection."""
        # Check for completed episodes via info dicts
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)
                self._episode_count += 1

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout — log aggregated metrics."""
        step = self.num_timesteps

        # --- Episode metrics (from completed episodes in this rollout) ---
        if self._episode_rewards:
            rewards = np.array(self._episode_rewards)
            lengths = np.array(self._episode_lengths)
            wandb.log(
                {
                    "episodes/reward_mean": np.mean(rewards),
                    "episodes/reward_median": np.median(rewards),
                    "episodes/reward_std": np.std(rewards),
                    "episodes/reward_min": np.min(rewards),
                    "episodes/reward_max": np.max(rewards),
                    "episodes/length_mean": np.mean(lengths),
                    "episodes/length_min": np.min(lengths),
                    "episodes/length_max": np.max(lengths),
                    "episodes/count": self._episode_count,
                    "episodes/completed_this_rollout": len(self._episode_rewards),
                    "global_step": step,
                },
                step=step,
            )
            # Clear for next rollout
            self._episode_rewards = []
            self._episode_lengths = []

        # --- PPO training metrics (from SB3 logger) ---
        # These are populated after each model.train() call
        if hasattr(self.model, "logger") and self.model.logger is not None:
            sb3_logger = self.model.logger
            name_to_value = getattr(sb3_logger, "name_to_value", {})

            ppo_metrics = {}
            key_mapping = {
                "train/policy_gradient_loss": "train/policy_loss",
                "train/value_loss": "train/value_loss",
                "train/entropy_loss": "train/entropy_loss",
                "train/approx_kl": "train/approx_kl",
                "train/clip_fraction": "train/clip_fraction",
                "train/clip_range": "train/clip_range",
                "train/explained_variance": "train/explained_variance",
                "train/learning_rate": "train/learning_rate",
                "train/loss": "train/total_loss",
                "train/n_updates": "train/n_updates",
                "rollout/ep_rew_mean": "rollout/ep_rew_mean",
                "rollout/ep_len_mean": "rollout/ep_len_mean",
            }

            for sb3_key, wandb_key in key_mapping.items():
                if sb3_key in name_to_value:
                    ppo_metrics[wandb_key] = name_to_value[sb3_key]

            if ppo_metrics:
                ppo_metrics["global_step"] = step
                wandb.log(ppo_metrics, step=step)

        # --- Environment-specific metrics ---
        # Access the training environment to get HO protocol stats
        env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else None
        if env is not None:
            # Unwrap to get the actual HandoverEnvPPO
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env

            if hasattr(unwrapped, "ho_procedure"):
                ho = unwrapped.ho_procedure
                env_metrics = {
                    "env/current_dataset_idx": unwrapped.dataset_idx,
                    "env/current_timestep": unwrapped.t,
                    "env/continuation_reset_count": getattr(
                        unwrapped, "continuation_reset_count", 0
                    ),
                    "env/survival_rate_pct_mean": np.mean(unwrapped.episode_survival_rates) if getattr(unwrapped, "episode_survival_rates", None) else 0.0,
                }

                # Count events from cumulative counters + active counters of current ongoing episode
                if hasattr(ho, "cntr"):
                    env_metrics.update(
                        {
                            "env/ho_prep_total": getattr(unwrapped, "lifetime_ho_prep_count", 0) + len(ho.cntr["ho_prep"].start_idxs),
                            "env/ho_exec_total": getattr(unwrapped, "lifetime_ho_exec_count", 0) + len(ho.cntr["ho_exec"].start_idxs),
                            "env/ho_completed_total": getattr(unwrapped, "lifetime_ho_completed_count", 0) + len(ho.cntr["ho_exec"].done_idxs),
                            "env/rlf_total": getattr(unwrapped, "lifetime_rlf_count", 0) + len(ho.cntr["rlfr"].start_idxs),
                            "env/pp_total": getattr(unwrapped, "lifetime_pp_count", 0) + len(ho.cntr["mtsc"].aborted_idxs),
                        }
                    )

                env_metrics["global_step"] = step
                wandb.log(env_metrics, step=step)

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        wandb.log(
            {
                "final/total_timesteps": self.num_timesteps,
                "final/total_episodes": self._episode_count,
            }
        )


def init_wandb(config, run_name: str, project_name: str = "handover-ppo"):
    """
    Initialize a WandB run for training.

    Parameters
    ----------
    config : Config
        Training configuration.
    run_name : str
        Name for the WandB run.
    project_name : str
        WandB project name.

    Returns
    -------
    wandb.Run
        The initialized WandB run.
    """
    # Convert config dataclass to dict for wandb
    config_dict = {}
    for key, value in vars(config).items():
        if not key.startswith("_"):
            try:
                config_dict[key] = value
            except Exception:
                config_dict[key] = str(value)

    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        tags=["ppo", "handover"],
        save_code=True,
    )

    return run
