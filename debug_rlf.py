import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

# Load data
config = Config()
data_dir = "data/processed"  # Hoặc "data/new_data"
rsrp_files = dl.get_filenames(data_dir, "rsrp")
sinr_files = dl.get_filenames(data_dir, "sinr")

use_speed_list = [30, 50]
rsrp_files, sinr_files, _ = ut.filenames_speed_filter(rsrp_files, sinr_files, use_speed_list)

print(f"Found {len(rsrp_files)} files")

# Load chỉ dataset đầu tiên
rsrp_db, sinr_db = dl.load_preprocess_dataset(config, data_dir, rsrp_files[0], sinr_files[0])

if config.clip_sinr:
    sinr_norm = ut.clipnorm(sinr_db, config.sinr_lower_clip, config.sinr_upper_clip)
else:
    sinr_norm = sinr_db

print(f"\n=== DATASET 0 STATISTICS ===")
print(f"Shape: {sinr_db.shape} (timesteps x BS)")
print(f"Min SINR: {sinr_db.min():.2f} dB")
print(f"Max SINR: {sinr_db.max():.2f} dB")
print(f"Mean SINR (all): {sinr_db.mean():.2f} dB")
print(f"Mean Max SINR (best BS mỗi step): {np.max(sinr_db, axis=1).mean():.2f} dB")

# Check Q_out threshold
q_out = -8.0
steps_below_qout = np.sum(np.max(sinr_db, axis=1) < q_out)
print(f"\nSteps where BEST BS < {q_out} dB: {steps_below_qout} ({100*steps_below_qout/sinr_db.shape[0]:.1f}%)")

# Check nếu < 30% steps có SINR tốt, dataset có thể quá khó
good_threshold = -3.0
good_steps = np.sum(np.max(sinr_db, axis=1) > good_threshold)
print(f"Steps where BEST BS > {good_threshold} dB: {good_steps} ({100*good_steps/sinr_db.shape[0]:.1f}%)")

print("\n" + "="*40)
