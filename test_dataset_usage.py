# test_dataset_usage.py
import sys
sys.path.insert(0, 'src')

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
import ho_optim_drl.utils as ut
from ho_optim_drl.gym_env import HandoverEnvPPO

config = Config()
data_dir = 'data/processed'

rsrp_files = dl.get_filenames(data_dir, 'rsrp')
sinr_files = dl.get_filenames(data_dir, 'sinr')

use_speed_list = [30, 50]
rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(
    rsrp_files, sinr_files, use_speed_list
)

# Load dataset đầu tiên
rsrp_db, sinr_db = dl.load_preprocess_dataset(
    config, data_dir, rsrp_files[0], sinr_files[0]
)

# Tạo environment
env = HandoverEnvPPO(config, [rsrp_db], [sinr_db], [sinr_db])

# Chạy 10000 steps và theo dõi dataset_idx
for i in range(10000):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    if done or truncated:
        print(f"Episode ended at step {i}, dataset_idx = {env.dataset_idx}")
        env.reset()

print(f"Cuoi cung: dataset_idx = {env.dataset_idx}")
print("Neu dataset_idx van = 0 sau khi reset nhieu lan, day la bug!")
