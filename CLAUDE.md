# Project Context
When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

# About this project
Dự án này là dự án sử dụng thuật toán PPO để UE tự quyết định handover tại từng timestep thay vì sử dụng các tiêu chuẩn 3GPP nhưng vẫn đảm bảo data rate, HOF và Pingpong. Phần code ban đầu được dùng để tái hiện lại bài báo @2025_A_Deep_Reinforcement_Learning-Based_Approach_for_Adaptive_Handover_Protocols.pdf. Tôi muốn dùng lại chính phần code này để train và validate mô hình PPO dựa trên data mới

# Key directories
- 'data/new_data': data mới dùng để train và validate model mới
- 'data/processed': data cũ dùng để train và validate model cũ
- 'scripts/train_ppo.py: xây dựng model và train ra model PPO
- 'scripts\validate_3gpp.py': file validate 3GPP
- 'scripts\validate_ppo.py': file validate model PPO sau khi train xong
- 'src\ho_optim_drl\config.py': file config cho PPO và môi trường vật lý
- 'src\ho_optim_drl\dataloader.py': sử dụng file này để load data, sử dụng 1 số bước như biến file raw thành ma trân nghịch đảo, lọc, unsampling.
- 'src\ho_optim_drl\gym_env': cài đặt môi trường cho 3GPP và 
- '2025_A_Deep_Reinforcement_Learning-Based_Approach_for_Adaptive_Handover_Protocols.pdf': Bài báo dùng phần data cũ 'processed'

Đầu vào là folder data gồm 2 loại new_data: data mới (mục tiêu của tôi là train model từ data này) và processed: data cũ được dùng để train ra model theo bài báo ban đầu

# Common command
- python -m run train_ppo: train model PPO
- python -m run validate_ppo: Validate the PPO-based protocol
- python -m run validate_3gpp: Validate 3GPP protocol
- python -m run plot_results: Plot the results

# Problems
Mục tiêu của tôi là sử dụng new_data để train và validate thuật toán PPO (file new_data là bộ data tôi tự gen bằng tool khác nên có thể không giống với @processed): Tuy nhiên mục tiêu vẫn là ứng dụng được thuật toán PPO vào @newdata để UE có thể tự quyết định handover tại mỗi timestep chỉ cần dựa vào RSRP và SINR của tất cả các BS tại timestep đó

Hiện tại, kết quả validate của model cũ ở cả PPO và 3GPP đều rất ổn và PPO có phần nhỉnh hơn 3GPP ở 1 số thông số. Tuy nhiên có 1 số vấn đề tôi đang gặp phải:
- Train lại model với file data cũ thì model không cho ra kết quả tốt => không rõ model ban đầu được train từ data nào và train như nào
- với file data mới, model train ra lại có kết quả validate PPO rất tệ, trong khi kết quả validate 3GPP vẫn ổn. Tôi muốn làm rõ vấn đề này
- Liệu có thể gen ra bộ data lớn hơn để tự train và dựa vào validate PPO để tự validate bộ data mới

## Analysis: Why PPO fails on new_data
1. **Distribution Shift**: PPO overfits to the SINR/RSRP patterns of `processed/`. `new_data` likely has different path loss, shadowing, or mobility profiles.
2. **Observation Gap**: The code only provides SINR to the agent. The paper likely uses both RSRP and SINR. Without RSRP, the agent lacks critical signal strength context.
3. **Reward Sensitivity**: The reward function relies on normalized SINR. If the range of SINR in `new_data` differs from `processed/`, the reward signal shifts.
4. **Narrow Training Range**: The current code only uses [30, 50] km/h, making it fragile to other speeds in `new_data`.

## Detailed Flow Comparison: Paper vs Code

| Step | Paper's Theoretical Flow | Code's Actual Implementation | Difference/Impact |
| :--- | :--- | :--- | :--- |
| **Observation** | $\text{RSRP}_{\text{all}}, \text{SINR}_{\text{all}}, \text{Cell ID}, \text{State}$ | $\text{SINR}_{\text{norm-all}}, \text{Serving Cell}, \text{MTS Flag}$ | **High**: Missing RSRP $\rightarrow$ blind to coverage strength. |
| **Decision** | Agent decides **whether** to trigger HO | Agent selects `target_cell`. If $\neq$ current, HO starts. | **Low**: Mathematically equivalent. |
| **Trigger** | Direct trigger based on policy | Agent $\rightarrow$ `ho_prep` (50 steps) $\rightarrow$ `ho_exec` (40 steps) | **Medium**: 90-step latency requires a predictive policy. |
| **Execution** | RRC state machine | Deterministic 3GPP state machine ($\text{N310} \rightarrow \text{T310} \rightarrow \text{RLF}$) | **Low**: Implementation is accurate to 3GPP. |
| **Reward** | Focus on Spectral Efficiency | Proxy: $\text{SINR}_{\text{norm}} + \text{Best-cell bonus} - \text{Penalties}$ | **Medium**: Optimizing for SINR, not directly for efficiency. |

### Alignment Roadmap
1. **Observation Sync**: Add normalized RSRP to `HandoverEnvPPO` observations.
2. **Temporal Alignment**: Review $t_{ho\_prep}$ and $t_{ho\_exec}$ relative to UE speed.
3. **Reward Refinement**: Integrate actual Spectral Efficiency into the reward signal.

# PPO Implementation Details

## 1. PPO Algorithm Configuration
**Library**: Stable-Baselines3 (PPO)
**Total timesteps**: 5,000,000
**Hyperparameters**:
- `n_steps`: 2000 (rollout length)
- `batch_size`: 200
- `n_epochs`: 10
- `learning_rate`: 5e-5 (linear schedule)
- `ent_coef`: 0.1 (entropy coefficient)
- `net_arch`: [64, 128, 64] (ReLU activations)
- Policy: `MlpPolicy`

**WandB sweep** explores:
- `ent_coef`: [0.001, 0.01, 0.1]
- `rew_const`: [0.8, 0.9, 1.0]

## 2. Environment: `HandoverEnvPPO`

### Observation Space
Shape: `(2 * n_bs + 1,)` normalized to [0, 1]
- One-hot serving cell indicator (n_bs)
- Normalized SINR values for all BSs (clipped [-10, 10] dB → scaled [0, 1])
- Ping-pong flag (MTS counter pending: 0 or 1)

### Action Space
`Discrete(n_bs)` - Select target base station index at each timestep.

### Episode Termination (Training)
- Radio Link Failure (RLF) if `terminate_on_rlf=True`
- Ping-pong event if `terminate_on_pp=True`
- End of dataset (truncation)

### Episode Flow
1. Reset: Connect to BS with max RSRP, initial observation
2. Step:
   - Agent selects action (target BS)
   - Environment passes action to `HOProcedurePPO.step(rsrp, sinr, target_cell)`
   - State machine updates HO procedures (prep, exec, timers, RLF recovery)
   - Reward computed
   - Next observation constructed
   - Episode may terminate on RLF/PP

## 3. Handover Protocol: `HOProcedurePPO`

A **3GPP-compliant state machine** implementing RRC handover procedures.

### Components
- `SyncSignal`: Monitors SINR vs Q_in/Q_out thresholds
- `RadioResourceControl (RRC)`: Manages pcell (serving), ncell (neighbor), tcell (transition)
- `GeneralCounter` timers/counters:
  - `n310` (out-of-sync counter), `n311` (in-sync counter), `t310` (RLF timer)
  - `rlfr` (RLF recovery timer)
  - `mtsc` (Minimum Time Stay - ping-pong guard)
  - `ho_prep` (HO preparation: 50 timesteps)
  - `ho_exec` (HO execution: 40 timesteps)

### State Machine (per timestep)
1. Step all counters (increment if active)
2. Radio link monitoring:
   - SINR(pcell) < Q_out → out-of-sync, start n310, reset n311
   - SINR(pcell) > Q_in → in-sync, reset n310
3. Out-of-sync → n310 → start T310
4. In-sync while T310 → start n311
5. T310 expiry → RLF → start RLF recovery
6. N311 max → cancel T310
7. Ping-pong: MTS timer starts after HO complete; new HO during MTS → PP
8. RLF recovery: disconnect, RLFR wait, cell search (SINR > Q_in), reconnect
9. HO preparation:
   - Agent action ≠ pcell → start/prepare HO timer
   - Target changes during prep → abort (if `permit_ho_prep_abort=True`)
   - Agent action = pcell during prep → abort prep
10. HO state machine:
    - HO prep done (50 steps) → start HO exec (40 steps), save pre-HO SINR
    - If T310 running at prep completion → declare RLF instead
    - HO exec done → check target sync:
      * Target in-sync → complete HO, connect to target
      * Target out-of-sync → RLF recovery
11. Update timelines

**Note**: The PPO learns **when to trigger HO** (by selecting target cell). The HO procedure itself is deterministic 3GPP logic.

## 4. Reward Function (`_get_reward()`)

Dense per-timestep reward:

```python
reward = 0.0
sinr_norm = clipped_and_scaled_SINR[serving_cell]
best_bs = argmax(sinr)

# 1. SINR-based reward (0 to ~1.29)
reward += sinr_norm[serving_cell]

# 2. Best cell bonus
if serving_cell == best_bs:
    reward += config.rew_const  # +0.95

# 3. Penalties
if ping_pong_detected:
    reward -= config.rew_const
if rlf_detected:
    reward -= 2 * config.rew_const  # -1.9
if out_of_sync (n310 or t310 pending):
    reward -= config.rew_const
```

**Reward constants** swept: [0.8, 0.9, 1.0]

**Goal**: Maximize SINR, stay on best cell, avoid HOF and ping-pong.

## 5. Training Process (`train_ppo.py`)

```
1. Load data from data/processed/*.mat
2. Speed filter: use_speed_list = [30, 50] km/h  ⚠️
3. Preprocess each dataset:
   - Load .mat files (RSRP, SINR)
   - Upsample (factor=1)
   - L3 filtering with w=0.1
   - Clip SINR to [-10, 10] dB, normalize to [0,1]
4. Create HandoverEnvPPO with all datasets (multi-dataset env)
5. SB3 environment check
6. WandB init if enabled
7. Create PPO model with config
8. model.learn(total_timesteps=5e6)
9.  save model 
10. WandB finish
```



## 6. Metrics Tracked

### Training (TensorBoard/WandB)
- `ep_rew_mean`: Mean episode reward
- `ep_len_mean`: Mean episode length
- `value_loss`, `policy_gradient_loss`
- `entropy_loss`, `explained_variance`
- `learning_rate`

### Validation (per dataset)
From `HOProcedurePPO.get_statistics()`:
- **Spectral efficiency**: `spectral_eff` (bits/s/Hz) and `max_spectral_eff` (upper bound)
- **Relative rate**: `r_rel = spectral_eff / max_spectral_eff`
- **SINR stats**: mean, median, Q1, Q3, variance (dB)
- **HO counters**: started/completed/aborted/failed for prep and exec
- **RLF rate**: `num_rlf / num_ho_exe_started`
- **Ping-pong rate**: `num_pp / num_ho_exe_started`
- **Connected time**: % of timesteps UE connected

Saved to CSV: `results/metrics/ppo_metrics.csv`

### Timeline Data (collected but not auto-logged)
- `s_action`: actions taken per timestep
- `s_pcell`, `s_tcell`: serving/target cell indices
- `sinr_timeline`, `rsrp_timeline`: SINR/RSRP of connected cell
- `sinr_at_ho_exe_pcell`, `sinr_after_ho_exe_tcell`: pre/post HO SINR

### PPO Policy
Learns a mapping from observations to actions. It **overfits** to the specific patterns in the training data (processed data). New data may have:
- Different SINR distribution (mean, variance, correlation)
- Different path loss models
- Different noise/interference characteristics
- Different mobility patterns (if new_data has different speeds)
- Different number of BSs or spatial configuration

Even though the state machine is the same, the **policy** that selects actions may produce:
- Premature handovers (too many HOs)
- Late handovers (RLFs)
- Oscillations (ping-pong)
- All because the learned value function is biased toward old data's SINR statistics.

### The Reward Function Sensitivity
Reward heavily depends on:
- `sinr_norm[serving_cell]`: if new data has different SINR scale (even after clipping/normalization), the absolute reward magnitudes shift
- `rew_const` bonus/penalty: fixed absolute value, but optimal trade-off might differ
- Episode termination: if PPO triggers many RLFs/PPs early, episodes become very short → sparse learning signal

### Distribution Shift
L3 filtering (w=0.1) smooths data, but if new_data has:
- Different sampling rate → different time correlations
- Different number of antennas/users → different interference patterns
- Different environment (urban vs rural)

The observation statistics (mean, std of SINR across all BSs) change, causing the policy to fail.


### Validation Strategy
1. Compare PPO vs 3GPP on SAME datasets (same speed splits)
2. If PPO still fails on new_data:
   - Check data quality: Are RSRP/SINR realistic? (no NaNs, reasonable ranges)
   - Compare distributions: mean/std of SINR across datasets
   - Run PPO in test mode (no early termination) to collect full trajectories
   - Visualize actions vs ground truth best BS: accuracy metric

## 10. Code References

- Training: `scripts/train_ppo.py:53-149`
- Validation: `scripts/validate_ppo.py:16-156`
- Environment: `src/ho_optim_drl/gym_env/ho_env_ppo.py`
- Protocol: `src/ho_optim_drl/gym_env/ho_protocol_ppo.py`
- Config: `src/ho_optim_drl/config.py`
- Data loader: `src/ho_optim_drl/dataloader.py`

All metrics and reward logic are in these f/iles. The PPO implementation is standard SB3; the custom parts are the environment and state machine.


