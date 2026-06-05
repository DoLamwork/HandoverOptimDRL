"""
Script to check for coverage holes (outages) in the SINR dataset file.
Finds all timesteps where even the best base station has SINR below the Qout threshold (-8.0 dB).
"""

import os
import numpy as np
import scipy.io

def check_sinr_outages(file_path: str, q_out_db: float = -8.0):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load dataset
    print(f"Loading dataset: {file_path}...")
    data = scipy.io.loadmat(file_path)
    
    # Identify variables
    var_name = 'sinr' if 'sinr' in data else [k for k in data.keys() if not k.startswith('__')][0]
    sinr_matrix = data[var_name]
    
    # Dimensions: can be (n_bs, time_steps) or (time_steps, n_bs)
    if sinr_matrix.shape[0] == 5:
        # (n_bs, time_steps)
        n_bs, time_steps = sinr_matrix.shape
        max_sinr_per_step = np.max(sinr_matrix, axis=0)
    else:
        # (time_steps, n_bs)
        time_steps, n_bs = sinr_matrix.shape
        max_sinr_per_step = np.max(sinr_matrix, axis=1)

    print(f"Dataset Dimensions: {time_steps} timesteps, {n_bs} base stations.")
    print(f"Checking for outages where Max SINR < {q_out_db} dB...")

    # Find all timesteps where max SINR is below threshold
    outage_mask = max_sinr_per_step < q_out_db
    outage_indices = np.where(outage_mask)[0]

    if len(outage_indices) == 0:
        print("[INFO] No coverage holes found in this dataset. Every step has at least one base station with SINR >= -8.0 dB.")
        return

    # Group contiguous blocks of outages
    outage_blocks = []
    if len(outage_indices) > 0:
        start_idx = outage_indices[0]
        prev_idx = outage_indices[0]
        
        for idx in outage_indices[1:]:
            if idx == prev_idx + 1:
                prev_idx = idx
            else:
                outage_blocks.append((start_idx, prev_idx))
                start_idx = idx
                prev_idx = idx
        outage_blocks.append((start_idx, prev_idx))

    print(f"\nFound {len(outage_blocks)} coverage hole (outage) block(s):")
    print("-" * 75)
    print(f"{'Block':<6} | {'Start':<8} | {'End':<8} | {'Duration (steps)':<18} | {'Max SINR in Block':<18}")
    print("-" * 75)

    n310_threshold = 10
    t310_threshold = 100
    guaranteed_rlf_threshold = n310_threshold + t310_threshold  # 110 steps

    guaranteed_rlf_blocks_count = 0

    for i, (start, end) in enumerate(outage_blocks):
        duration = end - start + 1
        block_sinrs = max_sinr_per_step[start:end+1]
        best_sinr = np.max(block_sinrs)
        
        # Check if this outage duration triggers guaranteed RLF (110 steps)
        status = ""
        if duration >= guaranteed_rlf_threshold:
            status = "[GUARANTEED RLF]"
            guaranteed_rlf_blocks_count += 1
        elif duration >= n310_threshold:
            status = "[T310 Timer Started]"

        print(f"#{i+1:<5} | {start:<8} | {end:<8} | {duration:<18} | {best_sinr:<18.2f} dB {status}")

    print("-" * 75)
    print(f"Summary:")
    print(f"  • Total timesteps with NO coverage: {len(outage_indices)} / {time_steps} ({len(outage_indices)/time_steps*100:.2f}%)")
    print(f"  • Number of guaranteed RLF blocks (duration >= 110 steps): {guaranteed_rlf_blocks_count}")
    
    if guaranteed_rlf_blocks_count > 0:
        print("\nConclusion:")
        print(f"  This script confirms that the agent physically CANNOT survive this entire dataset.")
        print(f"  Due to {guaranteed_rlf_blocks_count} severe coverage hole(s), any agent will experience RLF and terminate early.")
        print("  To train effectively across all files, you must cycle datasets on every reset!")

if __name__ == "__main__":
    file_path = os.path.join('data', 'processed', 'sinr_30kmh_0.mat')
    check_sinr_outages(file_path)
