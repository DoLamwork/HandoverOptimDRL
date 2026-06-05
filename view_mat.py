"""
Interactive CLI tool to view full .mat file values at each timestep.
Prints data as a clean, formatted table. Supports scrolling, jumping to specific timesteps,
and exporting the entire dataset to a text file.
"""

import os
import sys
import numpy as np
import scipy.io

def print_table_header(var_name):
    print("\n" + "=" * 60)
    print(f" Datatable: {var_name} values (dB/dBm) per timestep")
    print("=" * 60)
    print(f"{'Timestep':<10} | {'BS 0':<8} | {'BS 1':<8} | {'BS 2':<8} | {'BS 3':<8} | {'BS 4':<8} | {'Best BS':<7}")
    print("-" * 60)

def print_row(t, values):
    best_bs = np.argmax(values)
    best_val = values[best_bs]
    # Highlight the best BS value with an asterisk
    row_strs = []
    for i, val in enumerate(values):
        if i == best_bs:
            row_strs.append(f"{val:>7.2f}*")
        else:
            row_strs.append(f"{val:>8.2f}")
    print(f"{t:<10} | {' | '.join(row_strs)} | BS {best_bs}")

def export_all_to_txt(matrix, var_name, output_path):
    print(f"\nExporting all timesteps to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f" Full export of variable '{var_name}' from .mat file\n")
        f.write(" (* indicates the best base station at that timestep)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Timestep':<10} | {'BS 0':<9} | {'BS 1':<9} | {'BS 2':<9} | {'BS 3':<9} | {'BS 4':<9} | Best\n")
        f.write("-" * 70 + "\n")
        
        for t in range(matrix.shape[1]):
            values = matrix[:, t]
            best_bs = np.argmax(values)
            row_strs = []
            for i, val in enumerate(values):
                if i == best_bs:
                    row_strs.append(f"{val:>8.2f}*")
                else:
                    row_strs.append(f"{val:>9.2f}")
            f.write(f"{t:<10} | {' | '.join(row_strs)} | BS {best_bs}\n")
    print(f"🎉 Export completed successfully! File saved at: {os.path.abspath(output_path)}")

def main():
    # Allow passing file path as argument, default to sinr_30kmh_0.mat
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'processed', 'sinr_30kmh_0.mat')
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        # Try to search in default path
        alt_path = os.path.join('data', 'processed', file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print("Please specify a valid path to a .mat file.")
            sys.exit(1)

    print(f"Loading file: {file_path} ...")
    data = scipy.io.loadmat(file_path)
    
    # Extract the correct variable (rsrp or sinr)
    var_name = 'sinr' if 'sinr' in data else ('rsrp' if 'rsrp' in data else [k for k in data.keys() if not k.startswith('__')][0])
    matrix = data[var_name]
    
    # Ensure shape is (5, timesteps)
    if matrix.shape[0] != 5:
        matrix = matrix.T
        
    n_bs, time_steps = matrix.shape
    print(f"Successfully loaded. Total timesteps: {time_steps}, Base stations: {n_bs}")
    
    current_step = 0
    page_size = 20
    
    print_table_header(var_name)
    
    while True:
        # Print a page of data
        end_step = min(current_step + page_size, time_steps)
        for t in range(current_step, end_step):
            print_row(t, matrix[:, t])
        
        current_step = end_step
        if current_step >= time_steps:
            print("\n--- End of file reached ---")
            current_step = 0 # wrap around
            
        print("\nCommands:")
        print("  [Enter]         Show next 20 timesteps")
        print("  j <step>        Jump to specific timestep (e.g. 'j 1920')")
        print("  export <path>   Export all data to a text file (e.g. 'export output.txt')")
        print("  q or exit       Quit the program")
        
        user_input = input("\nEnter command: ").strip().lower()
        
        if user_input in ['q', 'exit']:
            print("Goodbye!")
            break
        elif user_input == '':
            # Continue to next page
            continue
        elif user_input.startswith('j ') or user_input.startswith('jump '):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                target = int(parts[1])
                if 0 <= target < time_steps:
                    current_step = max(0, target - 5) # show 5 steps before the target
                    print_table_header(var_name)
                    print(f"--- Jumped to around timestep {target} ---")
                else:
                    print(f"Error: Step must be between 0 and {time_steps - 1}")
            else:
                print("Error: Invalid jump command format. Use 'j <number>'")
        elif user_input.startswith('export'):
            parts = user_input.split()
            out_file = parts[1] if len(parts) > 1 else 'mat_export.txt'
            export_all_to_txt(matrix, var_name, out_file)
        else:
            print("Unknown command. Please try again.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated. Goodbye!")
