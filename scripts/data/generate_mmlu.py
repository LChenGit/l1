"""
Script to generate prompt-based QA datasets from MMLU (cais/mmlu, 'all' split) and save them as .parquet files.

Usage:
    python script_name.py --local_dir /path/to/save

Arguments:
    --local_dir   The base directory where output files will be saved.

Output:
    - Saves one Parquet file for each `num_tokens` setting under the given local directory.
    - Only the first 1000 shuffled entries from the test set are saved.
    - Subdirectories:
        - For positive `num_tokens`:   local_dir/data_<num_tokens>/mmlu_1000.parquet
        - For negative `num_tokens`:   local_dir/data9_<num_tokens>/mmlu_1000.parquet
        - For num_tokens = -1:         local_dir/data/mmlu_1000.parquet
"""

import pandas as pd
import numpy as np
import os
import argparse
from datasets import load_dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--local_dir', type=str, required=True, help='Local directory to save files')
args = parser.parse_args()

# Ensure base output directory exists
os.makedirs(args.local_dir, exist_ok=True)

# Load dataset
ds_mmlu = load_dataset("cais/mmlu", "all")

# Process and save with different num_tokens
for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for idx in range(len(ds_mmlu['test'])):
        row = ds_mmlu['test'][idx]
        options = row['choices']
        options_str = ""
        for j in range(len(options)):
            options_str += f"{chr(65 + j)}. {options[j]}\n"

        # Build the full question
        question = row['question'] + "\n\nOptions:\n" + options_str
        correct_choice = chr(65 + row['answer'])

        # Token prompt logic
        if num_tokens < -1:
            question += "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + \
                        f" Think for maximum {abs(num_tokens)} tokens."
        else:
            question += "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + \
                        (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")

        all_data.append({
            "data_source": "mmlu",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": correct_choice,
                "num_tokens": num_tokens
            },
            "extra_info": {
                'split': 'test',
                'index': idx,
                'subject': row['subject']
            }
        })

    # Shuffle and truncate to 1000 examples
    np.random.seed(42)
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    all_data = [all_data[i] for i in indices[:1000]]

    # Determine output path
    if num_tokens == -1:
        output_dir = os.path.join(args.local_dir, 'data')
    elif num_tokens < -1:
        output_dir = os.path.join(args.local_dir, f'data9_{num_tokens}')
    else:
        output_dir = os.path.join(args.local_dir, f'data_{num_tokens}')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mmlu_1000.parquet')
    pd.DataFrame(all_data).to_parquet(output_path)