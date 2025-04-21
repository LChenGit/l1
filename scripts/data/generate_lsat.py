"""
Script to generate prompt-based QA datasets from AGIEval LSAT and save them as .parquet files.

Usage:
    python script_name.py --local_dir /path/to/save

Arguments:
    --local_dir   The base directory where output files will be saved.

Output:
    - Saves one Parquet file for each `num_tokens` setting under the given local directory.
    - Subdirectories:
        - For positive `num_tokens`:   local_dir/data_<num_tokens>/lsat.parquet
        - For negative `num_tokens`:   local_dir/data9_<num_tokens>/lsat.parquet
        - For num_tokens = -1:         local_dir/data/lsat.parquet
"""

import pandas as pd
import os
import argparse
from datasets import load_dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--local_dir', type=str, required=True, help='Local directory to save files')
args = parser.parse_args()

# Ensure base output directory exists
os.makedirs(args.local_dir, exist_ok=True)

# Load LSAT dataset
ds_lsat = load_dataset("dmayhem93/agieval-lsat-ar")

# Process and save with different num_tokens
for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []

    for idx in range(len(ds_lsat['test'])):
        question = ds_lsat['test'][idx]['query']
        correct_choice = chr(65 + ds_lsat['test'][idx]['gold'][0])  # gold is a list like [2]

        # Append prompt logic
        if num_tokens < -1:
            question += "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + \
                        f" Think for maximum {abs(num_tokens)} tokens."
        else:
            question += "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + \
                        (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")

        all_data.append({
            "data_source": "lsat",
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
                'index': idx
            }
        })

    # Determine output path
    if num_tokens == -1:
        output_dir = os.path.join(args.local_dir, 'data')
    elif num_tokens < -1:
        output_dir = os.path.join(args.local_dir, f'data9_{num_tokens}')
    else:
        output_dir = os.path.join(args.local_dir, f'data_{num_tokens}')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lsat.parquet')
    pd.DataFrame(all_data).to_parquet(output_path)