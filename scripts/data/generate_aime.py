"""
Script to generate prompt-based math QA datasets from AIME2025 and save them as .parquet files.

Usage:
    python script_name.py --local_dir /path/to/save

Arguments:
    --local_dir   The base directory where output files will be saved.

Output:
    - Saves one Parquet file for each `num_tokens` setting under the given output directory.
    - Subdirectories:
        - For positive `num_tokens`:   local_dir/data_<num_tokens>/aime2025.parquet
        - For negative `num_tokens`:   local_dir/data9_<num_tokens>/aime2025.parquet
        - For num_tokens = -1:         local_dir/aime2025.parquet (baseline version)
        
Description:
    For each test sample in the combined AIME2025-I and AIME2025-II datasets, the script:
    - Adds a reasoning instruction prompt based on `num_tokens`
    - Stores metadata, prompt, and answer into a dictionary
    - Exports the results to a .parquet file
"""

import pandas as pd
import os
import argparse
from datasets import load_dataset, concatenate_datasets
import random


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--local_dir', type=str, required=True, help='Output directory to save files')
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.local_dir, exist_ok=True)

ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test']
ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II")['test']
# Concatenate the two datasets
ds = concatenate_datasets([ds1, ds2])


for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for i in range(len(ds)):
        question = ds[i]['question'].strip()
        if num_tokens >=-1:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")
        else:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + (f" Think for maximum {abs(num_tokens)} tokens.")


        all_data.append({
                    "data_source": "aime2025",
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ds[i]['answer'],
                        "num_tokens": num_tokens
                    },
                    "extra_info": {
                        'split': 'test',
                        'index': i
                    }
                })
    if num_tokens == -1:
        output_path = os.path.join(args.local_dir, 'aime2025.parquet')
    else:
        if num_tokens < 0:
            output_dir = os.path.join(args.local_dir, f'data9_{num_tokens}')
        else:
            output_dir = os.path.join(args.local_dir, f'data_{num_tokens}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'aime2025.parquet')
    
    # Save the data
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path)
    print(f"Saved data to {output_path}")
    