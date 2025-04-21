"""
Script to generate prompt-based QA datasets from GPQA and save them as .parquet files.

Usage:
    python script_name.py --output_dir /path/to/save

Arguments:
    --output_dir   The base directory where output files will be saved.

Output:
    - Saves one Parquet file for each `num_tokens` setting under the given output directory.
    - Subdirectories:
        - For positive `num_tokens`:   output_dir/data_<num_tokens>/gpqa.parquet
        - For negative `num_tokens`:   output_dir/data9_<num_tokens>/gpqa.parquet
"""

import pandas as pd
import os
import argparse
import random
from datasets import load_dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save files')
args = parser.parse_args()

# Ensure base output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load dataset
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

# Process each num_tokens variant
for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for idx in range(len(ds['train'])):
        correct_answer = ds['train'][idx]['Correct Answer'].strip()
        incorrect_answers = [
            ds['train'][idx]['Incorrect Answer 1'].strip(),
            ds['train'][idx]['Incorrect Answer 2'].strip(),
            ds['train'][idx]['Incorrect Answer 3'].strip()
        ]

        # Shuffle choices
        shuffled_choices = incorrect_answers + [correct_answer]
        random.shuffle(shuffled_choices)

        # Construct question with options
        question = ds['train'][idx]['Question'] + "\n\nOptions:\n"
        for j, choice in enumerate(shuffled_choices):
            question += f"{chr(65 + j)}. {choice}\n"

        correct_choice = chr(65 + shuffled_choices.index(correct_answer))

        if num_tokens < -1:
            question += "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + \
                        f" Think for maximum {abs(num_tokens)} tokens."
        else:
            question += "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + \
                        (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")

        all_data.append({
            "data_source": "gpqa",
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

    # Set output path
    if num_tokens == -1:
        output_path = os.path.join(args.output_dir, 'gpqa.parquet')
    elif num_tokens < -1:
        output_dir = os.path.join(args.output_dir, f'data9_{num_tokens}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'gpqa.parquet')
    else:
        output_dir = os.path.join(args.output_dir, f'data_{num_tokens}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'gpqa.parquet')

    # Save DataFrame
    pd.DataFrame(all_data).to_parquet(output_path)