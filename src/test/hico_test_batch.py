import os
import time
import json
from glob import glob
import subprocess
import argparse
from lightning import seed_everything

def main(gpu_index=0, random_seed=42, text_file=None, mask_dir=None, mask_range=None, output_dir=None):
    """
    Main function to perform style mixing on masked images.
    Args:
        gpu_index (int): GPU index to use for CUDA_VISIBLE_DEVICES.
        random_seed (int): Random seed for reproducibility.
        text_file (str): Path to the text description JSON file.
        mask_dir (str): Directory containing mask images.
        mask_range (tuple): Range of mask indices to process.
        output_dir (str): Directory to save the output results.
    """
    # Seed for reproducibility
    seed_everything(random_seed)

    # Start timer
    start_time = time.time()

    # Default paths if not provided
    text_file = text_file or "data/mmcelebahq/text.json"
    mask_dir = mask_dir or "data/mmcelebahq/mask"
    mask_range = mask_range or (27000, 30000)
    output_dir = output_dir or "data/hico"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load text descriptions
    try:
        with open(text_file, mode="r") as f:
            texts = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {text_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {text_file}.")
        return

    # Prepare mask paths
    masks = [
        os.path.join(mask_dir, f"{i}.png")
        for i in range(mask_range[0], mask_range[1])
    ]

    # Iterate through masks
    for mask in masks:
        if not os.path.exists(mask):
            print(f"Warning: Mask file {mask} not found, skipping.")
            continue

        base_name = os.path.basename(mask).split(".")[0]
        text_key = base_name + ".jpg"

        if text_key not in texts:
            print(f"Warning: Description for {text_key} not found, skipping.")
            continue

        text = texts[text_key][0]
        print(f"Processing: {base_name}")
        print(f"Description: {text}")
        save_path = os.path.join(output_dir,text_key)
        # Construct command
        command = [
            "python", "src/inference/hico_infer.py",
            f"--device=cuda:{gpu_index}",
            f"--seed={random_seed}",
            f"--mask={mask}",
            f"--prompt='{text}'",
            f"--output_dir={output_dir}",
        ]
        full_command = " ".join(command)

        # Execute command
        try:
            subprocess.run(full_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {base_name}: {e}")
            continue

    # Calculate elapsed time
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per image: {total_time / len(masks):.4f} seconds")
    print(f"FPS: {len(masks) / total_time:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Mixing on Masked Images")
    parser.add_argument("--gpu_index", type=int, default=7, help="GPU index to use")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--text_file", type=str, help="Path to text description JSON file")
    parser.add_argument("--mask_dir", type=str, help="Directory containing mask images")
    parser.add_argument("--mask_range", type=int, nargs=2, metavar=('START', 'END'), help="Range of mask indices to process")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output results")
    
    args = parser.parse_args()
    
    main(
        gpu_index=args.gpu_index,
        random_seed=args.random_seed,
        text_file=args.text_file,
        mask_dir=args.mask_dir,
        mask_range=tuple(args.mask_range) if args.mask_range else None,
        output_dir=args.output_dir
    )
