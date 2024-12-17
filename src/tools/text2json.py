import argparse
import os
import json
from rich.progress import track


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str,default="data/mmcelebahq/text")
    parser.add_argument("--out_path", type=str, default="data/mmcelebahq/text.json")

    args = parser.parse_args()
    return args


def main():
    args:argparse.Namespace = get_args()
    filenames = [os.path.join(args.text_path,f"{i}.txt") for i in range(30000)]
    text_json = {}
    for filename in track(filenames):
        with open(filename,mode='r') as f:
            prompts = f.readlines()
            prompts = [p.strip() for p in prompts]
        base_name = os.path.basename(filename).split(".")[0]
        text_json[f"{base_name}.jpg"] = prompts
    with open(args.out_path,mode='w') as f:
        json.dump(text_json,f,indent=4)
    print("Done")   


if __name__ == "__main__":
    main()
