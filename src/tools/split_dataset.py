import argparse
from glob import glob
import os
from sklearn.model_selection import train_test_split
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/mmcelebahq")
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    all_images = glob(os.path.join(args.dataset_path,"image","*"))
    filenames = [os.path.basename(x).split(".")[0] for x in all_images]
    num_val = int(args.val_ratio * len(filenames))
    train = filenames[:-num_val]
    val = filenames[-num_val:]
    with open(os.path.join(args.dataset_path,"train.txt"),"w") as f:
        for x in train:
            f.write(x+"\n")
    with open(os.path.join(args.dataset_path,"val.txt"),"w") as f:
        for x in val:
            f.write(x+"\n")

if __name__ == "__main__":
    main()