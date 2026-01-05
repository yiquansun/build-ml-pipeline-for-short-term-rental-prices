import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
import os

def go(args):
    # Initialize W&B run
    run = wandb.init(project="nyc_airbnb", job_type="train_val_test_split")
    
    # Read local CSV file directly (skip run.use_artifact)
    df = pd.read_csv(args.input_artifact)
    
    # Split dataset
    if args.stratify_by.lower() != "none":
        stratify_col = df[args.stratify_by]
    else:
        stratify_col = None

    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col
    )

    # Save locally
    os.makedirs("artifacts", exist_ok=True)
    trainval_path = os.path.join("artifacts", args.trainval_artifact)
    test_path = os.path.join("artifacts", args.test_artifact)
    trainval.to_csv(trainval_path, index=False)
    test.to_csv(test_path, index=False)

    # Log to W&B
    trainval_artifact = wandb.Artifact(args.trainval_artifact, type="dataset")
    trainval_artifact.add_file(trainval_path)
    run.log_artifact(trainval_artifact)

    test_artifact = wandb.Artifact(args.test_artifact, type="dataset")
    test_artifact.add_file(test_path)
    run.log_artifact(test_artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--trainval_artifact", type=str, default="trainval_data.csv")
    parser.add_argument("--test_artifact", type=str, default="test_data.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stratify_by", type=str, default="none")
    args = parser.parse_args()
    go(args)
