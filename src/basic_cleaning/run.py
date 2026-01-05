#!/usr/bin/env python
import argparse
import logging
import wandb
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

##-----------------original-----------------------------------------------------------
# def download_data():
    # logger.info("Executing download step...")
    # Hier echte Download-Logik einfügen
    # z.B. wandb artifact download
    # artifact_path = wandb.run.use_artifact("dataset:latest").file()


# def clean_data():
    # logger.info("Executing cleaning step...")
    # Hier deine Cleaning-Logik einfügen


# def go(steps):
    # run = wandb.init(project="eda_project", job_type="basic_cleaning")

    # Parameter festlegen oder optional erweitern
    # run.config.update({
        # "parameter1": 1,
        # "parameter2": 2,
        # "parameter3": "test"
    # })

    # logger.info(f"Running steps: {steps}")

    # if "download" in steps.lower():
        # download_data()

    # if "clean" in steps.lower() or "all" in steps.lower():
        # clean_data()

    # wandb.finish()


# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Basic cleaning pipeline")
    # parser.add_argument("--steps", type=str, default="download",
                        # help="Which step to run: 'download', 'clean', or 'all'")
    # args = parser.parse_args()
    # go(args.steps)

##-----------------original end-----------------------------------------------------------

def download_data(input_artifact: str) -> str:
    """
    Downloads a W&B artifact and returns the path to the CSV file.
    """
    logger.info(f"Downloading artifact {input_artifact} from W&B...")
    artifact = wandb.use_artifact(input_artifact)
    artifact_dir = artifact.download()

    # Find the CSV inside the artifact directory
    csv_files = [f for f in os.listdir(artifact_dir) if f.endswith(".csv")]
    if len(csv_files) != 1:
        raise ValueError("Expected exactly one CSV file in the artifact")

    csv_path = os.path.join(artifact_dir, csv_files[0])
    logger.info(f"Artifact CSV located at {csv_path}")

    return csv_path


def clean_data(input_path: str, min_price: float, max_price: float) -> pd.DataFrame:
    """
    Cleans the data: filters outliers based on price.
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info(f"Filtering rows with price between {min_price} and {max_price}...")
    df_clean = df[(df["price"] >= min_price) & (df["price"] <= max_price)].copy()

    # Optional: reset index
    df_clean.reset_index(drop=True, inplace=True)

    # Save cleaned CSV
    cleaned_csv = "clean_sample.csv"
    df_clean.to_csv(cleaned_csv, index=False)
    logger.info(f"Cleaned data saved to {cleaned_csv}")
    return cleaned_csv


def upload_cleaned_data(cleaned_csv: str, output_artifact: str, output_type: str, output_description: str):
    """
    Logs the cleaned CSV as a W&B artifact.
    """
    logger.info(f"Uploading cleaned data as artifact {output_artifact}...")
    artifact = wandb.Artifact(
        name=output_artifact,
        type=output_type,
        description=output_description
    )
    artifact.add_file(cleaned_csv)
    wandb.log_artifact(artifact)
    logger.info("Upload complete.")


def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")

    # Download raw data
    artifact_path = download_data(args.input_artifact)

    # Clean data
    cleaned_csv = clean_data(
        input_path=artifact_path,
        min_price=args.min_price,
        max_price=args.max_price
    )

    # Upload cleaned artifact
    upload_cleaned_data(
        cleaned_csv=cleaned_csv,
        output_artifact=args.output_artifact,
        output_type=args.output_type,
        output_description=args.output_description
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning step")

    # W&B artifact parameters
    parser.add_argument("--input_artifact", type=str, required=True, help="Name of the input artifact")
    parser.add_argument("--output_artifact", type=str, required=True, help="Name of the output artifact")
    parser.add_argument("--output_type", type=str, required=True, help="Type of the artifact (e.g., dataset)")
    parser.add_argument("--output_description", type=str, required=True, help="Description of the artifact")

    # Cleaning parameters
    parser.add_argument("--min_price", type=float, required=True, help="Minimum allowed price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum allowed price")

    # W&B project
    parser.add_argument("--project_name", type=str, default="nyc_airbnb", help="W&B project name")

    args = parser.parse_args()
    go(args)