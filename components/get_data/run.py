#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os

import wandb

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# def go(args):

    # run = wandb.init(job_type="download_file")
    # run.config.update(args)

    # logger.info(f"Returning sample {args.sample}")
    # logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    # log_artifact(
        # args.artifact_name,
        # args.artifact_type,
        # args.artifact_description,
        # os.path.join("data", args.sample),
        # run,
    # )
# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Download URL to a local destination")

    # parser.add_argument("sample", type=str, help="Name of the sample to download")

    # parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    # parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    # parser.add_argument(
        # "artifact_description", type=str, help="A brief description of this artifact"
    # )

    # args = parser.parse_args()

    # go(args)
    
    

## Suggestion-----------------------------------------------


def go(args):
    run = wandb.init(job_type="download_file")
    run.config.update(vars(args))

    # MLflow übergibt nur den Namen, hier absolute Pfad erzeugen
    sample_path = os.path.abspath(args.sample)

    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"File not found: {sample_path}")
    ##-------------------------------
    # ✅ MLflow Artifact loggen
    # with mlflow.start_run():
        # artifact_folder = "raw_data"  # Unterordner im MLflow Artifact Store
        # mlflow.log_artifact(sample_path, artifact_folder)
        # print(f"✅ sample.csv als Artifact geloggt: {artifact_folder}/{os.path.basename(sample_path)}")
    
    ##-------------------------------
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases from {sample_path}")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        sample_path,
        run
    )
    run.finish()
    logger.info(f"✅ Upload abgeschlossen: {args.artifact_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a local file to W&B")

    parser.add_argument("sample", type=str, help="Pfad zur Datei, die hochgeladen werden soll")
    parser.add_argument("artifact_name", type=str, help="Name des W&B Artifacts")
    parser.add_argument("artifact_type", type=str, help="Typ des Artifacts")
    
    # Nimmt alle verbleibenden Argumente als Beschreibung
    parser.add_argument(
        "artifact_description",
        type=str,
        nargs='+',  # <- Alle Argumente zusammenfassen
        help="Beschreibung des Artifacts"
    )

    args = parser.parse_args()

    # Zusammenfügen aller Worte zu einem String
    args.artifact_description = " ".join(args.artifact_description)

    go(args)

