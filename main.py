import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
## suggestion-------------
import shutil

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')  # Adding version_base for Python 3.13 compatibility
def go(config: DictConfig):

    ## ---------------new added----------------------------
    components_dir = os.path.join(hydra.utils.get_original_cwd(), "components")
    ## ---------------new added ends-----------------------
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    print(f"Steps parameter: {steps_par}")  # <--- Debug
    active_steps = steps_par.split(",") if steps_par != "all" else _steps
    print(f"Active steps: {active_steps}")  # <--- Debug

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        ## original-----------------------------------------------------------
        # if "download" in active_steps:
        ##    Download file and load in W&B
            # _ = mlflow.run(
                # f"{config['main']['components_repository']}/get_data",
                # "main",
                # env_manager="conda",
                # parameters={
                    # "sample": config["etl"]["sample"],
                    # "artifact_name": "sample.csv",
                    # "artifact_type": "raw_data",
                    # "artifact_description": "Raw file as downloaded"
                # },
            # )
        
        ## suggestion-------------------------------------------------------- 
        if "download" in active_steps:
            print("##################in step download")
            ## 1️⃣ MLflow Download Step: get_data Component
            result = mlflow.run(
                os.path.join(config['main']['components_repository'], "get_data"),
                "main",
                env_manager="conda",
                parameters={
                    #"sample": config["etl"]["sample"],
                    "sample": r"data\sample1.csv",
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw_file_as_downloaded"
                },
            )
            print("################after result")
            run_id = result.run_id
            print(f"MLflow run_id: {run_id}")

            #----------------------------------------------------
            # 2️⃣ Alle Artifacts vom Run auflisten
            artifacts = mlflow.artifacts.list_artifacts(run_id)
            sample_artifact_path = None
            for a in artifacts:
                print(f" - {a.path} (is_dir={a.is_dir})")
                if "sample.csv" in a.path:
                    sample_artifact_path = a.path
                    break

            # if sample_artifact_path is None:
                # raise FileNotFoundError("sample.csv konnte im MLflow-Run nicht gefunden werden!")

            # 3️⃣ Artifact lokal herunterladen
            local_sample_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=sample_artifact_path
            )
            print(f"✅ sample.csv temporär heruntergeladen: {local_sample_path}")
            
            
            #-----------------------------------------------------
            # MLflow Artifact-Ordner holen
            client = mlflow.tracking.MlflowClient()
            artifact_uri = client.get_run(run_id).info.artifact_uri

            local_artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)
            print(f"✅ Artifacts lokal: {local_artifacts_path}")
            
            print("🔹 local_artifacts_path:", local_artifacts_path)
            print("🔹 Inhalt:", os.listdir(local_artifacts_path))

            # W&B Artifact anlegen
            run = wandb.init(
                project=config["main"]["project_name"],
                job_type="download",
                save_code=True
            )

            artifact = wandb.Artifact(
                name="raw_data",
                type="raw_data",
                description="Raw file as downloaded"
            )
            source_csv = r"C:\Users\DFFFNGM\build-ml-pipeline-for-short-term-rental-prices\components\get_data\data\sample1.csv"
            shutil.copy(source_csv, os.path.join(local_artifacts_path, "sample.csv"))
            print("✅ sample.csv wurde im Artifact-Ordner angelegt")
            
            sample_file = os.path.join(local_sample_path, "sample.csv")
            # Prüfen, ob die Datei existiert
            if not os.path.isfile(sample_file):
                raise FileNotFoundError(f"sample.csv nicht gefunden unter {sample_file}")

            print(f"✅ sample.csv gefunden unter: {sample_file}")

            artifact.add_file(sample_file)  # ✅ richtiger Pfad zur Datei
            run.log_artifact(artifact)
            run.finish()
            print(f"✅ sample.csv erfolgreich als W&B Artifact hochgeladen.")
            
            #-------------------------------------------
        
            # 2️⃣ Pfad zum sample.csv Artifact im MLflow Run finden
            artifacts = mlflow.artifacts.list_artifacts(run_id)
            sample_path = None
            for a in artifacts:
                print(f" - {a.path} (is_dir={a.is_dir})")
                if "sample.csv" in a.path:
                    sample_path = a.path

            # if sample_path is None:
                # raise FileNotFoundError("sample.csv konnte im MLflow-Run nicht gefunden werden!")
          
            
        if "basic_cleaning" in active_steps:
            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "dataset",  # changed to 'dataset' from 'clean_sample'
                    "output_description": "Data_with_outliers_and_null_values_removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass
            
        if "train_val_test_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",  # output of basic_cleaning
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "output_train": "train.csv",
                    "output_val": "val.csv",
                    "output_test": "test.csv"
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################
            # You might need to capture the output of the data split step like this:
            # (Note: Use the variable name your code expects)
            trainval_data_local_path = "trainval_data.csv"
            
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"), # Changed 'components' to 'src'
                "main",
                parameters={
                    "trainval_artifact": trainval_data_local_path,  # path to trainval_data.csv
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                }
            )

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                    os.path.join(os.getcwd(), "components", "test_regression_model"), # Changed project_config to config
                    "main",
                    parameters={
                        "mlflow_model": "yiquan_sun-cariad/build-ml-pipeline-for-short-term-rental-prices-src_train_random_forest/rf_tfidf10_mf0.5:prod",
                        "test_dataset": "test_data.csv:v0"
                    },
                )


if __name__ == "__main__":
    go()
