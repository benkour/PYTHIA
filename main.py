import json



with open('configs/pipeline_components.json', 'r') as f:
        parameters = json.load(f)


if parameters.get("create_data", False):
    from data_creation_pipeline import create_data

if parameters.get("train_model", False):
    from model_training_pipeline import run
    run.main()
if parameters.get("run_bo", False):
    from bo_pipeline import run_bo
    run_bo.main()