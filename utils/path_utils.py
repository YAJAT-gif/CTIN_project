import os

def get_project_root():
    # This assumes this file is inside `ctin_project/utils/`
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_dataset_path(filename):
    return os.path.join(get_project_root(), "ctin_project", "Datasets", filename)

def get_model_path(filename):
    return os.path.join(get_project_root(), "ctin_project", "Inference", "Datasets", filename)
