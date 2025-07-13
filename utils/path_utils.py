import os

def get_project_root():
    """Returns the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_dataset_path(filename):
    """Returns the full path to a file in the Datasets folder."""
    return os.path.join(get_project_root(), "ctin_project", "Datasets", filename)

def get_model_path(filename):
    """Returns the full path to a file in the Inference/Datasets or main dir."""
    inference_path = os.path.join(get_project_root(), "ctin_project", "Inference", "Datasets", filename)
    alt_path = os.path.join(get_project_root(), "ctin_project", filename)
    return inference_path if os.path.exists(inference_path) else alt_path
