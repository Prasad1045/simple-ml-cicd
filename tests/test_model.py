# tests/test_model.py
import subprocess
import os

def test_training_script_runs_and_creates_model():
    # Run the training script 
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    
    # Check if the script ran successfully (return code 0)
    assert result.returncode == 0, f"Training script failed with error:\n{result.stderr}"
    
    # Check if the model artifact file was created
    model_file_path = "model.pkl"
    assert os.path.exists(model_file_path), "Model file (model.pkl) was not created."
    
    # Clean up the created model file
    if os.path.exists(model_file_path):
        os.remove(model_file_path)