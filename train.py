import subprocess
import os

def train_model():
   
    print("Starting YOLOv9 training...")

    yolov9_path = os.path.join("src", "yolov9")
    train_script_path = os.path.join(yolov9_path, "train_dual.py")
    cfg_path = os.path.join(yolov9_path, "models", "detect", "custom-yolov9-s.yaml")
    data_path = os.path.join(yolov9_path, "data", "custom.yaml")
    hyp_path = os.path.join(yolov9_path, "data", "hyps", "hyp.scratch-high.yaml")
    weights_path = os.path.join(yolov9_path, "yolov9-s.pt")

    command = [
        "python", train_script_path,
        "--batch", "8",
        "--cfg", cfg_path,
        "--epochs", "100",
        "--data", data_path,
        "--hyp", hyp_path,
        "--weights", weights_path,
        "--device", "0"
    ]

    try:
        subprocess.run(command, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find the training script at {train_script_path}.")
        print("Please ensure you have cloned the yolov9 repository into the 'src' directory.")

if __name__ == "__main__":
    train_model()