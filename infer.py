import subprocess
import os
import glob
import json
import cv2
import argparse
from pathlib import Path

def run_inference(input_folder, weights_path):

    print(f"Running inference on images in: {input_folder}")
    
    yolov9_path = os.path.join("src", "yolov9")
    detect_script_path = os.path.join(yolov9_path, "detect_dual.py")
    
    project_path = os.path.join(yolov9_path, "runs", "detect")
    
    command = [
        "python", detect_script_path,
        "--weights", weights_path,
        "--source", input_folder,
        "--project", project_path,
        "--name", "exp",
        "--exist-ok",
        "--save-txt",
        "--save-conf",
        "--device", "0"
    ]
    
    try:
        subprocess.run(command, check=True)
        labels_path = os.path.join(project_path, "exp", "labels")
        print(f"Detection complete. Labels saved to: {labels_path}")
        return labels_path
    except Exception as e:
        print(f"An error occurred during YOLOv9 detection: {e}")
        return None

def convert_to_json(labels_folder, images_folder, output_json_path):

    print("Converting detection results to submission JSON format...")
    submission_list = []

    image_files = sorted(glob.glob(os.path.join(images_folder, '*.jpg')))

    for img_path in image_files:
        image_id = Path(img_path).stem
        txt_file = os.path.join(labels_folder, f"{image_id}.txt")
        
        h, w = cv2.imread(img_path).shape[:2]
        qrs_list = []

        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    if not line:
                        continue
                    
                    parts = line.split()
                    x_center, y_center, width, height, conf = map(float, parts[1:6])
                    
                    if conf < 0.56:
                         continue
                         
                    x_center_abs = x_center * w
                    y_center_abs = y_center * h
                    width_abs = width * w
                    height_abs = height * h

                    x_min = x_center_abs - width_abs / 2
                    y_min = y_center_abs - height_abs / 2
                    x_max = x_center_abs + width_abs / 2
                    y_max = y_center_abs + height_abs / 2
                    
                    qrs_list.append({"bbox": [x_min, y_min, x_max, y_max]})

        submission_list.append({"image_id": image_id, "qrs": qrs_list})

    with open(output_json_path, 'w') as f:
        json.dump(submission_list, f, indent=4)
        
    print(f"Submission JSON saved successfully at: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and generate submission file.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder of images.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output submission.json file.')
    
    parser.add_argument(
        '--weights', 
        type=str, 
        default="src/yolov9/runs/train/exp/weights/best.pt", 
        help='Path to the trained model weights (.pt file).'
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        print("Please train the model first using 'python train.py' or provide a valid path.")
    else:
        labels_output_folder = run_inference(args.input, args.weights)
        
        if labels_output_folder:
            convert_to_json(labels_output_folder, args.input, args.output)
