import os
import glob
import json
import cv2
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input folder with images')
    parser.add_argument('--output', type=str, required=True, help='output JSON file')
    parser.add_argument('--weights', type=str, default='yolov9/runs/train/exp/weights/best.pt')
    return parser.parse_args()

def run_inference(input_folder, output_json, weights):
    os.chdir('yolov9')
    os.system(f'python detect_dual.py --weights {weights} --source {input_folder} '
              f'--save-txt --save-conf --project ../yolov9/runs/detect --name exp_test_images --exist-ok')
    
    pred_folder = '../yolov9/runs/detect/exp_test_images/labels'
    submission_list = []

    for txt_file in sorted(glob.glob(os.path.join(pred_folder, '*.txt'))):
        image_id = os.path.splitext(os.path.basename(txt_file))[0]
        img_path = os.path.join(input_folder, image_id + '.jpg')
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        qrs_list = []
        with open(txt_file, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                if line == '':
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                x_center, y_center, width, height, conf = map(float, parts[1:6])
                if conf < 0.5:
                    continue
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                qrs_list.append({"bbox": [x_min, y_min, x_max, y_max]})
        submission_list.append({"image_id": image_id, "qrs": qrs_list})

    with open(output_json, 'w') as f:
        json.dump(submission_list, f, indent=4)
    print(f"Submission JSON saved at: {output_json}")

def main():
    opt = parse_opt()
    run_inference(opt.input, opt.output, opt.weights)

if __name__ == "__main__":
    main()
