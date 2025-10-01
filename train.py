import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/demo_images/', help='dataset folder')
    parser.add_argument('--weights', type=str, default='yolov9/yolov9-s.pt', help='initial weights')
    parser.add_argument('--cfg', type=str, default='yolov9/models/detect/custom-yolov9-s.yaml')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()

def main():
    opt = parse_opt()
    os.chdir('yolov9')
    os.system(f'python train_dual.py --batch {opt.batch} --cfg {opt.cfg} '
              f'--epochs {opt.epochs} --data {opt.data} --weights {opt.weights} '
              f'--hyp data/hyps/hyp.scratch-high.yaml --device 0')

if __name__ == "__main__":
    main()
