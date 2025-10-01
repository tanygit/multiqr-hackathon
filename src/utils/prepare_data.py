import os
import argparse

def create_file_lists(dataset_path):
  
    print(f"Processing dataset at: {dataset_path}")
 
    train_img_path = os.path.join(dataset_path, "images", "train")
    val_img_path = os.path.join(dataset_path, "images", "val")
    train_txt_path = os.path.join(dataset_path, "train.txt")
    val_txt_path = os.path.join(dataset_path, "val.txt")

    # Create train.txt
    try:
        with open(train_txt_path, "w") as f:
            img_list = sorted(os.listdir(train_img_path))
            for img_name in img_list:
                full_path = os.path.abspath(os.path.join(train_img_path, img_name))
                f.write(full_path + '\n')
        print(f"Successfully created {train_txt_path} with {len(img_list)} entries.")
    except FileNotFoundError:
        print(f"Error: Directory not found at {train_img_path}. Please check your dataset path.")
        return

    # Create val.txt
    try:
        with open(val_txt_path, "w") as f:
            img_list = sorted(os.listdir(val_img_path))
            for img_name in img_list:
                full_path = os.path.abspath(os.path.join(val_img_path, img_name))
                f.write(full_path + '\n')
        print(f"Successfully created {val_txt_path} with {len(img_list)} entries.")
    except FileNotFoundError:
        print(f"Error: Directory not found at {val_img_path}. Please check your dataset path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data lists (train.txt, val.txt) for YOLOv9 training.")
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='Path to the root of your dataset directory (e.g., /path/to/QR_Dataset).'
    )
    args = parser.parse_args()
    
    create_file_lists(args.path)