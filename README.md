# Multi-QR Code Detection using YOLOv9

This repository contains the complete source code for a multi-QR code detection model, submitted for the MultiQR Hackathon. The solution is built upon the YOLOv9 object detection framework and includes a pre-trained model for immediate inference.

---

## 1. Setup Instructions

Follow these steps to set up the environment and prepare the project.


### Installation Steps

1.  **Clone this Submission Repository:**
    ```bash
    git clone <your-github-repo-url>
    cd multiqr-hackathon
    ```

2.  **Clone the YOLOv9 Framework:**
    The core model code is required from the official YOLOv9 repository. The following command clones it directly into the `src/` directory, which is where the scripts expect it to be.
    ```bash
    git clone https://github.com/WongKinYiu/yolov9.git src/yolov9
    ```

3.  **Install Dependencies:**
    Install all the necessary Python libraries using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. How to Run Inference (Using the Provided Trained Model)

A pre-trained `best.pt` model is included in the `trained_models/` directory. You can use it directly to generate predictions.

### Command Format
The inference script requires you to specify the input image folder, the output JSON path, and the path to the model weights.
```bash
python infer.py --input <path_to_images_folder> --output <path_to_save_json> --weights <path_to_model.pt>

