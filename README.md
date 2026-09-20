# 🚗 Real-Time Traffic Density Estimation with YOLOv8

## 🔍 Overview
This project harnesses the power of YOLOv8's real-time detection capabilities to tackle Traffic Density Estimation, a crucial aspect of urban and traffic management systems. The primary objective is to accurately count vehicles within designated areas in video frames to evaluate traffic flow. The insights garnered from this data are instrumental in pinpointing peak traffic times, identifying bottlenecks, and aiding urban planning.

This repository adds a Docker Compose pipeline for downloading the Kaggle dataset, preparing YOLO paths, fine-tuning, and running headless inference.


## 🎯 Objectives
The pivotal milestones achieved in our project include:
* **YOLOv8 Model Selection and Assessment:** Commencing with the selection of a pre-trained YOLOv8 model and evaluating its baseline performance on the COCO dataset for vehicle detection purposes.
* **Specialized Vehicle Dataset Curation:** Assembling and annotating a targeted dataset dedicated to vehicles to enhance the model's detection accuracy for a range of vehicle types.
* **Model Refinement for Superior Detection:** Applying transfer learning techniques to fine-tune the YOLOv8 model, with a special focus on detecting vehicles from aerial views, thus significantly improving precision and recall rates.
* **Thorough Evaluation of Model Performance:** Conducting a detailed analysis of learning curves, confusion matrices, and performance metrics to ensure the model's reliability and its capability to generalize.
* **Generalization and Inference on Test Data:** Verifying the model's robustness through generalization tests using validation images, an unseen test image, and a test video, showcasing its real-world applicability and efficiency.
* **Real-Time Traffic Analysis:** Developing an algorithm to quantify traffic density by real-time vehicle counting and traffic intensity analysis on test video footage.
* **Preparation for Cross-Platform Deployment:** Exporting the optimized model in the ONNX format to ensure cross-platform compatibility and facilitate deployment across diverse environments.


## 📚 Dataset Description

### 🌐 Overview
The **Top-View Vehicle Detection Image Dataset for YOLOv8** is essential for tasks like traffic monitoring and urban planning. It provides a unique perspective on vehicle behavior and traffic patterns from aerial views, facilitating the creation of AI models that can understand and analyze traffic flow comprehensively.

### 🔍 Specifications
- 🚗 **Class**: 'Vehicle' including cars, trucks, and buses.
- 🖼️ **Total Images**: 626
- 📏 **Image Dimensions**: 640x640 pixels
- 📂 **Format**: YOLOv8 annotation format

### 🔄 Pre-processing
Each image is carefully pre-processed and standardized to ensure consistency and high-quality training data for our model.

### 🔢 Dataset Split
The dataset is meticulously split into:
- **Training Set**: 536 images for model training with diverse scenarios.
- **Validation Set**: 90 images for unbiased model performance evaluation.

### 🎭 Augmentation on Training Set
Augmentations, including horizontal flips, are applied to enhance the training set's robustness, ensuring the model learns to generalize well across varied traffic conditions.

### 🚀 Significance
This dataset is pivotal in developing sophisticated vehicle detection models and shaping intelligent transportation systems for smarter city infrastructures.

### 🗃️ Sources
- The dataset is curated from [Pexels](https://www.pexels.com/search/videos/), offering diverse top-view videos for a rich vehicle detection dataset.
- Images were extracted using [Roboflow](https://universe.roboflow.com/farzad/vehicle_detection_yolov8) at a rate of 1 frame per second, ensuring a wide variety of vehicle types and scenarios.

### 📌 Access
The dataset is publicly available on Roboflow and Kaggle:
- Roboflow Project: [Top-View Vehicle Detection](https://universe.roboflow.com/farzad/vehicle_detection_yolov8)
- Kaggle Dataset: [Top-View Vehicle Detection Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset)


## 🎥 YouTube Demo
Real-Time Traffic Density Estimation with YOLOv8 in Action:

[![Traffic Density Estimation Demo](https://img.youtube.com/vi/5SxQfWLENh8/0.jpg)](https://youtu.be/5SxQfWLENh8)


## 📁 File Descriptions

- **`AGENTS.md`**: Instructions for coding agents (layout, ADR 0001, Compose, public API).
- **`docs/adr/`**: Architecture Decision Records. [0001](docs/adr/0001-modular-oop-and-delivery.md) is binding for new Python, Compose, and README changes.
- **`scripts/traffic_analysis/`**: OOP inference package. Each file owns one class: private `_` helpers and fields, public methods plus read-only `@property`. Reading order is in `__init__.py`. Run with `python -m scripts.traffic_analysis`.
- **`scripts/real_time_traffic_analysis.py`**: Thin launcher equivalent to the module above.
- **`scripts/kaggle_download/`**: OOP Kaggle download stage; run with `python -m scripts.kaggle_download`.
- **`scripts/dataset_preparation/`**: OOP dataset discovery and `data.yaml` normalization stage; run with `python -m scripts.dataset_preparation`.
- **`scripts/model_training/`**: OOP YOLO training and artifact export stage; run with `python -m scripts.model_training`.
- **`scripts/model_quantization/`**: OOP OpenVINO FP32/INT8 export and comparison stage; run with `python -m scripts.model_quantization`.
- **`scripts/download_kaggle_dataset.py`**, **`prepare_dataset.py`**, **`train_model.py`**: Compatibility launchers used by Compose; all behavior lives in the packages above.
- **`tests/`**: Standard-library unit tests for the functional stages; no Kaggle access, model download, or GPU is required.
- **`Dockerfile`**: CPU inference image (CPU PyTorch wheels).
- **`Dockerfile.train`**: Training image (PyTorch CUDA runtime + Kaggle CLI).
- **`Dockerfile.quantize`**: CPU OpenVINO/NNCF quantization image.
- **`compose.yaml`**: Independent Compose services for download, prepare, train, quantization, and inference.
- **`compose.gpu.yaml`**: NVIDIA GPU override for the `train` service.
- **`.env.example`**: Template for Kaggle credentials and optional training overrides.
- **`requirements.txt`**: Python dependencies for CPU inference.
- **`requirements-train.txt`**: Python dependencies for training.
- **`requirements-quantization.txt`**: Extra NNCF dependency for INT8 post-training quantization.
- **`LICENSE.txt`**: Project license.
- **`data/`**: Created at runtime (gitignored). Raw Kaggle extract, prepared `data.yaml`, training runs, and copied sample video.
- **`models/`**: Created at runtime. Holds trained, OpenVINO, and comparison artifacts.
- **`output/`**: Created at runtime (gitignored). Docker inference writes `processed_sample_video.avi` here.

Weights, the demo video, cover images, the original Jupyter notebook, and the demo GIF are not stored in this tree. Train to produce weights; copy or download `sample_video.mp4` for inference. The original notebook lives on [Kaggle](https://www.kaggle.com/code/farzadnekouei/real-time-traffic-density-estimation-with-yolov8) and in the [upstream GitHub repo](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation).


## 🐳 Run with Docker Compose

Use `docker compose` (plugin), not `docker-compose`. Every training/inference stage is an independent one-shot service. Do **not** run `docker compose up` without a service name: that would start download, prepare, train, and inference together.

| Service | Purpose |
|---------|---------|
| `download-data` | Kaggle download into `data/raw/` |
| `prepare-data` | Path fix-up → `data/dataset/data.yaml` |
| `train` | Fine-tune + export weights |
| `quantize-model` | Export OpenVINO FP32/INT8 + compare accuracy, latency, and size |
| `traffic-analysis` | Headless inference (`DISPLAY_VIDEO=false`) |

### Training pipeline (Kaggle)

**Requirements:** Docker Compose and Kaggle API credentials. GPU training additionally
requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

1. Copy `.env.example` to `.env` and set `KAGGLE_USERNAME` / `KAGGLE_KEY`, **or** put `kaggle.json` in `./.kaggle/`.
2. Accept the dataset rules on Kaggle for [Top-View Vehicle Detection](https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset).

Complete the one-time setup first:

```bash
mkdir -p data models .kaggle output && cp .env.example .env
```

Edit `.env` with your Kaggle API key, then run and verify each stage independently.
Each command exits after completing only its own stage.

**Stage 1 — Download the Kaggle dataset**

```bash
docker compose run --rm download-data
```

Verify that `data/raw/.download_complete` and the extracted dataset exist.

**Stage 2 — Prepare the YOLO dataset configuration**

```bash
docker compose run --rm prepare-data
```

Verify `data/dataset/data.yaml`. If the dataset includes the demo video, this stage
also writes `data/sample_video.mp4`.

**Stage 3 — Train and export with an NVIDIA GPU**

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm train
```

Verify the training run in `data/runs/detect/train/` and the exported models at
`models/best.pt` and `models/best.onnx`.

To train on CPU instead, use `docker compose run --rm train` (Compose sets
`TRAIN_DEVICE=cpu`). Stage 2 fails clearly when Stage 1 has not produced a dataset,
and Stage 3 fails clearly when Stage 2 has not produced `data/dataset/data.yaml`;
neither command automatically runs an earlier stage.

Optional overrides in `.env`: `TRAIN_EPOCHS`, `TRAIN_BATCH`, `BASE_MODEL`,
`TRAIN_DEVICE`, `KAGGLE_DATASET`. Extra training knobs used by `scripts/train_model.py`
include `TRAIN_PATIENCE`, `TRAIN_LR0`, `TRAIN_LRF`, `TRAIN_DROPOUT`, `TRAIN_SEED`,
and `EXPORT_ONNX`. Dataset preparation also accepts `SAMPLE_VIDEO_OUTPUT` (default
`/data/sample_video.mp4`).

#### Fix NVIDIA runtime errors on WSL2 / Linux

If the host `nvidia-smi` works but Compose reports the following error, the GPU driver
is available to WSL but the Docker daemon does not have the NVIDIA runtime:

```text
could not select device driver "nvidia" with capabilities: [[gpu]]
```

For Docker Engine installed directly inside WSL/Linux, install and configure the
NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

Verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If Docker Desktop supplies the Docker daemon, update Docker Desktop and the Windows
NVIDIA driver, enable this WSL distribution under **Settings → Resources → WSL
Integration**, then restart Docker Desktop. Do not configure a second Docker daemon
inside WSL.

After the verification command succeeds, run Stage 3:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm train
```

Default `docker compose run --rm train` uses **CPU**, so it does not fail when the
NVIDIA runtime is missing. GPU training must merge `compose.gpu.yaml`. Weights are
written to `models/best.pt` and `models/best.onnx`.

### OpenVINO INT8 quantization and comparison

After training and dataset preparation, run the independent CPU quantization stage:

```bash
docker compose run --rm --build quantize-model
```

It calibrates INT8 with the dataset named by `data/dataset/data.yaml`, exports both
models from the same `best.pt`, and validates them under the same OpenVINO runtime:

- `models/best_fp32_openvino_model/`: the pre-quantization FP32 baseline
- `models/best_int8_openvino_model/`: the post-training INT8 model
- `models/quantization_comparison.json`: mAP50-95, mAP50, inference ms/image,
  artifact bytes, accuracy deltas, speedup, and compression ratio

The comparison deliberately uses OpenVINO for both models, so the reported delta is
caused by precision rather than by changing from PyTorch to OpenVINO. Latency is
hardware- and workload-specific; compare results produced by the same container run.
Optional `.env` overrides are `QUANT_IMGSZ`, `QUANT_BATCH`, `QUANT_DEVICE`, and
`QUANT_CALIBRATION_FRACTION` (range `(0, 1]`, default `1.0`).

### Inference (traffic analysis)

**Prerequisites:** Docker with Compose plugin, `models/best.pt`, and `sample_video.mp4`
at the **repository root** (Compose bind-mounts `./sample_video.mp4`). After Stage 2:

```bash
cp data/sample_video.mp4 sample_video.mp4
mkdir -p models output
docker compose run --rm --build traffic-analysis
```

The first build may take several minutes (CPU PyTorch wheels in the inference image).

Output: `output/processed_sample_video.avi`. The container runs headless
(`DISPLAY_VIDEO=false`); there is no OpenCV window.

Select the inference artifact with `MODEL_VARIANT`. Supported values are `pytorch`
(default), `openvino-fp32`, and `openvino-int8`:

```bash
MODEL_VARIANT=openvino-int8 \
  docker compose run --rm --build traffic-analysis
```

`MODEL_PATH` remains available as a full-path override and takes precedence over
the selected variant.


## 🚀 Instructions for Local Execution

### 1️⃣. Initial Setup
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation.git
    ```
    This Dockerized copy follows the same layout for inference (`scripts/traffic_analysis/`, `models/best.pt`, `sample_video.mp4`).
2. **Navigate to the Project Directory**:
    ```bash
    cd YOLOv8_Traffic_Density_Estimation
    ```

### 2️⃣. Model development
Use the Compose training stages above, or open the original pipeline on
[Kaggle](https://www.kaggle.com/code/farzadnekouei/real-time-traffic-density-estimation-with-yolov8)
/ the [upstream notebook](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation).
This working tree does not include `real-time_traffic_density_estimation_yolov8.ipynb`.

### 3️⃣. Watching the Real-Time Performance
1. Place the selected model under `models/` and `sample_video.mp4` in the project root (or set `MODEL_PATH` / `VIDEO_PATH`). Local `MODEL_VARIANT` values resolve to `models/best.pt`, `models/best_fp32_openvino_model/`, or `models/best_int8_openvino_model/`.
2. Install inference dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (`ultralytics` will pull a PyTorch build; for a display window you need a non-headless OpenCV, so you may prefer `pip install ultralytics opencv-python` instead of `opencv-python-headless`.)
3. Run the analysis package from the repository root:
    ```bash
    python -m scripts.traffic_analysis
    ```
    Defaults: `DISPLAY_VIDEO=true` (press `q` to quit), output `processed_sample_video.avi` in the current directory. Headless:
    ```bash
    DISPLAY_VIDEO=false python -m scripts.traffic_analysis
    ```

### Run unit tests

The functional-stage tests use temporary files and test doubles, so they do not
download data or run YOLO:

```bash
python -m unittest discover -v
```


## 🔗 Additional Resources

- 🎥 **Project Demo**: Watch the live demonstration of this project on [YouTube](https://www.youtube.com/watch?v=5SxQfWLENh8).
- 🌐 **Kaggle Notebook**: Interested in a Kaggle environment? Explore the notebook [here](https://www.kaggle.com/code/farzadnekouei/real-time-traffic-density-estimation-with-yolov8).
- 🌐 **Dataset Source**: Available on both [Roboflow](https://universe.roboflow.com/farzad/vehicle_detection_yolov8) and [Kaggle](https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset).
- 🤝 **Connect on LinkedIn**: Have questions or looking for collaboration? Let's connect on [LinkedIn](https://linkedin.com/in/farzad-nekouei-7535aa53/).
