# Perception for Autonomous Systems - Final Project

This project implements a perception pipeline for autonomous systems, focusing on object detection, tracking, and stereo vision. It utilizes YOLOv8 for object detection and a modified SORT algorithm for robust multi-object tracking, including occlusion handling.
## Features

* **Object Detection:** Uses YOLOv8 to detect various classes of objects (Person, Bicycle, Car, Motorcycle, Bus, Truck).

* **Multi-Object Tracking:** Implements the SORT (Simple Online and Realtime Tracking) algorithm with enhancements for handling occlusions and scene changes.

* **Stereo Vision:** Includes tools for stereo camera calibration and image rectification to enable depth estimation.

* **Depth Estimation:** Integrates depth maps to provide distance measurements for tracked objects.

* **Visualization:** Real-time visualization of bounding boxes, object classes, and depth information.
## Project Structure

* `run_tracker.py`: The main entry point for running the detection and tracking pipeline.

* `tracking/`: Contains the implementation of the SORT tracker and utility functions.

* `stereo_calibration/`: Scripts for stereo camera calibration, rectification, and disparity calculation.

* `Detection_models/`: Directory for storing trained YOLO models.

* `output/`: Directory for saving processed videos and images.

* `data/`: Directory for dataset storage.
## Installation

Ensure you have Python installed. The project relies on the following external libraries:

* `ultralytics` (YOLOv8)

* `opencv-python` (cv2)

* `numpy`

* `scipy` (for linear assignment in tracking)

* `lap` (optional, for faster linear assignment)

You can install the required dependencies using pip:

```bash
pip install ultralytics opencv-python numpy scipy lap
```
## Data Setup

The `data` folder is not included in the repository. You need to recreate the folder structure and place your dataset (e.g., KITTI) in the appropriate locations.

1. **Create the Data Directory:**

Create a `data` folder in the project root.

2. **Raw Sequence Data:**

Place your raw stereo image sequences in `data/raw_seq1/`. The expected structure for the stereo pipeline is:

```
data/

└── raw_seq1/

├── image_02/

│ └── data/

│ ├── 0000000000.png

│ ├── ...

└── image_03/

└── data/

├── 0000000000.png

├── ...

```

3. **Calibration Data:**

If you are performing calibration, place your checkerboard images in `data/calib_data/`:

```

data/
|
└── calib_data/
	|
	├── image_02/
	|
	│   └── data/
	|
	|	    ├── *.png
	|
	└── image_03/
	|
	|	└── data/
	|	|
	|	|	├── *.png
	
```
4. **Configuration:**

Ensure you have the camera calibration file `config/calib_cam_to_cam.txt` if you are using the stereo pipeline.
## Usage

### 1. Prepare Rectified Images (Stereo Pipeline)

Before running the tracker, you need to rectify the raw stereo images and generate depth maps.

Run the stereo pipeline script:

```bash

python stereo_calibration/stereo_pipeline.py

```

This script reads from `data/raw_seq1/` and saves rectified images and depth maps to `output/seq1_rectified` (or similar, check the script variables).

### 2. Run the Tracker

Once you have the rectified images in the output folder (e.g., `output/seq2_rectified`), you can run the object detection and tracking pipeline.

```bash

python run_tracker.py

```

**Note:**

* Check `run_tracker.py` and ensure `IMAGE_FOLDER` points to your generated output directory (e.g., `output/seq1_rectified`).

* The default model is `Detection_models/best.pt`. You can specify a different model using the `--model` argument.

```bash

python run_tracker.py --model Detection_models/yolov8s.pt --output_video output/my_results.mp4

```

### Stereo Calibration (Optional)

If you need to recalculate stereo parameters:

```bash

python stereo_calibration/Calibration.py

```

This will save the calibration results to `output/stereo_params.npz`.
## Notes

* The tracker is configured to handle specific classes relevant to autonomous driving (Car, Pedestrian, Cyclist, etc.).

* The system expects input images in `output/seq2_rectified` by default. You may need to adjust `IMAGE_FOLDER` in `run_tracker.py` to point to your dataset.