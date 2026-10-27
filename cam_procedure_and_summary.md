# Project Summary: RealSense D455 & YOLOv8 RTX 3050 Ti Robotics Perception Pipeline

This document outlines the complete procedure, technical decisions, and architectural rationale implemented for setting up the Intel RealSense D455 depth camera and YOLOv8 instance segmentation pipeline on Ubuntu 22.04 with an NVIDIA RTX 3050 Ti GPU and ROS 2 Humble.

---

## 1. System & Hardware Environment Setup

### Ubuntu 22.04 LTS & Partition Space Management
* **Action:** Cleaned and maintained the single 137 GB NVMe partition (`/dev/nvme0n1p4`), purging cached installer wheels (`pip`), obsolete apt `.deb` archives, and leftover virtual machine ISOs to reclaim 21+ GB of free space.
* **Why:** Heavy AI workloads (PyTorch, cuDNN, TensorRT) require substantial temporary and persistent storage during extraction and compilation. Maintaining sufficient free space prevented `Errno 28 (No space left on device)` errors.

### NVIDIA Driver 580 & Secure Boot
* **Action:** Installed the proprietary `nvidia-driver-580` supporting CUDA 13.0.
* **Why:** Ubuntu automatically leveraged pre-signed NVIDIA kernel modules (`linux-modules-nvidia-580-generic`) from official Canonical repositories, allowing seamless installation under UEFI Secure Boot without manual MOK (Machine Owner Key) enrollment prompts.

---

## 2. Python Environment Architecture

### Python 3.10 Virtual Environment (`cam_venv`) with `uv`
* **Action:** Installed `uv` globally and created a strict Python 3.10 virtual environment (`cam_venv`) without `--system-site-packages`.
* **Why:** 
  * **ROS 2 Humble Compatibility:** ROS 2 Humble relies on C++ extension modules compiled specifically for Python 3.10 (`cpython-310`). Using Python 3.12 would cause immediate `ModuleNotFoundError` or segmentation faults when importing `rclpy`.
  * **Isolation:** Keeping the venv isolated prevents global package clutter while allowing on-demand dynamic injection of ROS paths (`sys.path.append()`) when building robotics nodes.
  * **Performance:** `uv` handles parallel wheel downloads and hard-linking, reducing package installation times from minutes to seconds.

---

## 3. Perception & AI Model Optimization

### Upgrading from YOLOv8 Nano to Small Segmentation (`yolov8s-seg.pt`)
* **Action:** Swapped `yolov8n-seg.pt` for `yolov8s-seg.pt` in perception scripts.
* **Why:** Profiling via automated CSV performance logging (`ai_stream_metrics_*.csv`) demonstrated that the RTX 3050 Ti consumed only ~45 MB of VRAM and achieved a blistering ~9ms inference latency on the Nano model. Leveraging the remaining headroom with the Small (`s-seg`) model dramatically improved bounding box accuracy and pixel-level mask precision on specular/reflective objects (like metal water bottles) while maintaining well over 30 FPS.

---

## 4. Intel RealSense D455 Depth & 3D Dimension Estimation

### Hardware Post-Processing Filters
* **Action:** Integrated RealSense `spatial_filter` and `hole_filling_filter` into the pipeline.
* **Why:** Metal and glossy objects cause specular reflections and infrared light scatter, leading to missing depth data (zero-depth holes). Hardware post-processing smooths edges and fills missing depth points.

### Real-Time 3D Bounding Box Dimensions
* **Action:** Implemented optical-axis de-projection formulas in `test_d455_dimensions.py`:
  $$\text{Real Width} = \frac{(x_2 - x_1) \cdot Z}{f_x}, \quad \text{Real Height} = \frac{(y_2 - y_1) \cdot Z}{f_y}, \quad \text{Real Length} = z_{\max} - z_{\min}$$
* **Why:** Moving from simple 2D bounding boxes to 3D physical measurements ($\text{Width} \times \text{Height} \times \text{Length}$) in meters/centimeters provides downstream robotic manipulation pipelines (such as MoveIt2 or ROS 2 grasp planners) with the exact spatial orientation and dimensions required to successfully pick up objects.