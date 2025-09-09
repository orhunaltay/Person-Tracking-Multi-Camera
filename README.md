# Person-Tracking-Multi-Camera

## Create the Environment
```
conda create -n face_track python -y
conda activate face_track
```
## Install the Requirements
Install the requirements inside the conda environment.

```
pip install requirements.txt
```
## For Dual Camera Use
I used Camo Studio to connect my phone and used my webcam for the second camera. But you can use another webcam for the camera if the camera works with your computer then the system will also recognize the camera.

## System Requirements
My computer is an Apple Silicon M4 chip therefore I used metal for the OpenCV needs therefore it is on the CPU system however it will also work on a GPU system.

## Running the System
After completing the requirements and the connecting a external camera run the below command inside the `face_track` environment.

```
python collective_tracking_2cams.py
```
