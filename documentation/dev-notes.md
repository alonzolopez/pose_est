# Design Requirements
The goal of this work is to provide a repurposable 6DOF pose estimation algorithm in a ROS package to provide advanced sensing for deployed robotic systems.
- **Repurposable** pose estimation (i.e. can estimate the pose of new objects if data is generated for them).
- **Edge-compatible** algorithm and software for pose estimation that can run on edge devices (e.g. the Jetson Xavier NX or Jetson Nano) at usable rates (e.g. 5 Hz or faster).
- **Markerless** pose estimation algorithms that work on unprepared objects without fiducials. 

# Resources
- [MobileNet V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) and [MobileNet V3 Large](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large) and [MobileNet V3 Small](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small)
- [Transfer Learning with TensorFlow Hub](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)
- [Transfer Learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Load images](https://www.tensorflow.org/tutorials/load_data/images)
- Use [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor the training.
- [Camera-to-Robot Pose Estimation from a Single Image](https://research.nvidia.com/publication/2020-03_DREAM)
- [Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects](https://arxiv.org/abs/1809.10790) and the [DOPE repo](https://github.com/NVlabs/Deep_Object_Pose)
- [Indirect Object-to-Robot Pose Estimation from an External Monocular RGB Camera](https://research.nvidia.com/publication/2020-07_Indirect-Object-Pose)
- [NVIDIA Dataset Utilities](https://github.com/NVIDIA/Dataset_Utilities) a.k.a. NVDU
- [NVIDIA Deep Learning Dataset Synthesizer](https://github.com/NVIDIA/Dataset_Synthesizer) a.k.a. NDDS
- [FAT Dataset](https://research.nvidia.com/publication/2018-06_Falling-Things), a.k.a. Falling Things: A Synthetic Dataset for 3D Object Detection and Pose Estimation
- [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/master/d9/d0c/group__calib3d.html) (OpenCV docs on SolvePnP and Camera Calibration)
- [Keras Computer Vision Code Examples](https://keras.io/examples/vision/)

# Dev Notes
## Virtual Environment Setup
Setup the virtual environment for this project.

On Ubuntu:
```bash
python3 -m venv pose-est-env
source pose-est-env/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --upgrade setuptools
pip install -r ./requirements.txt
curl -fsSL https://deb.nodesource.com/setup_15.x | sudo -E bash -
sudo apt install npm nodejs
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
```
On Windows:
```bash
python -m venv pose-est-env
.\pose-est-env\Scripts\activate
pip install -r .\requirements.txt
```

## Dataset Setup
1. Install the [NVDU tool](https://github.com/NVIDIA/Dataset_Utilities) from source.

    ```
    cd ~/catkin_ws/src/pose_est/
    git clone https://github.com/NVIDIA/Dataset_Utilities.git
    cd Dataset_Utilities
    pip install -e .
    ```
2. Setup the YCB objects while in the same directory
    ```
    nvdu_ycb -s
    ```
3. Download the dataset from the [Falling things page](https://research.nvidia.com/publication/2018-06_Falling-Things) (*External Links* > *dataset*). It will take a while to download the whole 42Gb dataset. While you're waiting, read the [FAT Dataset](fat_dataset.md) readme.

    Extract the dataset to the `~catkin_ws/src/pose_est/data/` folder.


>Note, when trying to use the [NVDU tool](https://github.com/NVIDIA/Dataset_Utilities) to list the available objects or download the YCB 3D model dataset, the following errors are observed:
>```
> nvdu_ycb -l
> Supported YCB objects:
> Traceback (most recent call last):
>   File "/home/alonzo/catkin_ws/src/pose_est/pose-est-env/bin/nvdu_ycb", line 8, in <module>
> sys.exit(main())
>   File "/home/alonzo/catkin_ws/src/pose_est/pose-est-env/lib/python3.8/site-packages/nvdu/tools/nvdu_ycb.py", line 234, in main
>     log_all_object_names()
>   File "/home/alonzo/catkin_ws/src/pose_est/pose-est-env/lib/python3.8/site-packages/nvdu/tools/nvdu_ycb.py", line 105, in log_all_object_names
>     for obj_name, obj_settings in all_ycb_object_settings.obj_settings.items():
> AttributeError: 'NoneType' object has no attribute 'obj_settings'
> ```
> ```
> nvdu_ycb -s
> Traceback (most recent call last):
>   File "/home/alonzo/catkin_ws/src/pose_est/pose-est-env/bin/nvdu_ycb", line 8, in <module>
>     sys.exit(main())
>   File "/home/alonzo/catkin_ws/src/pose_est/pose-est-env/lib/python3.8/site-packages/nvdu/tools/nvdu_ycb.py", line 237, in main
>     setup_all_ycb_models()
>   File "/home/alonzo/catkin_ws/src/pose_est/pose-est-env/lib/python3.8/site-packages/nvdu/tools/nvdu_ycb.py", line 214, in setup_all_ycb_models
>     for obj_name, obj_settings in all_ycb_object_settings.obj_settings.items():
> AttributeError: 'NoneType' object has no attribute 'obj_settings'
> ```
> However, `nvdu_ycb -h` does work.


## Visualizing the Dataset
To visualize data in the dataset, navigate to a folder and use the `nvdu_viz` command to visualize select images. 
```
nvdu_viz -m <path-to-model> -n <selected-images>
```

For example,
```
cd ~/catkin_ws/src/pose_est/data/fat/single/010_potted_meat_can_16k/kitchen_0
nvdu_viz -m /home/alonzo/catkin_ws/src/pose_est/Dataset_Utilities/nvdu/data/ycb/original/010_potted_meat_can/google_16k -n 000000* 000001*
```

Use the [NVDU Controls](https://github.com/NVIDIA/Dataset_Utilities#controls) to control how the frames, keypoints, and models render.

>**Note:** some of the images may throw an error. For example, 
>```
>nvdu_viz -m /home/alonzo/catkin_ws/src/pose_est/Dataset_Utilities/nvdu/data/ycb/original/010_potted_meat_can/google_16k -n 000002*
>```
>because they try to visualize the json:
>```
>visualize_dataset_frame: frame_image_file_path: ./000002.left.json - frame_data_file_path: ./000002.left.json
>```
>but they work when you specify the image path in more detail:
>```
>nvdu_viz -m /home/alonzo/catkin_ws/src/pose_est/Dataset_Utilities/nvdu/data/ycb/original/010_potted_meat_can/google_16k -n 000002.right* 000002.left.jpg
>```
> The following error was also observed:
> ```
> pyglet.gl.ContextException: Could not create GL context
> ```
> and the solution was to restart the computer.

## Interpreting the Dataset
See [File Details](fat_dataset.md#file-details) for details on the annotation files for both camera properties and json image labels.
