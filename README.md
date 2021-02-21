# video-tracker

## Description

Application for tracking various objects in video stream. 

## Installation

### Clone repository and all submodules
- ```git clone https://github.com/vnkrtv/video-tracker.git && cd video-tracker```
- ```git pull --recurse-submodules```
- ```git submodule init```
- ```git submodule update --remote --recursive```

### OpenCV installation

Installing all system requirements described in [this article](https://funvision.blogspot.com/2019/12/opencv-web-camera-and-video-streams-in.html)  
After installing them is's able to build OpenCV

- ```cd opencv```
- ```mkdir build && cd build```
- ```cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_GSTREAMER=ON -D WITH_FFMPEG=ON [-D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1]..```
- ```make [-j 9]```
- ```sudo make install```

### dlib installation
- ```cd dlib/dlib```
- ```mkdir build && cd build```
- ```cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_SHARED_LIBS=1 -D USE_AVX_INSTRUCTIONS=ON [-D DLIB_USE_CUDA=1] ..```
- ```make [-j 9]```
- ```sudo make install```

### Building application

- ```cmake -DCMAKE_BUILD_TYPE=RELEASE .```
- ```cmake --build cmake-build-release --target video_tracker [-- -j 9]```

## Usage
```
Options: 

   --video-src, -v [string] Video source (video file, ip camera, video device)  
  --model-path, -m [string] MobileNetSSD folder path  
 --classes, -c [integer...] Set of detected classes ID. Full set could be found 
                            in README. Default classes: persons and cars  
  --confidence, -t [number] Model's confidence coefficient. Default value: 0.4  
                     --cuda Use GPU with CUDA  
```
## Model

MobileNet is using in project for objects detection. Model is pre-trained and taken from https://github.com/chuanqi305/MobileNet-SSD//. It was trained in Caffe-SSD framework. This model can detect 20 classes.
Available classes can be found in table below: 

| Class name   | Class ID  |
|--------------|----|
| background   | 0  |
| aeroplane    | 1  |
| bicycle      | 2  |
| bird         | 3  |
| boat         | 4  |
| bottle       | 5  |
| bus          | 6  |
| car          | 7  |
| cat          | 8  |
| chair        | 9  |
| cow          | 10 |
| dining table | 11 |
| dog          | 12 |
| horse        | 13 |
| motorbike    | 14 |
| person       | 15 |
| potted plant | 16 |
| sheep        | 17 |
| sofa         | 18 |
| train        | 19 |
| tv monitor   | 20 |

To make application detect multiple classes, you need specify special ```-c, --classes``` flag. Example:
- ```video_tracker --video-src /dev/video0 --model-path MobileNetSSD --classes {8,12}``` - in this case app will detectObjects only cats and dogs using camera /dev/video0