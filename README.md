# video-tracker

## Description

Coming soon...

## Installation

- ```git clone https://github.com/vnkrtv/video-tracker.git && cd video-tracker```
- ```cat goturn-files/goturn.caffemodel.zip* > goturn-files/goturn.caffemodel.zip```
- ```unzip goturn-files/goturn.caffemodel.zip -d goturn-files/ && rm goturn-files/goturn.caffemodel.zip*```
- ```cmake -DCMAKE_BUILD_TYPE=Debug .```
- ```cmake --build cmake-build-debug --target video_tracker -- -j 9```

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

MobileNet is using in project for objects detection. Model is pre-trained and taken from https://github.com/chuanqi305/MobileNet-SSD//. It was trained in Caffe-SSD framework. This model can detectObjects 20 classes.
Some available classes can be found in table below: 

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

To make application detectObjects multiple classes, you need specify special ```-c, --classes``` flag. Example:
- ```video_tracker --video-src /dev/video0 --model-path MobileNetSSD --classes {8,12}``` - in this case app will detectObjects only cats and dogs using camera /dev/video0