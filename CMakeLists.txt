cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 20)

project( video_detect )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(video_detect src/main.cpp src/args.hpp src/argparse.hpp src/detector.hpp src/model.hpp src/model.cpp)
target_link_libraries( video_detect ${OpenCV_LIBS} )
