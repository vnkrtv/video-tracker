cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 20)

project(video_tracker)

add_executable(video_tracker src/main.cpp
        src/args.hpp src/argparse.hpp src/processor.hpp src/model.hpp
        src/model.cpp src/multitracker.cpp src/multitracker.hpp
        src/db.cpp src/db.hpp
        src/speed_detector.cpp src/speed_detector.hpp src/processor.cpp)

find_package(SQLite3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(video_tracker ${SQLite3_LIBS})
target_link_libraries(video_tracker ${OpenCV_LIBS})
target_link_libraries(video_tracker sqlite3)
target_link_libraries(video_tracker dlib)

#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")