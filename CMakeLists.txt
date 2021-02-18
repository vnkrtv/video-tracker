cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 20)

project(video_tracker)

#set(CMAKE_CXX_COMPILER ${CMAKE_SOURCE_DIR}/bin/i686-linux-gnu-g++)
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")
set(WITH_GSTREAMER ON)
set(WITH_FFMPEG=ON)
#set(DCMAKE_SHARED_LINKER_FLAGS " -Wl,-rpath-link $FFMPEG_LIBDIR -L$FFMPEG_LIBDIR")
#set(DCMAKE_MODULE_LINKER_FLAGS " -Wl,-rpath-link $FFMPEG_LIBDIR -L$FFMPEG_LIBDIR")
#set(DCMAKE_EXE_LINKER_FLAGS " -static-libgcc -static-libstdc++ -Wl,-rpath-link $FFMPEG_LIBDIR -L$FFMPEG_LIBDIR")
#add_subdirectory(dlib)
#add_subdirectory(opencv)
#add_library(dlib SHARED IMPORTED)
#set_target_properties(dlib PROPERTIES IMPORTED_LOCATION "dlib/build/dlib/libdlib.a")
#include(dlib/cmake)

add_executable(video_tracker src/main.cpp
        src/args.hpp src/argparse.hpp src/detector.hpp src/model.hpp
        src/model.cpp src/tracker.cpp src/tracker.hpp src/db.cpp src/db.hpp)

#set_target_properties(video_detect PROPERTIES LINK_SEARCH_START_STATIC 1)
#set_target_properties(video_detect PROPERTIES LINK_SEARCH_END_STATIC 1)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_package(SQLite3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(video_tracker ${SQLite3_LIBS})
target_link_libraries(video_tracker ${OpenCV_LIBS})
target_link_libraries(video_tracker sqlite3)
target_link_libraries(video_tracker dlib)

#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")