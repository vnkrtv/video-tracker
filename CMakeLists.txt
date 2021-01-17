cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 20)

project( video_detect )

#set(CMAKE_CXX_COMPILER ${CMAKE_SOURCE_DIR}/bin/i686-linux-gnu-g++)
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")
#add_subdirectory(dlib)

add_executable(video_detect src/main.cpp
        src/args.hpp src/argparse.hpp src/detector.hpp src/model.hpp
        src/model.cpp src/tracker.cpp src/tracker.hpp)

#set_target_properties(video_detect PROPERTIES LINK_SEARCH_START_STATIC 1)
#set_target_properties(video_detect PROPERTIES LINK_SEARCH_END_STATIC 1)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( video_detect ${OpenCV_LIBS} )
#target_link_libraries( video_detect dlib::dlib)

#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")