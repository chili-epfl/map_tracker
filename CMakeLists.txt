# Usage:
#
#    $ mkdir build && cd build
#    $ cmake ..  -DANDROID_WRAPPER=ON -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DOpenCV_DIR=../../../deps/OpenCV-android-sdk/sdk/native/jni/
#    or
#    $cmake ../source  -DANDROID_WRAPPER=ON -DCMAKE_TOOLCHAIN_FILE=../source/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DOpenCV_DIR=~/Libs/OpenCV-android-sdk/sdk/native/jni/
#    or
#    $cmake same_previous -DJAVA_INCLUDE_PATH=/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/include 
#           -DJAVA_AWT_LIBRARY=/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/libawt.dylib
#           -DJAVA_JVM_LIBRARY=/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/server/libjvm.dylib

cmake_minimum_required(VERSION 2.8)
project(ThymioTracker)

set(ANDROID_WRAPPER OFF CACHE BOOL "Compile for Android with Java wrapper")

set(MapTracker_SOURCES
    src/Map_Tracker.hpp
    src/Map_Tracker.cpp
    src/Landmark.hpp
    src/Landmark.cpp
    src/Generic.hpp
    src/Generic.cpp
    src/Calibrator.hpp
    src/Calibrator.cpp
    )

find_package(OpenCV REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-long-long -Wno-vla -pedantic")

# add_subdirectory(brisk)

include_directories(${PROJECT_SOURCE_DIR}/src
                    ${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS})


    
add_library(map_tracker SHARED ${MapTracker_SOURCES})
add_subdirectory(tests)
add_subdirectory(tools)


target_link_libraries(map_tracker ${OpenCV_LIBRARIES})
