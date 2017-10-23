# Usage:
#
#    $ mkdir build && cd build
#    $ cmake ..


cmake_minimum_required(VERSION 2.8)
project( FindMarkers )
find_package( OpenCV REQUIRED )
add_executable( FindMarkers detect_markers.cpp )
target_link_libraries( FindMarkers ${OpenCV_LIBS} )

FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT)

FIND_PATH(ARTP_INCLUDES ARToolKitPlus/ARToolKitPlus.h)
FIND_LIBRARY(ARTP_LIBS ARToolkitPlus)
INCLUDE_DIRECTORIES(${ARTP_INCLUDES})

ADD_EXECUTABLE(multi multiTest.c)
TARGET_LINK_LIBRARIES(multi ${ARTP_LIBS} ${OpenCV_LIBS} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLUT_glut_LIBRARY})
