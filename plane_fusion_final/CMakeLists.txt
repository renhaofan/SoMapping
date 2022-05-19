CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(PlaneFusion)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

# specify OpenGL version
# credit: https://blog.csdn.net/u010815558/article/details/90288869
# credit: https://github.com/ilpincy/argos3/issues/88
if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

#################################
# Add additional compiler flags #
#################################
IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
ENDIF()
IF(NOT MSVC_IDE)
    SET(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
    #  SET(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++")
    SET(CMAKE_CXX_FLAGS "-fPIC -O3 ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++")
    #SET(CMAKE_CXX_FLAGS "-fPIC -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF()

################################
# Specify the libraries to use #
################################
# opencv 3.4.5
SET(OpenCV_DIR "/usr/local/opencv345/share/OpenCV")
SET(CUDA_HEADERS "${PROJECT_SOURCE_DIR}/3rdparty/cuda-samples/Common")

FIND_PACKAGE(OpenCV REQUIRED)#opencv 3.4.5
FIND_PACKAGE(Eigen REQUIRED) #eigen 3.3.7
FIND_PACKAGE(Ceres REQUIRED) #1.14.0
FIND_PACKAGE(OpenGL REQUIRED)#4.6 `glxinfo | grep "OpenGL"`
ADD_LIBRARY(GLEW_STATIC STATIC IMPORTED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(DBoW3 REQUIRED) #master
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake) #11.6
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseGLUT.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseGLEW.cmake)

INCLUDE_DIRECTORIES(
        ${GLEW_INCLUDE_DIRS}
        #/usr/local/cuda/samples/common/inc
        ${CUDA_HEADERS}
        ${CMAKE_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenGL_INCLUDE_DIRS}
        ${DBoW3_INCLUDE_DIRS}
)

message(STATUS "-------------------------------------------")
message(STATUS "OpenCV lib")
message(STATUS "        dir:${OpenCV_DIR}")
message(STATUS "        version:${OpenCV_VERSION}")
message(STATUS "        includedir:${OpenCV_INCLUDE_DIRS}")
#message(STATUS "        libdir:${OpenCV_LIBS}")
message(STATUS "-------------------------------------------")


#############################
# Specify the project files #
#############################
#add_executable(${PROJECT_NAME} ./Plane_detector/Plane_detector.h)
INCLUDE(${PROJECT_SOURCE_DIR}/IncludeProjectFiles.cmake)
SET(THIRD_LIB
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${DBoW3_LIBS}
    ${OPENGL_LIBRARY}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARY}
    )
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES} ${THIRD_LIB})
SET(HEADERS ./Plane_detector/Plane_detector.h)

##########################################
# Specify the target and where to put it #
##########################################
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

IF(WITH_CUDA)
    CUDA_ADD_EXECUTABLE(FUSION main_code.cpp )
ELSE()
    ADD_EXECUTABLE(FUSION main_code.cpp )
ENDIF()

TARGET_LINK_LIBRARIES(FUSION ${PROJECT_NAME} ${CUDA_LIBRARY} rt cuda)
