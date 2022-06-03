cmake_minimum_required(VERSION 3.10)

project(PlaneFusion)

# std c++ 11
set(CMAKE_CXX_STANDARD 11)
# add cmake path, have no idea about why include needed indeed or link glew bug.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UseGLUT.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UseGLEW.cmake)

# set executable path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# set lib path
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
# soft link src/bin---->build/bin
# Warnings: possible bug in windows,
# Possible solution: set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
#if(APPLE OR UNIX)
#    execute_process(COMMAND ln -f -s ${CMAKE_BINARY_DIR}/bin WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
#else()
#    file(COPY ${CMAKE_BINARY_DIR}/bin DESTINATION ${CMAKE_SOURCE_DIR})
#endif()

# mkdir to store log info
execute_process(COMMAND mkdir ${CMAKE_BINARY_DIR}/bin/main_viewport_screenshot)
# Flag logging concerned
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOGGING")
#set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -DLOGTOSTDERR")

# screenshot
execute_process(COMMAND mkdir ${CMAKE_BINARY_DIR}/bin/log)

# set utils
file(GLOB UTILS_SOURCES
    Log/*.cpp
    Timer/*.cpp
)
include_directories(${CMAKE_SOURCE_DIR}/Log)
include_directories(${CMAKE_SOURCE_DIR}/Timer)

# add config.h
include_directories(${CMAKE_SOURCE_DIR}/Config)



# deprecate warnings, specify OpenGL version
# credit: https://blog.csdn.net/u010815558/article/details/90288869
if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

#################################
# Add additional compiler flags #
#################################
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
endif()
if(NOT MSVC_IDE)
    set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "-fPIC -O3 ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++ -Wno-unknown-pragmas")
    # -g for gdb debug, -march=native for memory alignment such that SIMD.
    #SET(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++")
    #SET(CMAKE_CXX_FLAGS "-fPIC -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
endif()

################################
# Specify the libraries to use #
################################
# opencv 3.4.5
set(OpenCV_DIR "/usr/local/opencv345/share/OpenCV")
set(CUDA_HEADERS "${PROJECT_SOURCE_DIR}/3rdparty/cuda-samples/Common")

find_package(OpenCV REQUIRED)#opencv 3.4.5
find_package(Eigen REQUIRED) #eigen 3.3.7
find_package(Ceres REQUIRED) #1.14.0
find_package(OpenGL REQUIRED)#4.6 `glxinfo | grep "OpenGL"`
add_library(GLEW_STATIC STATIC IMPORTED)
find_package(GLEW REQUIRED)
find_package(DBoW3 REQUIRED) #master

include_directories(
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

#######################
# Print libraries info#
#######################
message(STATUS "-------------------------------------------")
message(STATUS "Ceres Library")
message(STATUS "        Dir:${Ceres_DIR}")
message(STATUS "        Version:${Ceres_VERSION}")
message(STATUS "        Include Dir:${Ceres_INCLUDE_DIRS}")
message(STATUS "OpenCV Library")
message(STATUS "        Dir:${OpenCV_DIR}")
message(STATUS "        Version:${OpenCV_VERSION}")
message(STATUS "        Include Dir:${OpenCV_INCLUDE_DIRS}")
#message(STATUS "        libdir:${OpenCV_LIBS}")
message(STATUS "-------------------------------------------")


#############################
# Specify the project files #
#############################
include(${PROJECT_SOURCE_DIR}/IncludeProjectFiles.cmake)
set(THIRD_LIB
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${DBoW3_LIBS}
    ${OPENGL_LIBRARY}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARY}
    )
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${THIRD_LIB})
set(HEADERS ./Plane_detector/Plane_detector.h)

##########################################
# Specify the target and where to put it #
##########################################
include(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

if(WITH_CUDA)
    CUDA_ADD_EXECUTABLE(FUSION main_code.cpp ${UTILS_SOURCES})
else()
    add_executable(FUSION main_code.cpp ${UTILS_SOURCES})
endif()

target_link_libraries(FUSION ${PROJECT_NAME} ${CUDA_LIBRARY} rt cuda)

##########################################
######### test google log ################
##########################################
add_executable(test_glog test_glog.cpp )
target_link_libraries(test_glog ${CERES_LIBRARIES})

## glog and gflags is depenceny of ceres, so just ceres
#find_package(Gflags REQUIRED)
#find_package(Glog REQUIRED)
#include_directories(${GFLAGS_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})

##########################################
######### test cudbscan ################
##########################################
set(SOURCE_CUDBSCAN
    test_cudbscan.cpp
    Timer/Timer.cpp
    cudbscan/DBSCANCPU.hpp
    cudbscan/cuDBSCAN.hpp
    cudbscan/cuDBSCAN.cu)
cuda_add_executable(test_cudbscan ${SOURCE_CUDBSCAN})
target_link_libraries(test_cudbscan ${CUDA_LIBRARIES})
