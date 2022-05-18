CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(PlaneFusion)


SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)


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

FIND_PACKAGE(OpenCV REQUIRED) #opencv 3.0.0
FIND_PACKAGE(Eigen REQUIRED) #eigen 3.2
FIND_PACKAGE(Ceres REQUIRED) #1.14.0
FIND_PACKAGE(OpenGL REQUIRED)#2.1.0
add_library(glew_static STATIC IMPORTED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(DBoW3 REQUIRED)
INCLUDE_DIRECTORIES(/usr/local/include/DBoW3)
include_directories(/usr/local/include/DBoW3)
#include_directories(/usr/include/eigen3)
#include_directories(${EIGEN3_INCLUDE_DIR})
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake) #9.0
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseGLUT.cmake) #2.8.0
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseGLEW.cmake)

INCLUDE_DIRECTORIES(
        ${GLEW_INCLUDE_DIRS}
        /usr/local/cuda/samples/common/inc
        ${CMAKE_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenGL_INCLUDE_DIRS}
        ${DBoW3_INCLUDE_DIRS}
)

#############################
# Specify the project files #
#############################
#add_executable(${PROJECT_NAME} ./Plane_detector/Plane_detector.h
#        )
INCLUDE(${PROJECT_SOURCE_DIR}/IncludeProjectFiles.cmake)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${OPENGL_LIBRARY} ${GLUT_LIBRARY} ${GLEW_LIBRARY} ${CUDA_LIBRARIES} ${DBoW3_LIBS})
set(HEADERS
        ./Plane_detector/Plane_detector.h)
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


#IF(MSVC_IDE)
#    SET_TARGET_PROPERTIES(${targetname} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG")
#ENDIF()
