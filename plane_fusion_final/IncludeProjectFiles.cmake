include_directories(${CMAKE_SOURCE_DIR}/Data_engine)
include_directories(${CMAKE_SOURCE_DIR}/Environment_Initializer)
include_directories(${CMAKE_SOURCE_DIR}/Main_engine)
include_directories(${CMAKE_SOURCE_DIR}/OurLib)
include_directories(${CMAKE_SOURCE_DIR}/Plane_detector)
include_directories(${CMAKE_SOURCE_DIR}/Render_engine)
include_directories(${CMAKE_SOURCE_DIR}/SLAM_system)
include_directories(${CMAKE_SOURCE_DIR}/Track_engine)
include_directories(${CMAKE_SOURCE_DIR}/UI_engine)
include_directories(${CMAKE_SOURCE_DIR}/Feature_detector)
include_directories(${CMAKE_SOURCE_DIR}/Associator)
include_directories(${CMAKE_SOURCE_DIR}/Preprocess_engine)
include_directories(${CMAKE_SOURCE_DIR}/Map_engine)
include(cmake/Flags.cmake)

file(GLOB CPU_SOURCES
        Data_engine/*.cpp
        Environment_Initializer/*.cpp
        Main_engine/*.cpp
        Map_engine/*.cpp
        OurLib/*.cpp
        Plane_detector/*.cpp
        Render_engine/*.cpp
        SLAM_system/*.cpp
        Track_engine/*.cpp
        UI_engine/*.cpp
        Feature_detector/*.cpp
        Associator/*.cpp
        Preprocess_engine/*.cpp
)


file(GLOB CUDA_SOURCES
        Map_engine/*.cu
        Render_engine/*.cu
        Track_engine/*.cu
        Feature_detector/Feature_detector_KernelFunc.cu
        Plane_detector/Plane_detector_KernelFunc.cu
        Preprocess_engine/Preprocess_KernelFunc.cu
)

#file(GLOB CUDA_HEADERS
#        Map_engine/*.cuh
#        Render_engine/*.cuh
#        Track_engine/*.cuh
#        Feature_detector/Feature_detector_KernelFunc.cuh
#        Plane_detector/Plane_detector_KernelFunc.cuh
#        Preprocess_engine/Preprocess_KernelFunc.cuh
#)

set(sources
        ${CPU_SOURCES}
        ${CUDA_SOURCES}
)

if(WITH_CUDA)
    CUDA_ADD_LIBRARY(${PROJECT_NAME} ${sources} ${templates})
else()
    ADD_LIBRARY(${PROJECT_NAME} ${sources} ${templates})
endif()
