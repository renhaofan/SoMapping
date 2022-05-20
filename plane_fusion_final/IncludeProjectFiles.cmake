set(DATA_ENGINE_SOURCES
        ../Data_engine/Data_engine.cpp
        ../Data_engine/Data_loader.cpp
        ../Data_engine/Data_writer.cpp
        )

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
file(GLOB CPU_HEADERS
        Data_engine/*.h
        Environment_Initializer/*.h
        Main_engine/*.h
        Map_engine/*.h
        OurLib/*.h
        Plane_detector/*.h
        Render_engine/*.h
        SLAM_system/*.h
        Track_engine/*.h
        UI_engine/*.h
        Feature_detector/*.h
        Associator/*.h
        Preprocess_engine/*.h
)

file(GLOB CUDA_SOURCES
        Map_engine/*.cu
        Render_engine/*.cu
        Track_engine/*.cu
        Feature_detector/Feature_detector_KernelFunc.cu
        Plane_detector/Plane_detector_KernelFunc.cu
        Preprocess_engine/Preprocess_KernelFunc.cu
)
file(GLOB CUDA_HEADERS
        Map_engine/*.cuh
        Render_engine/*.cuh
        Track_engine/*.cuh
        Feature_detector/Feature_detector_KernelFunc.cuh
        Plane_detector/Plane_detector_KernelFunc.cuh
        Preprocess_engine/Preprocess_KernelFunc.cuh
)

set(sources
        ${CPU_SOURCES}
        ${CUDA_SOURCES}
)

set(headers
        ${CPU_HEADERS}
        ${CUDA_HEADERS}
)

include(cmake/Flags.cmake)
if(WITH_CUDA)
    CUDA_ADD_LIBRARY(${PROJECT_NAME} ${sources} ${headers} ${templates})
else()
    ADD_LIBRARY(${PROJECT_NAME} ${sources} ${headers} ${templates})
endif()
