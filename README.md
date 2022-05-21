# SoMapping
## BUG
```
CUDA error at /home/steve/code/mycode/SoMapping/plane_fusion_final/Preprocess_engine/Preprocess_engine.cpp:298 code=702(cudaErrorLaunchTimeout) "cudaMemcpy(this->dev_raw_depth, raw_depth.data, this->raw_depth_size.width * this->raw_depth_size.height * sizeof(RawDepthType), cudaMemcpyHostToDevice)" 
```
```

CUDA error at /home/steve/code/mycode/SoMapping/plane_fusion_final/Map_engine/Mesh_generator.cpp:67 code=2(cudaErrorMemoryAllocation) "cudaMalloc((void **)&(this->dev_planar_triangles), this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f))" 

```

### wrong calibration file will cause this BUG in offline dataset mode.
```
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(3.4.5) /home/steve/Downloads/Source-Archive-main/OpenCV/opencv-3.4.5/modules/video/src/lkpyramid.cpp:1231: error: (-215:Assertion failed) (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 in function 'calc'

```
### Fast UI_switch key 4 and 8
## 1. clang-format
command clang-format for folder
```bash
find plane_fusion_final -iname *.h -o -iname *.cpp | xargs clang-format -i -style=Google
```
```bash
find plane_fusion_final -iname *.cuh -o -iname *.cu | xargs clang-format -i -style=Google
```
## 2. GUI-Interaction
<kdb>Esc</kdb> exit program.
### 2.1 Display
<kbd>1</kbd> - OpenGL right-hand frame, red-x, green-y, blue-z.

<kbd>2</kbd> - Trajectory, blue-?, red-?.

<kbd>3</kbd> - Voxel, yellow color by default.

<kbd>4</kbd> - Point cloud per frame, yellow color by default.

<kbd>5</kbd> - Normal vectors of point cloud.

<kbd>6</kbd> - TSDF map.

<kbd>8</kbd> - Pesudo color point cloud per frame.

<kbd>9</kbd> - Plane detection-? blue color by default.

<kbd>0</kbd> - Planar supervoxel-?


### 2.2 FreeView Control

<kbd>w</kbd> - Move forward. 

<kbd>s</kbd> - Move backward.

<kbd>a</kbd> - Shift Left.

<kbd>d</kbd> - Shift Right.

<kbd>e</kbd> - Move forward-?

<kbd>q</kbd> - Move backward-?

<kbd>←</kbd>, <kbd>→</kbd> yaw slowly, aka rotate around self-y axis.

<kbd>↑</kbd>, <kbd>↓</kbd> pitch slowly, aka rotate around self-x axis.

mouse left button: rotate fast.

mouse right button: translate againist z axis.

## 3. environment
Ubuntu20.04 Desktop,  CUDA Driver Version = 11.6, CUDA Runtime Version = 11.6, cudnn = 8.3.1.
```bash
$ g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
### Dependency
#### 1. OpenGL
```bash
sudo apt install -y libglu1-mesa-dev mesa-common-dev mesa-utils
sudo apt install -y freeglut3-dev libglm-dev libassimp-dev
sudo apt install -y libglfw3 libglfw3-dev libglfw3-doc
```
#### 2. eigen 3.3.7
Default version on ubuntu20.04 ppa.
```
sudo apt-get install -y libeigen3-dev libeigen3-doc
```
#### 3. ceres-solver 1.14.0
Install dependencies firstly.
```
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse (optional)
sudo apt-get install libsuitesparse-dev
```

Install ceres.
```bash
$ git clone -b 1.14.0 git@github.com:ceres-solver/ceres-solver.git
$ cd ceres-solver
$ mkdir build
$ cd build
$ cmake ..
$ make -j2
$ sudo make install
```
#### 4. OpenCV 3.4.5 
OpenCV3.x should work(not fully test).
#### 5. DBow3
If you have installed the contrib_modules, use cmake option `-DUSE_CONTRIB=ON` to enable SURF.
```bash
$ git clone git@github.com:rmsalinas/DBow3.git
$ cmake \
-D OpenCV_DIR=/usr/local/opencv345/share/OpenCV \
-D USE_CONTRIB=ON \
..
```
#### 6. CUDA utils

```bash
$ git clone -b v11.6 git@github.com:NVIDIA/cuda-samples.git
```

# 3. Demo
Download the tum dataset and tools.
```bash
cd /***/rgbd_dataset_freiburg1_xyz
python associate.py rgb.txt depth.txt > associate.txt
```
For convinence, download the packed demo dataset.

[Google Drive](https://drive.google.com/file/d/1mKQMt5lAL81yTE4n24guziDD8ig69K2Z/view?usp=sharing)
| 
[Baidu disk](https://pan.baidu.com/s/1MHVoY4y3URaP4SMSIn9e6w)

After build the project, you should find binary executable file `FUSION` located in `/*/SoMapping/plane_fusion_final/bin`
```bash
cd bin
./FUSION /*****/Files/TUM/TUM1.txt /******/rgbd_dataset_freiburg1_xyz 1
```