# SoMapping
## 1. clang-format
command clang-format for folder
```bash
find plane_fusion_final -iname *.h -o -iname *.cpp | xargs clang-format -i -style=Google
```
```bash
find plane_fusion_final -iname *.cuh -o -iname *.cu | xargs clang-format -i -style=Google
```
## 2. environment
Ubuntu20.04 Desktop,  CUDA Driver Version = 11.6, CUDA Runtime Version = 11.6, cudnn = 8.3.1.
```bash
$ g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
### 2.1 Dependency
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
#### 2. ceres-solver 1.14.0
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
#### 3. OpenCV 3.4.5 
OpenCV3.x should work(not fully test).
#### 4. DBow3
If you have installed the contrib_modules, use cmake option `-DUSE_CONTRIB=ON` to enable SURF.
```bash
$ git clone git@github.com:rmsalinas/DBow3.git
$ cmake \
-D OpenCV_DIR=/usr/local/opencv345/share/OpenCV \
-D USE_CONTRIB=ON \
..
```
#### 5. CUDA utils

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