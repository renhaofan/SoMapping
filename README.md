# SoMapping
## clang-format
command clang-format for folder
```bash
find plane_fusion_final -iname *.h -o -iname *.cpp | xargs clang-format -i -style=Google
```
```bash
find plane_fusion_final -iname *.cuh -o -iname *.cu | xargs clang-format -i -style=Google
```
## Dependency
used eigen 3.3.7, default version ubuntu20.04.
### ceres-solver 1.14.0
```bash
$ git clone -b 1.14.0 git@github.com:ceres-solver/ceres-solver.git
$ cd ceres-solver
$ mkdir build
$ cd build
$ cmake ..
```

```bash
$ git clone git@github.com:rmsalinas/DBow3.git
$ cmake \
-D OpenCV_DIR=/usr/local/opencv345/share/OpenCV \
-D USE_CONTRIB=ON \
..
```

```bash
$ git clone -b v11.6 git@github.com:NVIDIA/cuda-samples.git
```