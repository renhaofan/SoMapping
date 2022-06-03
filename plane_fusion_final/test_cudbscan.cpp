#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "cudbscan/DBSCANCPU.hpp"
//#include "cudbscan/Timer.hpp"
#include "Timer/Timer.h"
#include "cudbscan/cuDBSCAN.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  int N = 20000;
  double eps = 0.1;
  int minClusterSize = 5;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  std::vector<Point2> vec;

  for (int i = 0; i < N; i++) {
    float r1 = static_cast<float>(dis(gen));
    float r2 = static_cast<float>(dis(gen));

    vec.push_back(Point2{r1, r2});
  }

  Timer t;
  double timeCUDA = 0;
  double timeCPU = 0;
  // test GPU
  t.start();
  cuDBSCAN scanCUDA(vec, eps, minClusterSize);
  int nClustersCUDA = scanCUDA.run();
  t.stop();

  timeCUDA = t.getElapsedTime();
  std::cout << "[CUDA] Number of clusters: " << nClustersCUDA
            << "\t Time: " << timeCUDA << std::endl;

  // test CPU
  t.start();
  DBSCANCPU scanCPU(vec, eps, minClusterSize);
  int nCluestersCPU = scanCPU.run();
  t.stop();

  timeCPU = t.getElapsedTime();
  std::cout << "[CPU] Number of clusters: " << nCluestersCPU
            << "\t Time: " << timeCPU << std::endl;
  std::cout << "Speedup was: " << timeCPU / timeCUDA << std::endl;

  // Verify the results
  assert(nClustersCUDA = nCluestersCPU);
  for (int i = 0; i < nClustersCUDA; i++) {
    assert(scanCUDA.clusters[i].size() == scanCPU.clusters[i].size());

    for (size_t j = 0; j < scanCUDA.clusters[i].size(); j++) {
      assert(scanCUDA.clusters[i][j] == scanCPU.clusters[i][j]);
    }
  }
}
