#include <iostream>
using namespace std;

//#include "ceres/ceres.h"
//#include "ceres/rotation.h"
#include <glog/logging.h>

int main(int argc, char** argv) {
#ifdef LOGGING

  FLAGS_log_dir = "./log";
  google::InitGoogleLogging(argv[0]);
#ifdef LOGTOSTDERR
  FLAGS_logtostderr = true;
#else
  // generate log file
  FLAGS_logtostderr = false;
#endif

  FLAGS_colorlogtostderr = true;           // Set log color
  FLAGS_logbufsecs = 0;                    // Set log output speed(s)
  FLAGS_max_log_size = 1024;               // Set max log file size
  FLAGS_stop_logging_if_full_disk = true;  // If disk is full
  LOG(INFO) << "info new";
  LOG(WARNING) << "warning new";
  LOG(ERROR) << "error new";
  fprintf(stderr, "File %s, Line %d, Function %s(), Unknown dataset\n",
          __FILE__, __LINE__, __FUNCTION__);
  LOG(FATAL) << "File " << __FILE__ << ", Line " << __LINE__ << ", Function "
             << __FUNCTION__ << "()";
  cout << "cout " << endl;
  google::ShutdownGoogleLogging();
#endif
  cout << "logging " << endl;
  return 0;
}
