#include "Log.h"

namespace Log {
void init(int* pargc, char*** pargv) {
  //    FLAGS_log_dir = "./log";
  std::string log_dir = "log";
  FLAGS_log_dir = "./" + log_dir;
  google::InitGoogleLogging(*(pargv)[0]);
  std::cout << "Log file located in "
            << "EXECUTABLE_OUTPUT_PATH" << std::endl;
  ;
#ifdef LOGTOSTDERR
  FLAGS_alsologtostderr = true;
#endif
  FLAGS_colorlogtostderr = true;  // set log color if termimal supports.
  FLAGS_logbufsecs = 0;           // set log output speed(s)
  FLAGS_max_log_size = 16;        // set max log file size(MB)
  FLAGS_stop_logging_if_full_disk = true;  // If disk if full

  LOG(INFO) << "[INFO]     argc :" << *pargc;
  LOG(INFO) << "[INFO]     argvs: ";
  for (int i = 0; i < (*pargc); ++i) {
    LOG(INFO) << "[INFO]     argv[" << i << "]: " << (*(pargv))[i];
  }
}

void shutdown() { google::ShutdownGoogleLogging(); }
}  // namespace Log
