#pragma once
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "Timer.h"
namespace Log {
static Timer timer;

extern void init(int* pargc, char*** pargv);
extern void shutdown();

#define LOG_INFO(s) LOG(INFO) << "[INFO]    " << s
#define LOG_WARNING(s) LOG(WARNING) << "[WARNING]    " << s
#define LOG_ERROR(s) LOG(ERROR) << "[ERROR]    " << s
#define LOG_FATAL(s) LOG(FATAL) << "[FATAL]    " << s
}  // namespace Log
