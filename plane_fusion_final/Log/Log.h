#pragma once
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "Timer.h"
namespace Log {
static Timer timer;

extern void init(int* pargc, char*** pargv);
extern void shutdown();

#define LOG_INFO(s) LOG(INFO) << "[INFO]    " << s;
#define LOG_WARNING(s) LOG(WARNING) << "[WARNING]    " << s;
#define LOG_ERROR(s) LOG(ERROR) << "[ERROR]    " << s;
#define LOG_FATAL(s) LOG(FATAL) << "[FATAL]    " << s;

#define LOG_INFO_I(a, b) LOG(INFO) << "[INFO]    " << a << b;
#define LOG_WARNING_W(a, b) LOG(WARNING) << "[WARNING]    " << a << b;
#define LOG_ERROR_E(a, b) LOG(ERROR) << "[ERROR]    " << a << b;
#define LOG_FATAL_F(a, b) LOG(FATAL) << "[FATAL]    " << a << b;

#define LOG_INFO_II(a, b, c) LOG(INFO) << "[INFO]    " << a << b << c;
#define LOG_WARNING_WW(a, b, c) LOG(WARNING) << "[WARNING]    " << a << b << c;
#define LOG_ERROR_EE(a, b, c) LOG(ERROR) << "[ERROR]    " << a << b << c;
#define LOG_FATAL_FF(a, b, c) LOG(FATAL) << "[FATAL]    " << a << b << c;

#define LOG_INFO_III(a, b, c, d) LOG(INFO) << "[INFO]    " << a << b << c << d;
#define LOG_WARNING_WWW(a, b, c, d) LOG(WARNING) << "[WARNING]    " << a << b << c << d;
#define LOG_ERROR_EEE(a, b, c, d) LOG(ERROR) << "[ERROR]    " << a << b << c << d;
#define LOG_FATAL_FFF(a, b, c, d) LOG(FATAL) << "[FATAL]    " << a << b << c << d;

}  // namespace Log
