#pragma once
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <string>

#include "Timer.h"
namespace Log {
static Timer timer;

extern void init(int* pargc, char*** pargv);
extern void shutdown();

/**
 * LOG_TIC(H)
 * LOG_TOC(H)
 */

#define LOG_TIC(NAME) \
std::chrono::steady_clock::time_point __TIMER##NAME##_START = std::chrono::steady_clock::now();


#define LOG_TOC(NAME) \
std::chrono::steady_clock::time_point __TIMER##NAME##_END = std::chrono::steady_clock::now(); \
{ \
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>( __TIMER##NAME##_END - __TIMER##NAME##_START ).count(); \
        LOG_INFO_IV("Timer ", #NAME, ": ", (d) * 1000, "ms"); \
}

#define LOG_INFO(s) LOG(INFO) << "[INFO]    " << s;
#define LOG_WARNING(s) LOG(WARNING) << "[WARNING]    " << s;
#define LOG_ERROR(s) LOG(ERROR) << "[ERROR]    " << s;
#define LOG_FATAL(s) LOG(FATAL) << "[FATAL]    " << s;

#define LOG_INFO_I(a, b) LOG(INFO) << "[INFO]    " << a << b;
#define LOG_WARNING_I(a, b) LOG(WARNING) << "[WARNING]    " << a << b;
#define LOG_ERROR_I(a, b) LOG(ERROR) << "[ERROR]    " << a << b;
#define LOG_FATAL_I(a, b) LOG(FATAL) << "[FATAL]    " << a << b;

#define LOG_INFO_II(a, b, c) LOG(INFO) << "[INFO]    " << a << b << c;
#define LOG_WARNING_II(a, b, c) LOG(WARNING) << "[WARNING]    " << a << b << c;
#define LOG_ERROR_II(a, b, c) LOG(ERROR) << "[ERROR]    " << a << b << c;
#define LOG_FATAL_II(a, b, c) LOG(FATAL) << "[FATAL]    " << a << b << c;

#define LOG_INFO_III(a, b, c, d) LOG(INFO) << "[INFO]    " << a << b << c << d;
#define LOG_WARNING_III(a, b, c, d) \
LOG(WARNING) << "[WARNING]    " << a << b << c << d;
#define LOG_ERROR_III(a, b, c, d) \
LOG(ERROR) << "[ERROR]    " << a << b << c << d;
#define LOG_FATAL_III(a, b, c, d) \
LOG(FATAL) << "[FATAL]    " << a << b << c << d;

#define LOG_INFO_IV(a, b, c, d, e) \
LOG(INFO) << "[INFO]    " << a << b << c << d << e;
#define LOG_WARNING_IV(a, b, c, d, e) \
LOG(WARNING) << "[WARNING]    " << a << b << c << d << e;
#define LOG_ERROR_IV(a, b, c, d, e) \
LOG(ERROR) << "[ERROR]    " << a << b << c << d << e;
#define LOG_FATAL_IV(a, b, c, d, e) \
LOG(FATAL) << "[FATAL]    " << a << b << c << d << e;

}  // namespace Log
