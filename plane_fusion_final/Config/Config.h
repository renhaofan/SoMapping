/**
 *  Copyright (C) All rights reserved.
 *  @file Config.h
 *  @brief Define some parameters and macros.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @todo void init();
 *  @todo void check();  check confilct and valid values, include only one slam
 * system set up true.
 *  @warning Current ctor, load json file once. need to finish function init().
 */

#pragma once

#include "json.hpp"
using json = nlohmann::json;

#include "Log.h"

// file operation
#include <dirent.h>
#ifdef _WIN32
#include <io.h>
#elif __linux__
#include <inttypes.h>
#include <unistd.h>
#define __int64 int64_t
#define _close close
#define _read read
#define _lseek64 lseek64
#define _O_RDONLY O_RDONLY
#define _open open
#define _lseeki64 lseek64
#define _lseek lseek
#define stricmp strcasecmp
#endif
#include <fstream>
#include <string>
#include <vector>

/**
 *  @brief Maximum number of hierarchy layers.
 *  @details Used directly by Hierarchy_image.h, SLAM_system_settings.h
 */
#define MAX_LAYER_NUMBER 8

/**
 *  @brief Flag whether enable SLAM debug code for
 *  @details Used directly by SLAM_system_settings.h
 */
#define COMPILE_DEBUG_CODE

class JSON_CONFIG {
 public:
  static JSON_CONFIG* instance_ptr;
  static JSON_CONFIG* instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new JSON_CONFIG();
    // check whether allocte memory succesfully.
    if (instance_ptr == nullptr) {
#ifdef LOGGING
      LOG_FATAL("Failed to allocate JSON CONFIG memory!");
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(): "
              "Failed to allocate JSON CONFIG memory.\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw "Failed to allocate JSON CONFIG memory!";
    }
    return instance_ptr;
  }
  json j;

  void init();
  void check();

  JSON_CONFIG();
  ~JSON_CONFIG();
};
