#include "Config.h"

JSON_CONFIG* JSON_CONFIG::instance_ptr = nullptr;

void JSON_CONFIG::init(const std::string& s) {
  std::ifstream jfile(s);
  if (!jfile.is_open()) {
#ifdef LOGGING
    LOG_WARNING("Failed to open json files, using default value.");
#endif
  }
  jfile >> j;
  jfile.close();
  if (jfile.is_open()) {
#ifdef LOGGING
    LOG_ERROR("Failed to close json files.");
#endif
  }
}

JSON_CONFIG::JSON_CONFIG() { init(); }

JSON_CONFIG::~JSON_CONFIG() {}
