/**
 *  @file Main_engine.h
 *  @brief class Main_engine header, Singleton object contains
 *         environment_initializer, data_engine, SLAM_system, render_engine.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include "Data_engine/Data_engine.h"
#include "Environment_Initializer/Environment_Initializer.h"
#include "SLAM_system/SLAM_system.h"
#include "UI_engine/UI_engine.h"

/**
 * @brief Singleton object,
 */
class Main_engine {
 public:
  /** @brief The pointer to this static object. */
  static Main_engine *instance_ptr;
  /** @brief Member function for instantiating this static object. */
  static Main_engine *instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new Main_engine();
    // check whether allocte memory succesfully.
    if (instance_ptr == nullptr) {
#ifdef LOGGING
      LOG_FATAL("Failed to allocate main engine memory!");
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(): "
              "Failed to allocate main engine memory.\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw "Failed to allocate main engine memory!";
    }
    return instance_ptr;
  }

  /** @brief Environment initializer. */
  Environment_Initializer *environment_initializer;
  /** @brief Data loader/writer. */
  Data_engine *data_engine;
  /** @brief SLAM system. */
  SLAM_system *SLAM_system_ptr;
  /** @brief Render engine. */
  Render_engine *render_engine_ptr;

  /** @brief Default constructor. */
  Main_engine();
  /** @brief Default deconstructor. */
  ~Main_engine();

  /**
   * @brief Initiate environment, data_engine, SLAM system,
   *        render_engine, UI_engine::instance()->init;
   * @param argc Input the first argument of main() function.
   * @param argv Input the second argument of main() function.
   * @param image_loader_ptr The pointer of Image_loader.
   * @param output_folder Output folder.(output estimated trajectory, mesh,
   *        etc.)
   * @param _is_ICL_NUIM_dataset Dataset coordinate mode.
   */
  void init(int argc, char **argv, Image_loader *image_loader_ptr,
            string output_folder = "./", bool _is_ICL_NUIM_dataset = false);
  /**
   * @brief Alias UI_engine::instance()->run().
   */
  void run();
};
