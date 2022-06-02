/**
 *  Copyright (C) All rights reserved.
 *  @file Environment_Initializer.h
 *  @brief class Environment_Initializer header.
 *         Check and print GPU device & CUDA infomation.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

class Environment_Initializer {
 public:
  /** @brief Max threads per CUDA block. */
  int max_TperB;
  /** @brief GPU clock frequency. */
  int GPU_clock_rate;

  /** @brief Default constructor. */
  Environment_Initializer();
  /** @brief Constructor with information print flag. */
  Environment_Initializer(bool print_detail);
  /** @brief Default deconstructor. */
  ~Environment_Initializer();

  /**
   * @brief Check GPU device, CUDA environment. And start logging.
   * @param argc Input the frist argument of main() function.
   * @param argv Input the second argument of main() function.
   * @todo if success, print `GPU Device 0: "Maxwell" with compute
   capability 5.0` err = cudaGetDeviceProperties(&deviceProps, devID); record as
   log, not print in stderr.
   */
  void init_environment(int argc, char **argv);

 private:
  /** @brief Flag about whether print detail infomation */
  bool print_detail_information = false;
};
