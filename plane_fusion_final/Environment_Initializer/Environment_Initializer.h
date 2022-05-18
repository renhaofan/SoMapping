#pragma once

//! Initiate the environment.
/*!

*/
class Environment_Initializer {
 public:
  //! Max threads per CUDA block.
  int max_TperB;
  //! GPU clock frequency.
  int GPU_clock_rate;

  //
  Environment_Initializer();
  ~Environment_Initializer();
  //! Constructor with information print flag
  Environment_Initializer(bool print_detail);

  //! Initiate environment.
  /*!
          \param	argc	Input the frist argument of main function.

          \param	argv	Input the second argument of main function.

          \return	void
  */
  void init_environment(int argc, char **argv);

 private:
  //!
  bool print_detail_informations = false;
};
