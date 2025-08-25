#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/parameter_handler.h>

class ParameterReader : public dealii::EnableObserverPointer
{
public:
  ParameterReader(dealii::ParameterHandler &prm);

  void
  read_parameters(const std::string &prm_file);

private:
  void
  declare_parameters();

  dealii::ParameterHandler &prm;
};

#endif