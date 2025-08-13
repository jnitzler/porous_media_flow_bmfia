#include "./parameter_reader.h"

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

ParameterReader::ParameterReader(dealii::ParameterHandler &prm)
  : prm(prm)
{}

void
ParameterReader::declare_parameters()
{
  prm.enter_subsection("File input and output");
  {
    prm.declare_entry("Numpy input file name",
                      "input",
                      dealii::Patterns::Anything(),
                      "Name of the numpy input file (with extension)");

    prm.declare_entry("Output file name",
                      "solution",
                      dealii::Patterns::Anything(),
                      "Name of the output file (without extension)");

    // dealii::DataOutInterface<1>::declare_parameters(prm);
  }
  prm.leave_subsection();

  prm.enter_subsection("Mesh and geometry parameters");
  {
    prm.declare_entry("Number of refinements",
                      "0",
                      dealii::Patterns::Integer(0),
                      "Number of global mesh refinement steps");
  }
  prm.leave_subsection();

  prm.enter_subsection("Finite element parameters");
  {
    prm.declare_entry("Pressure polynomial degree",
                      "1",
                      dealii::Patterns::Integer(1),
                      "Degree of the polynomial approximation for pressure");
  }
  prm.leave_subsection();
}

void
ParameterReader::read_parameters(const std::string &prm_file)
{
  declare_parameters();

  prm.parse_input(prm_file);
}
