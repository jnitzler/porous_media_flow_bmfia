#ifndef DARCY_H
#define DARCY_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>
#include <string>
#include <vector>

// ----- Define the class structure ------------------------------------------
namespace darcy
{

  // ----------- Darcy class ---------------------------
  template <int dim>
  class Darcy
  {
  public:
    explicit Darcy(dealii::ParameterHandler &prm);

    void
    run();

  private:
    void
    generate_coordinates();
    void
    run_simulation();
    void
    read_input_npy(const std::string &input_path);
    void
    generate_ref_input();
    void
    read_primary_solution(const std::string &input_path);
    void
    setup_grid_and_dofs();
    void
    assemble_approx_schur_complement();
    void
    assemble_system();
    void
    solve();
    void
    output_pvtu(const std::string &output_path) const;
    void
    output_full_velocity_npy(const std::string &output_path);
    void
    output_velocity_at_observation_points_npy(const std::string &output_path);
    void
    output_field_at_observation_points_npy(const std::string &output_path);

    void
    write_data_to_npy(const std::string   &file_path,
                      std::vector<double> &data,
                      unsigned int         rows,
                      unsigned int         columns) const;
    void
    read_upstream_gradient_npy(const std::string &input_file_path);
    void
    overwrite_adjoint_rhs();
    void
    final_inner_adjoint_product();

    unsigned int degree_p; // degree of pressure space
    unsigned int degree_u; // degree of velocity space
    unsigned int n_refinements;
    std::string  npy_input_file_path;
    std::string  output_file_prefix;

    dealii::ParameterHandler &prm;

    // Triangulation<dim> triangulation;
    dealii::parallel::distributed::Triangulation<dim> triangulation;
    std::unique_ptr<dealii::FESystem<dim>>            fe;
    dealii::DoFHandler<dim>                           dof_handler;

    // random field setup
    dealii::FESystem<dim>   rf_fe_system;
    dealii::DoFHandler<dim> rf_dof_handler;

    dealii::AffineConstraints<double> constraints;
    dealii::AffineConstraints<double> preconditioner_constraints;

    dealii::TrilinosWrappers::BlockSparseMatrix system_matrix;
    dealii::TrilinosWrappers::BlockSparseMatrix precondition_matrix;

    dealii::TrilinosWrappers::MPI::BlockVector solution;
    dealii::TrilinosWrappers::MPI::BlockVector system_rhs;
    dealii::TrilinosWrappers::MPI::BlockVector temp_vec;
    dealii::TrilinosWrappers::MPI::BlockVector solution_distributed;

    dealii::TrilinosWrappers::BlockSparseMatrix block_mass_matrix;
    std::vector<double>                         grad_log_lik_x_distributed;
    std::vector<double>                         grad_log_lik_x;
    std::vector<double> grad_log_lik_x_partial_distributed;

    dealii::FullMatrix<double>
      grad_pde_x; // New PDE gradient that needs to be implemented

    // random field stuff
    dealii::TrilinosWrappers::MPI::Vector
      x_vec; // the input vector for the simulation
    dealii::TrilinosWrappers::MPI::Vector
      x_vec_distributed; // the input vector for the simulation

    // Vector<double> random_field_vec;
    std::vector<dealii::Point<dim>> spatial_coordinates;
    dealii::Point<dim>              shift_point;
    std::vector<double>             output_data;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer; // timer for sub programs
    std::vector<double> adjoint_data_vec; // adjoint data vector with partial
                                          // derivative of likelihood
    std::vector<std::vector<double>>
      data_vec; // general input data which will be split to actual data
    dealii::TrilinosWrappers::MPI::BlockVector
      solution_primary_problem; // distributed solution vector of the primary
                                // problem
    dealii::TrilinosWrappers::MPI::BlockVector solution_primary_distributed;

    void
    setup_system_matrix(
      const std::vector<dealii::IndexSet> &system_partitioning,
      const std::vector<dealii::IndexSet> &system_relevant_partitioning);

    void
    setup_approx_schur_complement(
      const std::vector<dealii::IndexSet> &system_partitioning,
      const std::vector<dealii::IndexSet> &system_relevant_partitioning);
  };
} // namespace darcy
#endif