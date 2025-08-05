#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/vector_operation.h>
#include <filesystem>
#include <string>

#include "darcy.h"
#include "darcy_general.h"
#include "npy.hpp"
#include "random_permeability.h"

namespace darcy
{

  // ---------------------- read upstream gradient npy -------------------------
  template <int dim>
  void
  Darcy<dim>::read_upstream_gradient_npy(const std::string &input_file_path)
  {
    dealii::TimerOutput::Scope timing_section(computing_timer,
                                              "read upstream gradient npy");
    // split the input file path into components
    std::filesystem::path my_path(input_file_path);
    std::filesystem::path base_dir = my_path.parent_path();

    const std::string adjoint_file_path =
      base_dir.string() + "/adjoint_data.npy";
    pcout << "Reading adjoint data from: " << adjoint_file_path << std::endl;

    std::vector<unsigned long> shape{};
    bool                       fortran_order;

    npy::LoadArrayFromNumpy(adjoint_file_path,
                            shape,
                            fortran_order,
                            adjoint_data_vec);

    // stucture of one data_vec: [grad_log_lik_y1, grad_log_lik_y2,
    // grad_log_lik_y3]
    int len_vec  = adjoint_data_vec.size();
    int num_data = len_vec / (dim + 1);

    data_vec.resize(num_data, std::vector<double>(dim + 1));
    for (unsigned int k = 0; k < num_data; ++k)
      {
        for (unsigned int i = 0; i < dim + 1; ++i)
          {
            data_vec[k][i] = adjoint_data_vec[i * num_data + k];
          }
      }
  }

  // ---------------------- read primary solution npy --------------------------
  template <int dim>
  void
  Darcy<dim>::read_primary_solution(const std::string &output_path)
  {
    dealii::TimerOutput::Scope timing_section(computing_timer,
                                              "read primary solution npy");
    std::vector<unsigned long> shape{};
    bool                       fortran_order;

    // split the input file path into components
    std::string filename = output_path + "_solution_full.npy";
    pcout << "Reading primary solution from " << filename << std::endl;

    std::vector<double> tmp_primary_solution;
    tmp_primary_solution.resize(
      dof_handler.n_dofs()); // temp vector for primary solution

    npy::LoadArrayFromNumpy(filename,
                            shape,
                            fortran_order,
                            tmp_primary_solution);

    pcout << "Primary solution read successfully!" << std::endl;
    // loop over all dofs on distributed solution vector
    pcout << "Writing primary solution to distributed solution vector..."
          << std::endl;

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        if (solution_primary_problem.in_local_range(i))
          {
            solution_primary_problem[i] = tmp_primary_solution[i];
          }
      }
    pcout
      << "Primary solution successfully written to distributed solution vector"
      << std::endl;
  }

  // ---------------------- overwrite adjoint rhs ------------------------------
  template <int dim>
  void
  Darcy<dim>::overwrite_adjoint_rhs()
  {
    dealii::TimerOutput::Scope timing_section(computing_timer,
                                              "Overwrite adjoint rhs");
    system_rhs         = 0;
    unsigned int x_dim = rf_dof_handler.n_dofs();
    grad_log_lik_x_partial_distributed.resize(x_dim);

    //! unused?
    dealii::FEValuesExtractors::Vector velocities(0);

    const unsigned int dofs_per_cell    = fe.n_dofs_per_cell();
    const unsigned int dofs_per_cell_rf = rf_fe_system.n_dofs_per_cell();

    dealii::MappingQ<dim>                        dummy_mapping(1);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices_rf(
      dofs_per_cell_rf);

    // start the cell loop
    for (const auto &cell_tria : triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(dof_handler);

        // only consider locally owned cells
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            std::vector<unsigned int> data_element_idx;

            // loop over experimental data points to find points on current cell
            for (unsigned int k = 0; k < data_vec.size(); ++k)
              {
                if (cell->point_inside(spatial_coordinates[k]))
                  {
                    data_element_idx.push_back(k);
                  }
              } // end loop experimental data points on current cell

            // loop over experimental data points on current cell
            for (unsigned int idx : data_element_idx)
              {
                // transform physical point to unit cell point
                dealii::Point<dim> current_cell_point =
                  dummy_mapping.transform_real_to_unit_cell(
                    cell, spatial_coordinates[idx]);

                // loop over dofs of random field in current cell
                for (unsigned int j = 0; j < dofs_per_cell_rf; ++j)
                  {
                    // get shape function value at current point for current
                    // dof-shape fun
                    double shape_value_rf =
                      rf_fe_system.shape_value(j, current_cell_point);
                    grad_log_lik_x_partial_distributed
                      [local_dof_indices_rf[j]] +=
                      data_vec[idx][dim] * shape_value_rf;
                  }

                // loop over solution dofs of current cell
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // get shape function value at current point for current
                    // dof-shape fun
                    double shape_value = fe.shape_value(i, current_cell_point);

                    // small hack to filter the correct component --> for
                    // specific dof all but one component should be 0
                    auto   components   = fe.get_nonzero_components(i);
                    double grad_log_lik = 0.0;

                    // loop over components/dim of output filed
                    for (unsigned int comp = 0; comp < components.size();
                         ++comp)
                      {
                        if (components[comp])
                          {
                            grad_log_lik = data_vec[idx][comp];
                            break; // Assuming only one component is non-zero
                                   // per shape function
                          } // end if statement component
                      } // end loop components

                    // write into global rhs (both components for velocity)
                    system_rhs(local_dof_indices[i]) +=
                      -grad_log_lik * shape_value;

                  } // end dof loop
              } // end experimental data loop

          } // end if locally owned
      } // end cell loop
    system_rhs.compress(dealii::VectorOperation::add);
    pcout << "Successfully overwritten rhs..." << std::endl;
  }

  // ---------------------- run method adjoint --------------------------------
  template <int dim>
  void
  Darcy<dim>::run(const std::string &input_path, const std::string &output_path)
  {
    generate_coordinates();
    read_upstream_gradient_npy(input_path);
    run_simulation(input_path, output_path);

    pcout << "Adjoint problem solved successfully!" << std::endl;
    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }

  // ---------------------- run simulation adjoint ----------------------------
  template <int dim>
  void
  Darcy<dim>::run_simulation(const std::string &input_path,
                             const std::string &output_path)
  {
    setup_grid_and_dofs();
    read_input_npy(input_path);
    // generate_ref_input();
    read_primary_solution(output_path); // this needs the dof handler hence
                                        // after setup_grid_and_dofs
    assemble_approx_schur_complement();
    assemble_system(); // we assemble a wrong rhs for the adjoint first and
                       // then overwrite it...
    overwrite_adjoint_rhs();
    solve();
    final_inner_adjoint_product();

    // --- write out the gradient with processor 0
    unsigned int rows    = grad_log_lik_x.size();
    unsigned int columns = 1;

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string filename = output_path + "_grad_solution.npy";
        write_data_to_npy(filename, grad_log_lik_x, rows, columns);
      }
  }

  // ---------------------- final inner adjoint product -----------------------
  template <int dim>
  void
  Darcy<dim>::final_inner_adjoint_product()
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Final inner adjoint product");
    pcout << "Final inner adjoint product with jacobi_k_mat_inv" << std::endl;

    // get tensor function and evaluate it at all dofs
    unsigned int           x_dim = rf_dof_handler.n_dofs();
    dealii::Tensor<2, dim> jacobi_k_mat_inv_value;

    // reinit the final gradient vector
    grad_log_lik_x.resize(x_dim);
    grad_log_lik_x_distributed.resize(x_dim);

    // quadrature formula, fe values and dofs
    // standard gauss quadrature
    const dealii::QGauss<dim> quadrature(
      degree_u + 1); // we choose a coarser quadrature here
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature,
                                    dealii::update_quadrature_points |
                                      dealii::update_values |
                                      dealii::update_JxW_values);
    dealii::FEValues<dim> fe_rf_values(rf_fe_system,
                                       quadrature,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);
    const unsigned int    n_q_points = quadrature.size();
    // other stuff
    dealii::FEValuesExtractors::Vector velocities(0);
    const unsigned int                 dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int rf_dofs_per_cell = rf_fe_system.n_dofs_per_cell();

    // local to global mapping for random field
    std::vector<dealii::types::global_dof_index> local_rf_dof_indices(
      rf_dofs_per_cell);

    // local to global dof mapping for solution
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);
    solution_distributed         = solution;
    solution_primary_distributed = solution_primary_problem;

    // instantiate some variables
    std::vector<dealii::Point<dim>> q_points(n_q_points);
    double                          JxW_q;

    // start the cell loop
    for (const auto &cell_tria : triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(rf_dof_handler);

        // only consider locally owned cells
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_rf_values.reinit(rf_cell);
            cell->get_dof_indices(local_dof_indices);
            q_points = fe_values.get_quadrature_points();
            std::vector<double> solution_local(dofs_per_cell);
            std::vector<double> solution_primary_local(dofs_per_cell);
            rf_cell->get_dof_indices(local_rf_dof_indices);

            // calculate k-th jacobi matrix
            std::vector<double> rf_value(n_q_points);
            fe_rf_values.get_function_values(x_vec_distributed, rf_value);

            // get local solutions once before the quadrature loop
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                solution_local[i] = solution_distributed[local_dof_indices[i]];
                solution_primary_local[i] =
                  solution_primary_distributed[local_dof_indices[i]];
              }
            // ------ loop over quadrature points -------------
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                JxW_q = fe_values.JxW(q);

                //  ----- precompute fe_values stuff
                std::vector<dealii::Tensor<1, dim>> fe_values_velocities_vec(
                  dofs_per_cell);
                for (unsigned int ii = 0; ii < dofs_per_cell; ++ii)
                  {
                    fe_values_velocities_vec[ii] =
                      fe_values[velocities].value(ii, q);
                  }

                // ------- loop over random field dofs per cell ------
                for (unsigned int k = 0; k < rf_dofs_per_cell; ++k)
                  {
                    RandomMedium::get_jacobi_inv_kmat(
                      rf_value[q],
                      fe_rf_values.shape_value(k, q),
                      jacobi_k_mat_inv_value);

                    // ------ outer loop over all dofs of the cell -----------
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const dealii::Tensor<1, dim> phi_i_u =
                          fe_values_velocities_vec[i];
                        double local_solution_i_JxW = solution_local[i] * JxW_q;
                        const auto pre_mult = phi_i_u * jacobi_k_mat_inv_value;

                        // ----- inner loop over all dofs of the cell ---------
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          {
                            const dealii::Tensor<1, dim> phi_j_u =
                              fe_values_velocities_vec[j];
                            double local_primary_j = solution_primary_local[j];

                            // compute inner product and add to global vector
                            grad_log_lik_x_distributed
                              [local_rf_dof_indices[k]] +=
                              local_solution_i_JxW * (pre_mult * phi_j_u) *
                              local_primary_j;
                          } // inner end loop dofs per cell

                      } // end outer dof loop
                  } // end loop over random field dofs

              } // end loop quadrature points

          } // end if cell is locally owned
      } // end loop cell
    pcout << "grad_log_x (distributed) successfully assembled!" << std::endl;

    dealii::Utilities::MPI::sum(grad_log_lik_x_distributed,
                                MPI_COMM_WORLD,
                                grad_log_lik_x);
    pcout << "Successfully summed grad_log_lik_x over processors" << std::endl;
  }

} // end namespace darcy

// ---------------------- main function adjoint -----------------------------
int
main(int argc, char *argv[])
{
  std::string input_file_path  = argv[1];
  std::string output_file_path = argv[2];

  try
    {
      using namespace darcy;
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);
      const unsigned int                       fe_degree = 1;
      Darcy<2>                                 mixed_laplace_problem(fe_degree);
      mixed_laplace_problem.run(input_file_path, output_file_path);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
