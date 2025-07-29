#include "darcy.h"

#include <cstddef>

#include "darcy_general.h"

namespace darcy // same namespace and in header file
{
  // --- output velocity at observation points to numpy array ---------
  template <int dim>
  void
  Darcy<dim>::output_velocity_at_observation_points_npy(
    const std::string &output_path)
  {
    // collect data of interest to write out in output_data
    // set up a dummy mapping
    MappingQ<dim> dummy_mapping(1);

    //  // get the solution values at the points of interest for time step
    solution_distributed = solution; // take care of the distributed solution
                                     //
    Utilities::MPI::RemotePointEvaluation<dim> evaluation_cache(1.0e-9);
    const auto data_array = VectorTools::point_values<dim>(dummy_mapping,
                                                           dof_handler,
                                                           solution_distributed,
                                                           spatial_coordinates,
                                                           evaluation_cache);

    unsigned int num_dofs_total = dof_handler.n_dofs();
    unsigned int num_vel_vecs   = data_array.size();

    // append them to the data vector
    std::vector<double> l_output_data(dim * num_vel_vecs);

    // loop over experimental data point locations and their values
    // resulting output format: row vector with first u_1 block then u_2 block
    for (unsigned int j = 0, k = 0; j < dim; ++j)
      {
        for (unsigned int i = 0; i < num_vel_vecs; ++i, ++k)
          {
            l_output_data[k] = data_array[i][j];
          }
      }


    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string  filename = output_path + "_sol.npy";
        const unsigned int num_data = l_output_data.size();
        write_data_to_npy(filename, l_output_data, num_data, 1);
      }
  }


  // --------- output field at observation points to numpy array
  // --- below: x-feature
  template <int dim>
  void
  Darcy<dim>::output_field_at_observation_points_npy(
    const std::string &output_path)
  {
    // collect data of interest to write out in output_data
    // set up a dummy mapping
    MappingQ<dim> dummy_mapping(1);

    x_vec_distributed = x_vec;

    // set up a cache / remote evaluation object
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_eval_obj;
    remote_eval_obj.reinit(spatial_coordinates, triangulation, dummy_mapping);

    const auto data_array = VectorTools::point_values<1>(remote_eval_obj,
                                                         rf_dof_handler,
                                                         x_vec_distributed);

    unsigned int num_x_vec = data_array.size();

    // loop over experimental data point locations and their values
    std::vector<double> feature_vec(num_x_vec);
    for (unsigned int i = 0; i < num_x_vec; i++)
      {
        feature_vec[i] = data_array[i];
      }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string  filename = output_path + "_features.npy";
        const unsigned int num_data = feature_vec.size();
        write_data_to_npy(filename, feature_vec, num_data, 1);
      }
  }



  // ----------------- output_npy -----------------
  template <int dim>
  void
  Darcy<dim>::output_full_velocity_npy(const std::string &output_path)
  {
    std::vector<double> full_solution;
    solution_distributed = solution;

    // only resize full solution vector on rank 0 to save memory
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        full_solution.resize(dof_handler.n_dofs(), 0.0);
      }

    auto custom_fuse = [&](const double &x, const double &y) { return x + y; };

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        double temp;
        temp = Utilities::MPI::reduce<double>(solution.in_local_range(i) ?
                                                solution[i] :
                                                0.0,
                                              MPI_COMM_WORLD,
                                              custom_fuse,
                                              0);

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            full_solution[i] = temp;
          }
      }

    // write the gathered solution that exists on rank 0 to one file
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::string filename = output_path + "_solution_full.npy";
        pcout << "Writing full solution to file: " << filename << std::endl;
        pcout << "Size of full solution: " << full_solution.size() << std::endl;
        pcout << "Number of dofs: " << dof_handler.n_dofs() << std::endl;
        // Assuming write_data_to_npy correctly writes the data in NumPy format
        write_data_to_npy(filename, full_solution, full_solution.size(), 1);
      } // end output_npy
  }

  // ----------------- output pvtu -----------------
  template <int dim>
  void
  Darcy<dim>::output_pvtu(const std::string &output_path) const
  {
    const std::size_t found    = output_path.find_last_of("/\\");
    const std::string filename = output_path.substr(found + 1) + "_solution";
    const std::string stripped_path = output_path.substr(0, found + 1);

    // define the solution vector
    std::vector<std::string> solution_names(dim, "u");
    solution_names.emplace_back("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.emplace_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation);

    // define the subdomains
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    // build the patches
    MappingQ<dim> mapping(1); // linear mapping
    data_out.build_patches(mapping, degree_u, DataOut<dim>::curved_inner_cells);

    constexpr unsigned int num_vtu_files    = 4;
    constexpr unsigned int n_digits_counter = 2;
    constexpr unsigned int cycle = 0; // a counter for iterative output
    std::cout << "stripped_path: " << stripped_path << std::endl;
    data_out.write_vtu_with_pvtu_record(stripped_path,
                                        filename,
                                        cycle,
                                        MPI_COMM_WORLD,
                                        n_digits_counter,
                                        num_vtu_files);

    // define the random field
    const std::string filename_rf =
      output_path.substr(found + 1) + "_random_field";
    const std::string stripped_path_rf = output_path.substr(0, found + 1);

    DataOut<dim> data_out_rf;
    data_out_rf.set_flags(flags);

    std::string    random_field_names = "k";
    Vector<double> rf(x_vec.size());
    for (unsigned int i = 0; i < x_vec.size(); ++i)
      {
        rf[i] = std::exp(x_vec[i]);
      }
    data_out_rf.add_data_vector(rf_dof_handler, rf, random_field_names);

    data_out_rf.build_patches(mapping, degree_p);
    data_out_rf.write_vtu_with_pvtu_record(stripped_path_rf,
                                           filename_rf,
                                           cycle,
                                           MPI_COMM_WORLD,
                                           n_digits_counter,
                                           num_vtu_files);
  }

  // ----------------- run methods -----------------
  template <int dim>
  void
  Darcy<dim>::run(const std::string &input_path, const std::string &output_path)
  {
    run_simulation(input_path, output_path);

    computing_timer.print_summary();
    computing_timer.reset();
    pcout << std::endl;
  }

  template <int dim>
  void
  Darcy<dim>::run_simulation(const std::string &input_path,
                             const std::string &output_path)
  {
    setup_grid_and_dofs();
    read_input_npy(input_path);
    // generate_ref_input(); // optional for testing: not gt
    generate_coordinates();
    assemble_approx_schur_complement();
    assemble_system();
    solve();

    // output the results
    output_pvtu(output_path);
    output_full_velocity_npy(output_path);
    output_velocity_at_observation_points_npy(output_path);
    output_field_at_observation_points_npy(output_path);

  } // end run simulation

} // namespace darcy

// ----------------- main -----------------
int
main(int argc, char *argv[])
{
  std::string input_file_path  = argv[1];
  std::string output_file_path = argv[2];

  try
    {
      using namespace darcy;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, numbers::invalid_unsigned_int);
      const unsigned int fe_degree = 1;
      Darcy<2>           mixed_laplace_problem(fe_degree);
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
