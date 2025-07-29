# Isotropic porous media flow and adjoint formulation for BMFIA
This repository contains a fully parallelized implementation of a 2D isotropic porous media flow, using deal.II. The governing equations that are solved read as follows:

```math
\begin{align*}
K^{-1}(\boldsymbol{x},\boldsymbol{c})\cdot\boldsymbol{u} + \nabla p&=0 \text{ in } \Omega\\
\text{div} \boldsymbol{u} &= 0 \text{ in } \Omega\\
p(\boldsymbol{c}) &= g(\boldsymbol{c}) \text{ on } \delta\Omega
\end{align*}
```

With $K$ being a $2\times2$ permeability tensor, $\boldsymbol{u}$ the flow's velocity and $p$ the pressure field.
We implemented an isotropic permeability field $K(\boldsymbol{x},\boldsymbol{c})=\exp(k(\boldsymbol{x},\boldsymbol{c}))\cdot I$, which is modeled by the scalar permeability field $k(\boldsymbol{x},\boldsymbol{c})$. The latter is parameterized by the vector $\boldsymbol{x}$, reflecting its finite element discretization (nodal values).

Please find more information in the associated publication.


## Executables
The project provides two executables:
-  1) `darcy.cc` $\rightarrow$ `darcy`: The actual main executable and forward solve of the governing equations for a specific choice permeability field parameterization $\boldsymbol{x}$ which imposes the mapping: $`\boldsymbol{y}=f(\boldsymbol{x})`$
-  2) `darcy_adjoint.cc` $\rightarrow$ `darcy_adjoint`: The associated adjoint problem that returns the derivative $`\frac{\partial g(f(\boldsymbol{x}))}{\partial \boldsymbol{x}}`$ for an objective function $`g`$

## Setup, installation, build and dependencies
This code requires the installation and setup of [deal.ii](https://www.dealii.org/), furthermore, the [Trilinos](https://trilinos.github.io/) project needs to be configured and installed. For parallel computing, respectively partitioning we furthermore require the installation of 
[p4est](pymc.io/projects/examples/en/latest/gallery.html). The implementation of the [random fields is a separate project](https://gitlab.lrz.de/adjoint_problems/random_fields) that needs to be downloaded. The import path for the respective header file might need to be adjusted in the source code.


## Running the executables
The executables require
- a path to an input file
- a path to the output directory
- the adjoint executable requires additionally a file `adjoint_data.npy` which has to be located in the same directory as the input file.

The primary problem can be started with `mpirun -np <num_procs> darcy <path/to/test_input.npy> <path/to/output_dir>`. The associated adjoint problem with analogous with `mpirun -np <num_procs> darcy_adjoint <path/to/test_input.npy> <path/to/output_dir>`


## Associated deal.II tutorials
The code is inspired by the deal.ii [tutorial 20](https://www.dealii.org/developer/doxygen/deal.II/step_20.html) for the overall idea, [tutorials 21](https://www.dealii.org/developer/doxygen/deal.II/step_21.html) and [22](https://www.dealii.org/developer/doxygen/deal.II/step_22.html) for handling block systems and further enhanced solution strategies. We furthermore used [tutorial 43](https://www.dealii.org/developer/doxygen/deal.II/step_43.html) for improved solvers and an efficient block-preconditioner. Finally, [tutorials 31](https://www.dealii.org/developer/doxygen/deal.II/step_31.html) and [32](https://www.dealii.org/developer/doxygen/deal.II/step_32.html) were used to parallelize the code for use on multiple processors.

