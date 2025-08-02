// preconditioner.h
#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>

namespace Preconditioner
{

  // -----  inverse matrix class ----------------------
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix
  {
  public:
    InverseMatrix(const MatrixType         &m,
                  const PreconditionerType &preconditioner);

    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    const MatrixType         &matrix;
    const PreconditionerType &preconditioner;
  };

  // constructor
  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType         &m,
    const PreconditionerType &preconditioner)
    : matrix(m)
    , preconditioner(preconditioner)
  {}

  // vmult function
  template <class MatrixType, class PreconditionerType>
  template <typename VectorType>
  void
  InverseMatrix<MatrixType, PreconditionerType>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    dealii::SolverControl solver_control(src.size(), 1e-7 * src.l2_norm());
    dealii::SolverCG<VectorType> cg(solver_control);

    dst = 0;

    try
      {
        cg.solve(matrix, dst, src, preconditioner);
      }
    catch (std::exception &e)
      {
        Assert(false, dealii::ExcMessage(e.what()));
      }
  }

  // -----  block-preconditioner class -----------------
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  class BlockSchurPreconditioner
  {
  public:
    BlockSchurPreconditioner(
      const dealii::TrilinosWrappers::BlockSparseMatrix &System,
      const PreconditionerTypeaS                        &ap_S_inv,
      const PreconditionerTypeM                         &ap_M_inv);

    void
    vmult(dealii::TrilinosWrappers::MPI::BlockVector       &dst,
          const dealii::TrilinosWrappers::MPI::BlockVector &src) const;

  private:
    const dealii::TrilinosWrappers::BlockSparseMatrix &system_matrix;
    const PreconditionerTypeaS                        &ap_S_inv;
    const PreconditionerTypeM                         &ap_M_inv;
  };

  // constructor
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::
    BlockSchurPreconditioner(
      const dealii::TrilinosWrappers::BlockSparseMatrix &System,
      const PreconditionerTypeaS                        &ap_S_inv,
      const PreconditionerTypeM                         &ap_M_inv)
    : system_matrix(System)
    , ap_S_inv(ap_S_inv)
    , ap_M_inv(ap_M_inv)
  {}

  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  void
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::vmult(
    dealii::TrilinosWrappers::MPI::BlockVector       &dst,
    const dealii::TrilinosWrappers::MPI::BlockVector &src) const
  {
    dealii::TrilinosWrappers::MPI::Vector utmp(src.block(1));
    // tmp(complete_index_set(system_matrix.block(1, 1).m())) // there is a
    // problem in parallel
    ap_M_inv.vmult(dst.block(0), src.block(0));
    system_matrix.block(1, 0).residual(utmp, dst.block(0), src.block(1));
    utmp *= -1;
    ap_S_inv.vmult(dst.block(1), utmp);
  }

} // end namespace Preconditioner

#endif
