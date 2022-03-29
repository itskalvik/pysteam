import numpy as np
import numpy.linalg as npla

from ..problem import OptimizationProblem
from . import GaussNewtonSolver


class LevMarqGaussNewtonSolver(GaussNewtonSolver):

  def __init__(self, problem: OptimizationProblem, **parameters) -> None:
    super().__init__(problem, **parameters)
    # override parameters
    self._parameters.update({
        "ratio_threshold": 0.25,
        "shrink_coeff": 0.1,
        "grow_coeff": 10.0,
        "max_shrink_steps": 50,
    })
    self._parameters.update(**parameters)

    self._diag_coeff = 1e-7

  def linearize_solve_and_update(self):

    # initialize new cost with old cost in case of failure
    new_cost = self._prev_cost

    # build the system
    A, b = self.build_gauss_newton_terms()
    grad_norm = npla.norm(b)  # compute gradient norm for termination check
    self._approx_hessian = A  # keep a copy of the LHS (i.e., the approximated Hessian)

    # perform LM search
    num_tr_decreases = 0
    num_backtrack = 0
    step_success = False
    while num_backtrack < self._parameters["max_shrink_steps"]:
      try:
        perturbation = self.solve_lev_marq(A, b)
        decomp_success = True
      except npla.LinAlgError:
        decomp_success = False

      if decomp_success:
        proposed_cost = self.propose_update(perturbation)
        actual_reduc = self._prev_cost - proposed_cost
        predicted_reduc = self.predict_reduction(A, b, perturbation)
        actual_to_predicted_ratio = actual_reduc / predicted_reduc
      else:
        actual_to_predicted_ratio = 0.0

      if decomp_success and actual_to_predicted_ratio > self._parameters["ratio_threshold"]:
        self.accept_proposed_state()
        self._diag_coeff = max(self._diag_coeff * self._parameters["shrink_coeff"], 1e-7)
        new_cost = proposed_cost
        step_success = True
        break
      else:
        if decomp_success:
          self.reject_proposed_state()
        self._diag_coeff = min(self._diag_coeff * self._parameters["grow_coeff"], 1e7)
        num_tr_decreases += 1

      num_backtrack += 1

    # print report line if verbose option is enabled
    if (self._parameters["verbose"]):
      print("Iteration: {0:4}  -  Cost: {1:10.4f}  -  TR Shrink: {2:6.3f}  -  AvP Ratio: {3:6.3f}".format(
          self._curr_iteration, new_cost, num_tr_decreases, actual_to_predicted_ratio))

    return step_success, new_cost, grad_norm

  def solve_lev_marq(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the Levenberg–Marquardt system of equations:
      A*x = b, A = (J^T*J + diagonalCoeff*diag(J^T*J))
    """
    # augment diagonal of the 'hessian' matrix
    if self._parameters["use_sparse_matrix"]:
      A.setdiag(A.diagonal() * (1 + self._diag_coeff))
    else:
      np.fill_diagonal(A, np.diag(A) * (1 + self._diag_coeff))

    # solve system
    try:
      lev_marq_step = self.solve_gauss_newton(A, b)
    except npla.LinAlgError:
      raise npla.LinAlgError('Decomposition Failure')
    finally:
      # revert diagonal of the 'hessian' matrix
      if self._parameters["use_sparse_matrix"]:
        A.setdiag(A.diagonal() / (1 + self._diag_coeff))
      else:
        np.fill_diagonal(A, np.diag(A) / (1 + self._diag_coeff))

    return lev_marq_step

  def predict_reduction(self, A: np.ndarray, b: np.ndarray, step: np.ndarray) -> float:
    # grad^T * step - 0.5 * step^T * Hessian * step
    grad_trans_step = b.T @ step
    step_trans_hessian_step = step.T @ A @ step
    return (grad_trans_step - 0.5 * step_trans_hessian_step)[0, 0]
