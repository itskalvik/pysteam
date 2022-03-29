import numpy as np
import numpy.linalg as npla

from ..problem import OptimizationProblem
from . import GaussNewtonSolver


class DoglegGaussNewtonSolver(GaussNewtonSolver):

  def __init__(self, problem: OptimizationProblem, **parameters) -> None:
    super().__init__(problem, **parameters)
    # override parameters
    self._parameters.update({
        "ratio_threshold_shrink": 0.25,
        "ratio_threshold_grow": 0.75,
        "shrink_coeff": 0.5,
        "grow_coeff": 3.0,
        "max_shrink_steps": 50
    })
    self._parameters.update(**parameters)

    self._trust_region_size = None

  def linearize_solve_and_update(self):

    # initialize new cost with old cost in case of failure
    new_cost = self._prev_cost

    # build the system
    A, b = self.build_gauss_newton_terms()
    grad_norm = npla.norm(b)  # compute gradient norm for termination check
    self._approx_hessian = A  # keep a copy of the LHS (i.e., the approximated Hessian)

    # get gradient descent step
    grad_descent_step = self.get_cauchy_point(A, b)
    grad_descent_norm = npla.norm(grad_descent_step)

    # get Gauss-Newton step
    gauss_newton_step = self.solve_gauss_newton(A, b)
    gauss_newton_norm = npla.norm(gauss_newton_step)

    # initialize trust region size (if first time)
    if self._trust_region_size is None:
      self._trust_region_size = gauss_newton_norm

    # perform dogleg step
    num_tr_decreases = 0
    num_backtrack = 0
    step_success = False
    while num_backtrack < self._parameters["max_shrink_steps"]:
      if (gauss_newton_norm <= self._trust_region_size):
        perturbation = gauss_newton_step
        dogleg_segment = "Gauss Newton"
      elif (grad_descent_norm >= self._trust_region_size):
        perturbation = (self._trust_region_size / grad_descent_norm) * grad_descent_step
        dogleg_segment = "Grad Descent"
      else:
        # trust region lies between the GD and GN steps, use interpolation
        assert gauss_newton_step.shape == grad_descent_step.shape

        # get interpolation direction
        gd_to_gn_vector = gauss_newton_step - grad_descent_step

        # calculate interpolation constant
        gd_dot_prod_gd_to_gn = (grad_descent_step.T @ gd_to_gn_vector)[0, 0]
        gd_to_gn_sqr_norm = npla.norm(gd_to_gn_vector)**2
        interp_const = (
            (-gd_dot_prod_gd_to_gn + np.sqrt(gd_dot_prod_gd_to_gn**2 +
                                             (self._trust_region_size**2 - grad_descent_norm**2) * gd_to_gn_sqr_norm)) /
            gd_to_gn_sqr_norm)
        perturbation = grad_descent_step + interp_const * gd_to_gn_vector
        dogleg_segment = "Interp GN&GD"

      proposed_cost = self.propose_update(perturbation)
      actual_reduc = self._prev_cost - proposed_cost
      predicted_reduc = self.predict_reduction(A, b, perturbation)
      actual_to_predicted_ratio = actual_reduc / predicted_reduc

      if actual_to_predicted_ratio > self._parameters["ratio_threshold_shrink"]:
        self.accept_proposed_state()
        if actual_to_predicted_ratio > self._parameters["ratio_threshold_grow"]:
          self._trust_region_size = max(self._trust_region_size,
                                        self._parameters["grow_coeff"] * npla.norm(perturbation))
        new_cost = proposed_cost
        step_success = True
        break
      else:
        self.reject_proposed_state()
        self._trust_region_size *= self._parameters["shrink_coeff"]
        num_tr_decreases += 1

      num_backtrack += 1

    # print report line if verbose option is enabled
    if (self._parameters["verbose"]):
      print(
          "Iteration: {0:4}  -  Cost: {1:10.4f}  -  TR Shrink: {2:6.3f}  -  AvP Ratio: {3:6.3f}  -  Dogleg Segment: {4:15}"
          .format(self._curr_iteration, new_cost, num_tr_decreases, actual_to_predicted_ratio, dogleg_segment))

    return step_success, new_cost, grad_norm

  def get_cauchy_point(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the Levenberg–Marquardt system of equations:
      A*x = b, A = (J^T*J + diagonalCoeff*diag(J^T*J))
    """
    num = npla.norm(b)**2
    den = (b.T @ A @ b)[0, 0]
    return (num / den) * b

  def predict_reduction(self, A: np.ndarray, b: np.ndarray, step: np.ndarray) -> float:
    """grad^T * step - 0.5 * step^T * Hessian * step"""
    grad_trans_step = b.T @ step
    step_trans_hessian_step = step.T @ A @ step
    return (grad_trans_step - 0.5 * step_trans_hessian_step)[0, 0]
