import numpy as np

from ..evaluable import Evaluable, Jacobians, Node


class SonarErrorEvaluator(Evaluable):
    """Currently just ignore the last entry."""

    def __init__(self, pt: Evaluable, meas_pt: np.ndarray):
        super().__init__()
        # compose_sonar_landmark_evaluator output 
        # [distance, theta]
        self._pt: Evaluable = pt

        # Sonar landmark in robot frame
        # [distance, alpha, beta, gamma]
        self._meas_pt: np.ndarray = np.zeros((3, 1))
        self._meas_pt[:2] = meas_pt
        self._meas_pt[2] = 1.570

    @property
    def active(self) -> bool:
        return self._pt.active

    @property
    def related_var_keys(self) -> set:
        return self._pt.related_var_keys

    def forward(self) -> Node:
        landmark_est = self._pt.forward()
        value = np.sum(self._meas_pt - landmark_est.value)

        print("Landmark Est:\n", landmark_est.value)
        print("Landmark GT:\n", self._meas_pt)
        print("Error:", value)

        return Node(value, landmark_est)

    def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
        if self._pt.active:
            self._pt.backward(-lhs, node.children[0], jacs)
