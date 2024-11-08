import numpy as np
from pylgmath import se3op

from ..evaluable import Evaluable, Jacobians, Node

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class ComposeSonarLandmarkEvaluator(Evaluable):
    """Evaluator for the composition of a transformation evaluator and sonar landmark state."""

    def __init__(self, transform: Evaluable, landmark: Evaluable):
        super().__init__()
        self._transform: Evaluable = transform # Sonar Pose (coincident with robot pose, T_wr)
        self._landmark: Evaluable = landmark # landmark x, y, z in world frame (homogeneous coordinates)

    @property
    def active(self) -> bool:
        return self._transform.active or self._landmark.active

    @property
    def related_var_keys(self) -> set:
        return self._transform.related_var_keys | self._landmark.related_var_keys

    def forward(self) -> Node:
        transform_child = self._transform.forward()
        landmark_child = self._landmark.forward()

        # Convert landmark from world frame to robot frame
        landmark_robot = transform_child.value.inverse().matrix() @ landmark_child.value

        print("Forward")
        print("T_rw:\n", transform_child.value.inverse().matrix())
        print("P_w:\n", landmark_child.value)
        print("P_r:\n", landmark_robot)

        # Get angle between each axis and landmark
        if np.sum(landmark_robot[:3]) > 0:
            # Distance to point in robot frame
            distance = np.linalg.norm(landmark_robot[:3, 0])
            # Angle between x-axis and point (ideally the sonar angle)
            theta = angle_between((1, 0, 0), landmark_robot[:3, 0])
            # Angle between z-axis and point (ideally 1.5708, i.e., perpendicular to the robot's z-axis)
            # Assuming that the robot's front direction is along the z-axis
            gamma = angle_between((0, 0, 1), landmark_robot[:3, 0])
        else:
            # Base case if the estimated landmark is at [0, 0, 0, 1]
            distance = 0.0
            theta = 0.0
            gamma = 0.0

        value = np.array([distance,
                          theta,
                          gamma]).reshape(-1, 1)        

        return Node(value, transform_child, landmark_child)

    def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
        transform_child, landmark_child = node.children

        if self._transform.active:
            homogeneous = node.value
            new_lhs = lhs @ se3op.point2fs(homogeneous)
            self._transform.backward(new_lhs, transform_child, jacs)

        if self._landmark.active:
            land_jac = transform_child.value.matrix()[:4, :3]
            self._landmark.backward(lhs @ land_jac, landmark_child, jacs)
