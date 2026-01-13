import numpy as np


class RotationMatrixX:
    def __init__(self, angle: float) -> None:
        self.angle = angle

    def get_matrix(self) -> np.ndarray:
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        return np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])


class RotationMatrixY:
    def __init__(self, angle: float) -> None:
        self.angle = angle

    def get_matrix(self) -> np.ndarray:
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        return np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])


class RotationMatrixZ:
    def __init__(self, angle: float) -> None:
        self.angle = angle

    def get_matrix(self) -> np.ndarray:
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        return np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])


def rotate_3d_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, angle: float, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate 3D points around a specified axis by a given angle.

    Parameters:
    - x, y, z: np.ndarray
        Arrays of the same shape containing coordinates of points.
    - angle: float
        Rotation angle in radians.
    - axis: str
        Axis to rotate around: 'x', 'y', or 'z'.

    Returns:
    - x_rot, y_rot, z_rot: np.ndarray
        Rotated coordinates with the same shape as input.
    """

    if axis.lower() == "x":
        # Rotation matrix around x-axis
        rotation_matrix = RotationMatrixX(angle).get_matrix()

    elif axis.lower() == "y":
        rotation_matrix = RotationMatrixY(angle).get_matrix()

    elif axis.lower() == "z":
        # Rotation matrix around z-axis
        rotation_matrix = RotationMatrixZ(angle).get_matrix()

    else:
        raise ValueError("Invalid axis specified. Choose from 'x', 'y', or 'z'.")

    # Apply rotation
    x_rot = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2] * z
    y_rot = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2] * z
    z_rot = rotation_matrix[2, 0] * x + rotation_matrix[2, 1] * y + rotation_matrix[2, 2] * z

    return x_rot, y_rot, z_rot


def convert_rz_to_xyz(
    R: np.ndarray, Z: np.ndarray, phi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cylindrical coordinates (R, Z) to Cartesian coordinates (X, Y).

    Parameters:
    - R: np.ndarray
        Radial coordinates.
    - Z: np.ndarray
        Axial coordinates.
    - phi: float
        Azimuthal angle in radians.

    Returns:
    - X, Y: np.ndarray
        Cartesian coordinates.
    """
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return X, Y, Z
