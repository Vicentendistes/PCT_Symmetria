import math

import torch.linalg
#import polyscope as ps
import numpy as np


def matmul(mats):
    out = mats[0]

    for i in range(1, len(mats)):
        out = np.matmul(out, mats[i])

    return out


# Returns the rotation matrix in X
def rotationX(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]], dtype=np.float32)


# Returns the rotation matrix in Y
def rotationY(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]], dtype=np.float32)


# Returns the rotation matrix in Y
def rotationZ(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)


# Returns the translation matrix
def translate(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]], dtype=np.float32)


# Stores the information of a symmetry plane
class SymmetryPlane:
    def __init__(self, point, normal):
        # 3D coords of a canonical plane (for drawing)
        self.coordsBase = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=np.float32)
        # Indices for the canonical plane
        self.trianglesBase = np.array([[0, 1, 3], [3, 1, 2]], dtype=np.int32)

        # The plane is determined by a normal vector and a point
        self.point = point.astype(np.float32)
        self.normal = normal
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-5)

        self.compute_geometry()

    # Applies a rotation to the plane
    def apply_rotation(self, rot):
        transf = rot.copy()
        transf = transf[0:3, 0:3]
        transf = np.linalg.inv(transf).T

        self.normal = transf @ self.normal

        self.compute_geometry()

    def apply_traslation(self, x, y, z):
        self.point[0] = self.point[0] + x
        self.point[1] = self.point[1] + y
        self.point[2] = self.point[2] + z

        # print(self.point)

    # Transforms the canonical plane to be oriented wrt the normal
    def compute_geometry(self):
        # Be sure the vector is normal
        self.normal = self.normal / (np.linalg.norm(self.normal) + 1e-6)
        # print(f'First normal: {self.normal}')
        a, b, c = self.normal

        h = np.sqrt(a ** 2 + c ** 2)

        if h < 0.0000001:
            angle = np.pi / 2

            T = translate(self.point[0], self.point[1], self.point[2])
            Rz = rotationZ(angle)
            transform = matmul([T, Rz])
        else:

            Rzinv = np.array([
                [h, -b, 0, 0],
                [b, h, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            Ryinv = np.array([
                [a / h, 0, -c / h, 0],
                [0, 1, 0, 0],
                [c / h, 0, a / h, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            T = translate(self.point[0], self.point[1], self.point[2])

            transform = matmul([T, Ryinv, Rzinv])

        ones = np.ones((1, 4))
        self.coords = np.concatenate((self.coordsBase.T, ones))

        self.coords = transform @ self.coords

        self.coords = self.coords[0:3, :].T


def get_angle(a, b, radians=True):
    """

    :param a: Shape B x N x 3
    :param b: Shape B x N x 3
    :param radians: Flag to determine if return degrees or radians
    :return: Shape B x N of angles  between each vector
    """
    inner_product = torch.einsum('bnd,bnd->bn', a, b)
    a_norm = torch.linalg.norm(a, dim=2)
    b_norm = torch.linalg.norm(b, dim=2)
    # Avoiding div by 0
    cos = inner_product / ((a_norm * b_norm) + 1e-8)
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos)
    if radians:
        return angle  # * 180 / math.pi
    else:
        return angle * 180 / math.pi


class SymPlane:
    def __init__(self, normal, point, confidence=None, normalize=False):
        if normalize:
            self.normal = torch.nn.functional.normalize(normal, dim=0)
        else:
            self.normal = normal
        self.point = point
        self.offset = - torch.dot(point, normal)
        self.confidence = confidence

    def to_tensor(self):
        if self.confidence is not None:
            conf = torch.tensor(self.confidence).unsqueeze(dim=0)
            out = torch.cat((self.normal, self.point, conf), dim=0)
        else:
            out = torch.cat((self.normal, self.point), dim=0)
        return out

    @staticmethod
    def from_tensor(plane_tensor, confidence=None, normalize=False):
        out = SymPlane(plane_tensor[0:3], plane_tensor[3:6], normalize=normalize)
        if confidence is not None:
            out.confidence = confidence
        return out

    @staticmethod
    def from_array(plane_array, confidence=None):
        plane_tensor = torch.tensor(plane_array)
        return SymPlane(plane_tensor[0:3], plane_tensor[3:6], confidence)

    def get_distance_to_points(self, points):
        # N x 3 @ 3 x 1 -> N x 1
        return points @ self.normal.view(3, 1) + self.offset

    def reflect_points(self, points):
        distances = self.get_distance_to_points(points).squeeze()
        return points - 2 * torch.einsum('p,d->pd', distances, self.normal)

    def get_angle_between_planes(self, another_plane):
        return 1 - torch.abs(
            torch.dot(
                self.normal, another_plane.normal
            )
        )

    def is_close(self, another_plane, angle_threshold=0.0872665, distance_threshold=0.01):

        angle, signed_distance = self.get_distances(another_plane)
        return angle < angle_threshold and torch.abs(signed_distance) < distance_threshold

    def get_distances(self, another_plane):
        angle = self.get_angle_between_planes(another_plane)
        distance = self.get_distance_to_points(another_plane.point.unsqueeze(0)).squeeze()
        return angle, distance

    def visualize_plane(self, name, points=None):
        if points is None:
            self._visualize_at(name, self.point)
        else:
            self._visualize_at(name, points.mean(dim=0))

    def __repr__(self):
        return f"N:{self.normal}, P:{self.point}, C:{self.confidence}"

    # def _visualize_at(self, name, point):
    #     point_in_plane = point - self.get_distance_to_points(point.unsqueeze(dim=0)).squeeze()
    #     internal_rep = SymmetryPlane(self.normal.detach().numpy(), point_in_plane.detach().numpy())
    #     ps.register_surface_mesh(
    #         name,
    #         internal_rep.coords,
    #         internal_rep.trianglesBase,
    #     )
