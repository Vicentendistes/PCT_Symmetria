import torch


def create_quat(axis, angle):
    # make sure axis is unit
    axis = axis / torch.norm(axis)
    r = torch.cos(angle / 2)
    i = torch.sin(angle / 2) * axis
    return torch.concat((r, i))


def conjugate(quat):
    return quat * torch.tensor([1, -1, -1, -1], device=quat.device)


def make_quat_points(points):
    zeros = torch.zeros(points.shape[0], device=points.device)
    return torch.concat((zeros.unsqueeze(-1), points), dim=1)


def ham_prod(a_quat, another_quat):
    a1, b1, c1, d1 = a_quat[0], a_quat[1], a_quat[2], a_quat[3]
    a2, b2, c2, d2 = another_quat[0], another_quat[1], another_quat[2], another_quat[3]
    r = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    return torch.stack((r, i, j, k))


def ham_prod_points(quat, quat_points, invert=False):
    if invert:
        return ham_prod(quat_points.T, quat).T
    else:
        return ham_prod(quat, quat_points.T).T


def apply_rotation(quat, points):
    quat_conjugate = conjugate(quat)
    quat_points = make_quat_points(points)  # N x 4
    uno = ham_prod_points(quat, quat_points)
    dos = ham_prod_points(quat_conjugate, uno, invert=True)
    return dos[:, 1:4]


def rotate_shape(axis, point_in_axis, angle, points):
    transformed_points = points - point_in_axis
    return apply_rotation(create_quat(axis, angle), transformed_points) + point_in_axis
