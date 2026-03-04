import math

import torch
from src.utils.quaternion import create_quat, apply_rotation, rotate_shape


class RotAxis:
    def __init__(self, axis, point, angle, confidence=None):

        self.axis = torch.nn.functional.normalize(axis, dim=0)
        self.point = point
        self.confidence = confidence
        self.angle = angle

        self.quat = create_quat(axis, angle)

    def to_tensor(self):
        if self.confidence is not None:
            conf = torch.tensor(self.confidence).unsqueeze(dim=0)
            out = torch.cat((self.axis, self.point, self.angle, conf), dim=0)
        else:
            out = torch.cat((self.axis, self.point, self.angle), dim=0)
        return out

    @staticmethod
    def from_tensor(rotation_tensor, confidence=None):
        out = RotAxis(rotation_tensor[0:3], rotation_tensor[3:6], rotation_tensor[6:7])
        if confidence is not None:
            out.confidence = confidence
        return out

    def rotate_points(self, points):
        return apply_rotation(self.quat, points - self.point)

    def get_angle_between_axis(self, another_axis):
        return 1 - torch.abs(
            torch.dot(
                self.axis, another_axis.axis
            )
        )

    def get_distance_to_points(self, points):
        return torch.norm(
            torch.linalg.cross(self.point - points, self.axis.unsqueeze(dim=0).repeat(points.shape[0], 1)),
            dim=1
        )

    def is_close(self, another_plane, angle_threshold, distance_threshold, rot_angle_threshold):
        angle, signed_distance, rot_angle_difference = self.get_distances(another_plane)
        return (angle < angle_threshold and
                torch.abs(signed_distance) < distance_threshold and
                torch.abs(rot_angle_difference) < rot_angle_threshold)

    def get_distances(self, another_axis):
        angle_between_axis = self.get_angle_between_axis(another_axis)
        distance = self.get_distance_to_points(another_axis.point.unsqueeze(0)).squeeze()
        rot_angle_difference = self.angle - another_axis.angle
        return angle_between_axis, distance, rot_angle_difference

    def __repr__(self):
        return f"N:{self.axis}, P:{self.point}, A:{self.angle}, C:{self.confidence}"

if __name__ == "__main__":
    one_axis = torch.tensor([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.57,
        0.0
    ])
    one_axis_obj = RotAxis.from_tensor(one_axis[0:7], one_axis[7])
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    print(one_axis_obj)
    print(one_axis_obj.get_distance_to_points(test_points))
    print(one_axis_obj.rotate_points(test_points))
    print(rotate_shape(one_axis_obj.axis, one_axis_obj.point, one_axis_obj.angle, test_points))