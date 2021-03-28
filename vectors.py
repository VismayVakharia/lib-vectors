#!/usr/share/env python3.8

from typing import Union, List, Tuple

import numpy as np
import quaternion

Quaternion = quaternion.quaternion


class Vector3D(object):

    @classmethod
    def from_list(cls, list_: Union[List, Tuple]) -> 'Vector3D':
        if len(list_) == 3:
            x = list_[0]
            y = list_[1]
            z = list_[2]
            return cls(x, y, z)
        else:
            return cls()

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = x, y, z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3D(x, y, z)

    def __mul__(self, other: Union[float, int]) -> 'Vector3D':
        return Vector3D(self.x * other, self.y * other, self.z * other)

    def __sub__(self, other) -> 'Vector3D':
        return self + other * (-1)

    def __eq__(self, other: 'Vector3D') -> bool:
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __ne__(self, other: 'Vector3D') -> bool:
        return (self.x, self.y, self.z) != (other.x, other.y, other.z)

    def __str__(self) -> str:
        return "<{0.x}, {0.y}, {0.z}>".format(self)

    def __repr__(self):
        return "<{0.x}, {0.y}, {0.z}>".format(self)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def norm(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        norm = self.norm()
        self.x = self.x / norm
        self.y = self.y / norm
        self.z = self.z / norm

    def midpoint(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D.from_list([ix / 2 for ix in self + other])

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        s1 = self.y * other.z - self.z * other.y
        s2 = self.z * other.x - self.x * other.z
        s3 = self.x * other.y - self.y * other.x
        return Vector3D(s1, s2, s3)

    def distance(self, other: 'Vector3D') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def rotate(self, axis: 'Vector3D', angle: float):
        axis.normalize()
        rotation_quaternion = quaternion.from_rotation_vector((axis * angle).as_list())
        rotated = quaternion.rotate_vectors(rotation_quaternion, self.as_list())
        self.x, self.y, self.z = rotated[0], rotated[1], rotated[2]
