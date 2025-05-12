# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:49:18 2022

@author: Daniel
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from sympy.plotting import plot as symplot


class Point2D:
    def __init__(self, x: float=0, y: float=0, name: str='', is_free: bool=False, is_junction: bool=False):
        self.x = x
        self.y = y
        self.name = name

        self.is_free = is_free
        self.is_junction = is_junction

        self.Sx_value = 0. if is_free else None
        self.Sy_value = 0. if is_free else None


    def asarray(self) -> np.ndarray:
        return np.array([self.x, self.y])


class PointCloud:
    def __init__(self, points: list[Point2D] = None):
        self.points: list[Point2D] = points
        if self.points is None:
            self.points = []

    def first_point(self) -> Point2D:
        return self.points[0]

    def last_point(self) -> Point2D:
        return self.points[-1]

    def addPoints(self, *points) -> None:
        for point in points:
            self.points.append(point)

    def addPointsFromArrays(self, xArray, yArray) -> None:
        for x, y in zip(xArray, yArray):
            self.points.append(Point2D(x, y))

    def add_points_from_point_cloud(self, pc: PointCloud):
        self.addPoints(*pc.points)

    def x_array(self) -> np.ndarray:
        xlist = []
        for point in self.points:
            xlist.append(point.x)
        return np.array(xlist)

    def y_array(self) -> np.ndarray:
        ylist = []
        for point in self.points:
            ylist.append(point.y)
        return np.array(ylist)

    def asarray(self) -> np.ndarray:
        return np.array([self.x_array(), self.y_array()])
