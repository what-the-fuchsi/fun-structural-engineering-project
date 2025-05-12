from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from geom import Point2D, PointCloud
from structures.junction import Junction
from utils.math_utils import Axis

if TYPE_CHECKING:
    from structures.straight_rod import StraightRod

class Structure:
    def __init__(self, members=None, name='', xA=0., yA=0.):
        self.members: list[Structure] = []
        self.parent: Structure = None
        self.points = PointCloud()
        self.member_points: set = set()
        self.junctions: list[Junction] = []

        if members is not None:
            for member in members:
                self.add_member(member)

        self.name: str = name
        self.area: float = None

        self.center_of_mass: Point2D = None

        self.xH: Point2D = None
        self.yH: Point2D = None

        self.xA: Point2D = xA
        self.yA: Point2D = yA

        self.du: np.ndarray = None

        self.staticMomentX: float = None
        self.staticMomentY: float = None
        self.momentOfInertiaX: float = None
        self.momentOfInertiaY: float = None
        self.momentOfInertiaXY: float = None

        self.youngsModulus: float = None

    def calculateMomentOfInertiaX(self):
        Jx = 0.
        for member in self.members:
            Jx += member.calculateMomentOfInertiaX()
        self.momentOfInertiaX = Jx
        return Jx

    def calculateMomentOfInertiaY(self):
        Jy = 0.
        for member in self.members:
            Jy += member.calculateMomentOfInertiaY()
        self.momentOfInertiaY = Jy
        return Jy

    def calculateMomentOfInertiaXY(self):
        Jxy = 0.
        for member in self.members:
            Jxy += member.calculateMomentOfInertiaXY()
        self.momentOfInertiaXY = Jxy
        return Jxy

    def calculate_static_moment_distr(self):
        for member in self.members:
            if isinstance(member, StraightRod):
                if member.y_u is None or member.x_u is None:
                    member.linear_sym_funcs()

                member.calculate_static_moment_distr(Axis.X)
                member.calculate_static_moment_distr(Axis.Y)

    def calculateStaticMomentX(self):
        Sx = 0.
        for member in self.members:
            Sx += member.calculateStaticMomentX()
        self.staticMomentX = Sx
        return Sx

    def calculateStaticMomentY(self):
        Sy = 0.
        for member in self.members:
            Sy += member.calculateStaticMomentY()
        self.staticMomentY = Sy
        return Sy

    def calculateCenterOfMass(self):
        if not self.staticMomentX:
            self.calculateStaticMomentX()

        if not self.staticMomentY:
            self.calculateStaticMomentY()

        if not self.area:
            self.calculateArea()

        xsp = self.staticMomentY / self.area
        ysp = self.staticMomentX / self.area

        center_of_mass = Point2D(xsp, ysp)
        self.center_of_mass = center_of_mass
        return center_of_mass

    def calculateArea(self):
        area = 0.
        for member in self.members:
            member.calculateArea()
            area += member.area
        self.area = area
        return area

    def display(self):
        for member in self.members:
            member.display('k')

    def transformTranslate(self, v):
        for member in self.members:
            member.transformTranslate(v)

    def transformRotate(self, alpha, centerPoint=Point2D()):
        print(self.members)
        for member in self.members:
            member.transformRotate(alpha, centerPoint=centerPoint)

    def add_member(self, member: Structure):
        self.members.append(member)
        member.parent = self

        for point in member.points.points:
            if point in self.member_points:
                if point.is_junction:
                    junc = None
                    for junction in self.junctions:
                        if point == junction.point:
                            junc = junction
                            break
                    junc.add_member(member)
                else:
                    other_members = []
                    for other_member in self.members:
                        if other_member is not member:
                            if point in other_member.points.points:
                                other_members.append(other_member)

                    junc = Junction(point=point, members=other_members)
                    self.junctions.append(junc)

        self.points.add_points_from_point_cloud(member.points)
        self.member_points = self.member_points.union(set(member.points.points)) 