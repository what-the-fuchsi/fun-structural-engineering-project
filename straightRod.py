# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:13:45 2022

@author: Daniel
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import geom
from geom import *


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

    # def calculateSx_u(self):
    #     for member in self.members:
    #         if isinstance(member, StraightRod):
    #             member.calculateS_u(member.y_u, axis=Axis.X)
    #
    # def calculateSy_u(self):
    #     for member in self.members:
    #         if isinstance(member, StraightRod):
    #             member.calculateS_u()

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
            # print(member.area)
            area += member.area
        self.area = area
        return area

    def display(self):
        for member in self.members:
            member.display('k')
        # plt.show()

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


class StraightRod(Structure):
    def __init__(self, points: geom.PointCloud, thickness: float, name=''):
        super().__init__(name=name)
        self.points: PointCloud = points
        self.thickness: float = thickness
        length = np.sqrt(
            (points.points[-1].x - points.points[0].x) ** 2 + (points.points[-1].y - points.points[0].y) ** 2)
        self.length: float = length

        self.u: np.ndarray = np.linspace(0, length, 100)

        # x(u) and y(u)
        self.x_u: SymbolicFunction = None
        self.y_u: SymbolicFunction = None
        self.Sx_u: SymbolicFunction = None
        self.Sy_u: SymbolicFunction = None

    def linear_sym_funcs(self):

        if not len(self.points.points) == 2:
            raise Exception('Cannot calculate linear function from more than 2 points. ' +
                            f'Structure has {len(self.points.points)} points.')

        x_u = SymbolicFunction('a*u + b', 'u a b')
        y_u = SymbolicFunction('c*u + d', 'u c d')

        com: Point2D = self.parent.center_of_mass
        p1: Point2D = self.points.first_point()
        p2: Point2D = self.points.last_point()

        self.x_u = self.determine_func(com.x, p1.x, p2.x, x_u)
        self.y_u = self.determine_func(com.y, p1.y, p2.y, y_u)

        return x_u, y_u

    def determine_func(self, com, v1, v2, f_u):
        u_max = self.length
        f_eq1 = f_u.expr.subs(f_u.symbols_dict['u'], 0) - v1 + com
        f_eq2 = f_u.expr.subs(f_u.symbols_dict['u'], u_max) - v2 + com
        f_sol = sp.solve([f_eq1, f_eq2])
        if f_sol:
            if isinstance(f_sol, set):
                raise Exception('Multiple solutions cannot be considered yet. TODO')
            # print(hasattr(x_u, 'expr'))
            # print(type(x_u))
            f_u_new = f_u.expr.subs(f_sol.items())
            f_u.expr = f_u_new
            return f_u
        else:
            return None

    def calculate_static_moment_distr(self, axis: Axis):
        # Sx(u) = int(y(u)*t,u)
        if not self.x_u or self.y_u:
            self.linear_sym_funcs()
        xi_u: SymbolicFunction = None
        if axis == Axis.X:
            xi_u = self.y_u
        elif axis == Axis.Y:
            xi_u = self.x_u

        S: SymbolicFunction = xi_u.integrate(xi_u.symbols_dict['u'], 0, xi_u.symbols_dict['u'])
        S = self._shift_static_moment_func(S)
        self._point_values(S, axis)

        self.__setattr__(f'S{axis.value}_u', S)
        return S

    # def calculateSx_u(self):
    #     # Sx(u) = int(y(u)*t
    #     if not self.y_u:
    #         self.linear_sym_funcs()
    #
    #     Sx: SymbolicFunction = self.y_u.integrate(self.y_u.symbols_dict['u'], 0, self.y_u.symbols_dict['u'])
    #     Sx = self._shift_static_moment_func(Sx)
    #     self._point_values(Sx, Axis.X)
    #
    #     self.Sx_u = Sx
    #     return Sx
    #
    # def calculateSy_u(self):
    #     if not self.x_u:
    #         self.linear_sym_funcs()
    #
    #     Sy: SymbolicFunction = self.x_u.integrate(self.x_u.symbols_dict['u'], 0, self.x_u.symbols_dict['u'])
    #     self._shift_static_moment_func(Sy)
    #     self._point_values(Sy, Axis.Y)
    #
    #     self.Sy_u = Sy
    #     return Sy

    def _point_values(self, S: SymbolicFunction, axis: Axis):
        p1: Point2D = self.points.first_point()
        p2: Point2D = self.points.last_point()

        attr_name: str = f'S{axis.value}_value'
        current_value1: float = p1.__getattribute__(attr_name)
        current_value2: float = p2.__getattribute__(attr_name)
        S_start_value: float = S.eval_single(u=self.u[0])
        S_end_value: float = S.eval_single(u=self.u[-1])

        # junc1: Junction = None
        # junc2: Junction = None
        # for junc in self.junctions:
        #     if p1 == junc.point:
        #         if junc1 is None:
        #             junc1 = junc
        #         else:
        #             raise Exception('Something is wrong with the junctions.')
        #
        #     if p2 == junc.point:
        #         if junc2 is None:
        #             junc2 = junc
        #         else:
        #             raise Exception('Something is wrong with the junctions.')
        #
        # if junc1 is not None:
        #     for member in junc1.members:
        #         if junc1 != member:
        #             junc_type: JunctionType = junc1.type_dict[(self, member)]
        #             # TODO



        self._set_point_value(S_start_value, current_value1, attr_name, p1)
        self._set_point_value(S_end_value, current_value2, attr_name, p2)

    def _set_point_value(self, S_value: float, current_value: float, attr_name: str, p: Point2D):
        if current_value is None:
            p.__setattr__(attr_name, np.abs(S_value))
        elif np.abs(S_value) != current_value:
            print(Exception(f'Conflicting point values for attr: {attr_name}\n' +
                            f'\tMember: {self.name}\n' +
                            f'\tMember value: {S_value}\n' +
                            f'\tPoint value: {p.__getattribute__(attr_name)}'))

    def _shift_static_moment_func(self, S):
        if self.points.first_point().is_free:
            S.expr -= S.eval_single(u=self.u[0])
            self.points.last_point().Sx_value = S.eval_single(u=self.u[-1])
        elif self.points.last_point().is_free:
            S.expr -= S.eval_single(u=self.u[-1])
            self.points.first_point().Sx_value = S.eval_single(u=self.u[0])
        else:
            pt = self.points.first_point()
            junc = self.find_junction(pt)
            # TODO

        return S

    def find_junction(self, pt: Point2D) -> Junction:
        junc = None
        for j in self.junctions:
            if j.point == pt:
                junc = j
                break
        return junc

    def calculateStaticMomentX(self):
        # Sx = int(y*t,du)
        # Sx = 1/2*sum((y[i]+y[i+1])*du[i])*t
        if not self.du:
            self.calculate_du()

        y = self.points.y_array()[:-1]
        y_next = self.points.y_array()[1:]
        Sx = 1 / 2 * sum((y + y_next) * self.du) * self.thickness
        self.staticMomentX = Sx
        return Sx

    def calculateStaticMomentY(self):
        # Sy = int(x*t,du)
        # Sy = 1/2*sum((x[i]+x[i+1])*du[i])*t
        if not self.du:
            self.calculate_du()

        x = self.points.x_array()[:-1]
        x_next = self.points.x_array()[1:]
        Sy = 1 / 2 * sum((x + x_next) * self.du) * self.thickness
        self.staticMomentX = Sy
        return Sy

    def calculate_du(self):
        total_length = 0.
        n_du = len(self.points.points) - 1
        du = np.array((n_du,))

        for i in range(n_du):
            point = self.points.points[i]
            next_point = self.points.points[i + 1]
            du[i] = np.sqrt((next_point.x - point.x) ** 2 + (next_point.y - point.y) ** 2)

        self.du = du
        return du

    def calculateMomentOfInertiaX(self):
        # Jx = int(y**2*t,du)
        # Jx = 1/3*sum((y[i]**2+y[i]*y[i+1]+y[i+1]**2) * t * delta_u[i],i=[0..len(y)])
        y = self.points.y_array()[:-1]
        y_next = self.points.y_array()[1:]

        Jx = 1 / 3 * np.sum((y ** 2 + y * y_next + y_next ** 2) * self.du) * self.thickness
        self.momentOfInertiaX = Jx
        return Jx

    def calculateMomentOfInertiaY(self):
        # Jx = int(y**2*t,du)
        # Jx = 1/3*sum((y[i]**2+y[i]*y[i+1]+y[i+1]**2) * t * delta_u[i],i=[0..len(y)])
        x = self.points.x_array()[:-1]
        x_next = self.points.x_array()[1:]

        Jy = 1 / 3 * np.sum((x ** 2 + x * x_next + x_next ** 2) * self.du) * self.thickness
        self.momentOfInertiaY = Jy
        return Jy

    def calculateMomentOfInertiaXY(self):
        # Jxy = int(x*y,du)
        # Jx = 1/6*sum((2*x[i]*y[i]+x[i]*y[i+1]+x[i+1]*y[i]+2*x[i+1]*y[i+1]) * t * delta_u[i],i=[0..len(y)])
        x = self.points.x_array()[:-1]
        x_next = self.points.x_array()[1:]

        y = self.points.y_array()[:-1]
        y_next = self.points.y_array()[1:]

        Jxy = 1 / 6 * np.sum((2 * x * y + x * y_next + x_next * y + 2 * x_next * y_next) * self.du) * self.thickness
        self.momentOfInertiaXY = Jxy
        return Jxy

    def calculateArea(self):
        # A = sum(l[i])*t for t=const
        if not self.du:
            self.calculate_du()

        area = sum(self.du) * self.thickness
        self.area = area
        return area

    def display(self, style=''):
        pointsArray = self.points.asarray()
        plt.plot(*pointsArray, style)
        plt.axis('equal')
        plt.grid(True)

    def display_Sx(self):
        Sx_u_d: np.ndarray = self.Sx_u.eval_array(u=self.u)
        xy_arr, pattern_arrs = self.transform_for_display_u(Sx_u_d)
        # print("Display Sx")
        plt.plot(*xy_arr, 'b')
        plt.plot(*pattern_arrs, 'b', linewidth=0.5)

    def display_Sy(self):
        Sy_u_d: np.ndarray = self.Sy_u.eval_array(u=self.u)
        xy_arr, pattern_arrs = self.transform_for_display_u(Sy_u_d)

        plt.plot(*xy_arr, 'b')
        plt.plot(*pattern_arrs, 'b', linewidth=0.5)

    def transform_for_display_u(self, S_u_d):
        xy_arr = np.array([self.u, S_u_d])

        # scale y
        max_y = np.max(np.abs(S_u_d))
        if max_y != 0:
            # determine longest member in parent structure
            mem_lengths = np.array([mem.length for mem in self.parent.members])
            scale_factor = 1 / max_y * 0.2 * np.max(mem_lengths)
            xy_arr[1, :] = xy_arr[1, :] * scale_factor

        pattern_arrs = np.array([[self.u, self.u], [[0] * len(S_u_d), xy_arr[1, :]]])
        pattern_arrs = pattern_arrs[:, :, ::int(len(S_u_d) / 10)]
        first_point = self.points.first_point()
        last_point = self.points.last_point()

        # find translation vector
        translation_arr = first_point.asarray()

        # find rotation angle
        v1 = last_point.asarray() - first_point.asarray()
        x_id = np.array([1, 0])  # norm = 1
        alpha = np.arccos(np.dot(v1, x_id) / (np.linalg.norm(v1)))
        if np.cross(x_id, v1) < 0:
            alpha -= np.pi
        # apply transformations
        xy_arr = MathUtils.translate_rotate(xy_arr, translation_arr, alpha, pivot=first_point.asarray())
        pattern_arrs[:, 0] = MathUtils.translate_rotate(pattern_arrs[:, 0], translation_arr, alpha,
                                                        pivot=first_point.asarray())
        pattern_arrs[:, 1] = MathUtils.translate_rotate(pattern_arrs[:, 1], translation_arr, alpha,
                                                        pivot=first_point.asarray())

        return xy_arr, pattern_arrs

    def transformTranslate(self, v) -> None:
        newPoints = PointCloud()
        newPoints.addPointsFromArrays(*MathUtils.transform_translate2D(self.points.asarray(), v))
        self.points = newPoints

    def transformRotate(self, alpha: float, pivot: Point2D = Point2D()) -> None:
        xArray, yArray = MathUtils.transform_rotate2D(self.points.asarray(), alpha, pivot=pivot)
        newPoints = PointCloud()
        newPoints.addPointsFromArrays(xArray, yArray)
        self.points = newPoints


class Junction:
    def __init__(self, point: Point2D = None, members: list[Structure] = []):
        self.point = point
        point.is_junction = True
        self.members: list[Structure] = []
        for member in members:
            self.add_member(member)

        self.type_dict: dict[tuple, JunctionType] = {}

    def resolve_members(self):
        for member in self.members:
            for other_member in self.members:
                right_condition = (other_member is not member
                                   and isinstance(member, StraightRod)
                                   and isinstance(other_member, StraightRod))
                if right_condition:
                    key = (member, other_member)
                    inv_key = (other_member, member)
                    if key not in self.type_dict and inv_key not in self.type_dict:
                        if member.u[0] == other_member.u[0]:
                            # scenario: <- ->
                            self.type_dict[key] = JunctionType.DIV
                            self.type_dict[inv_key] = JunctionType.DIV
                        elif member.u[0] == other_member.u[-1] or member.u[-1] == other_member.u[0]:
                            # scenario: ->-> or <-<-
                            self.type_dict[key] = JunctionType.CONT
                            self.type_dict[inv_key] = JunctionType.CONT
                        elif member.u[-1] == other_member.u[-1]:
                            # scenario: -><-
                            self.type_dict[key] = JunctionType.CONV
                            self.type_dict[inv_key] = JunctionType.CONV
                        else:
                            raise Exception('Faulty junction.')

    def add_member(self, member: Structure):
        self.members.append(member)
        member.junctions.append(self)


class JunctionType(Enum):
    CONT = 1
    CONV = 2
    DIV = 3

class Axis(Enum):
    X = 'x'
    Y = 'y'
    XY = 'xy'