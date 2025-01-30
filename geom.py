# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:49:18 2022

@author: Daniel
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from sympy.plotting import plot as symplot


class SymbolicFunction:
    def __init__(self, str_expr: str, symbols: str, **kwargs):
        self.symbols: list[str] = []
        self.symbols_dict: dict[str, sp.Symbol] = {}
        if symbols.strip() != "":
            x = sp.symbols(symbols)
            if type(x) == sp.Symbol:
                x = [x]
            self.symbols = symbols.split(' ')
            self.symbols_dict = dict(zip(self.symbols, x))
            for key in kwargs:
                if key in self.symbols:
                    i = self.symbols.index(key)
                    expr = expr.subs(x[i], kwargs[key])

        expr = sp.sympify(str_expr)
        self.expr: sp.Expr = expr

    def eval_array(self, **kwargs):
        array_length: int = len(list(kwargs.values())[0])
        result: np.ndarray = np.zeros(array_length)
        for i in range(array_length):
            kwargs_i = {key: v[i] for key, v in kwargs.items()}
            result[i] = self.eval_single(**kwargs_i)

        return result

    def eval_single(self, **kwargs):
        if len(kwargs) < len(self.expr.free_symbols):
            raise Exception('Not enough arguments.')

        result = self.expr
        for key in kwargs:
            if key in self.symbols_dict and self.symbols_dict[key] in result.free_symbols:
                result: sp.Expr = result.subs(self.symbols_dict[key], kwargs[key])

        if len(result.free_symbols):
            raise Exception('Warning: Not fully evaluated.')

        return float(result)

    def integrate(self, var: sp.Symbol = None, lb=None, ub=None):
        if not var:
            var = self.symbols[0]

        if (lb is None and ub is not None) or (lb is not None and ub is None):
            raise Exception('Bounds not correctly given.')
        else:
            int_expr: sp.Expr = sp.integrate(self.expr, (var, lb, ub))
            sym_str = ''.join([str(x) + ' ' for x in int_expr.free_symbols]).strip()
            return SymbolicFunction(str(int_expr), sym_str)

    def display(self):
        symplot(self.expr)

    def factor(self):
        expr = sp.factor(self.expr)
        self.expr = expr
        return expr


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


class MathUtils:
    @staticmethod
    def transform_translate2D(arr: np.ndarray, v: np.ndarray) -> np.ndarray:
        translation_matrix = np.identity(3)
        translation_matrix[:2, -1] += v

        return np.matmul(translation_matrix, np.vstack([arr, [1] * len(arr[0, :])]))[:2, :]

    @staticmethod
    def transform_rotate2D(arr: np.ndarray, alpha: float,
                           pivot: np.ndarray = np.array([0, 0])) -> np.ndarray:
        T = np.array([[1, 0, pivot[0]],
                      [0, 1, pivot[1]],
                      [0, 0, 1]])
        T_inv = np.array([[1, 0, -pivot[0]],
                          [0, 1, -pivot[1]],
                          [0, 0, 1]])

        R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])

        # print(np.matmul(T_inv,[self.xCoords,self.yCoords,1]))
        return np.matmul(T, np.matmul(R, np.matmul(
            T_inv, np.vstack([arr, [1] * len(arr[0, :])]))))[:2, :]

    @staticmethod
    def translate_rotate(arr: np.ndarray, v: np.ndarray, alpha: float, pivot: np.ndarray=np.array([0, 0])) -> np.ndarray:
        return MathUtils.transform_rotate2D(MathUtils.transform_translate2D(arr, v), alpha, pivot=pivot)
