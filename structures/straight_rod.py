from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Optional

from geom import Point2D, PointCloud
from structures.base import Structure
from structures.junction import Junction
from utils.math_utils import Axis
from utils.symbolic import SymbolicFunction

class StraightRod(Structure):
    def __init__(self, points: PointCloud, thickness: float, name=''):
        super().__init__(name=name)
        self.points: PointCloud = points
        self.thickness: float = thickness
        length = np.sqrt(
            (points.points[-1].x - points.points[0].x) ** 2 + 
            (points.points[-1].y - points.points[0].y) ** 2)
        self.length: float = length

        self.u: np.ndarray = np.linspace(0, length, 100)

        # x(u) and y(u)
        self.x_u: Optional[SymbolicFunction] = None
        self.y_u: Optional[SymbolicFunction] = None
        self.Sx_u: Optional[SymbolicFunction] = None
        self.Sy_u: Optional[SymbolicFunction] = None

    def linear_sym_funcs(self):
        """Calculate linear symbolic functions for x(u) and y(u)."""
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
        """Determine coefficients for linear function."""
        u_max = self.length
        f_eq1 = f_u.expr.subs(f_u.symbols_dict['u'], 0) - v1 + com
        f_eq2 = f_u.expr.subs(f_u.symbols_dict['u'], u_max) - v2 + com
        f_sol = sp.solve([f_eq1, f_eq2])
        if f_sol:
            if isinstance(f_sol, set):
                raise Exception('Multiple solutions cannot be considered yet. TODO')
            f_u_new = f_u.expr.subs(f_sol.items())
            f_u.expr = f_u_new
            return f_u
        else:
            return None

    def calculate_static_moment_distr(self, axis: Axis):
        """Calculate static moment distribution for given axis."""
        if not self.x_u or not self.y_u:
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

    def _point_values(self, S: SymbolicFunction, axis: Axis):
        """Calculate point values for static moment function."""
        u_max = self.length
        S_0 = float(S.expr.subs(S.symbols_dict['u'], 0))
        S_max = float(S.expr.subs(S.symbols_dict['u'], u_max))

        p1 = self.points.first_point()
        p2 = self.points.last_point()

        attr_name = f'S{axis.value}'
        self._set_point_value(S_0, None, attr_name, p1)
        self._set_point_value(S_max, None, attr_name, p2)

    def _set_point_value(self, S_value: float, current_value: float, attr_name: str, p: Point2D):
        """Set point value for static moment."""
        if not hasattr(p, attr_name):
            setattr(p, attr_name, S_value)
        elif current_value is None or abs(current_value) > abs(S_value):
            setattr(p, attr_name, S_value)

    def _shift_static_moment_func(self, S):
        """Shift static moment function by thickness."""
        S.expr = S.expr * self.thickness
        return S

    def find_junction(self, pt: Point2D) -> Optional[Junction]:
        """Find junction at given point."""
        for junction in self.parent.junctions:
            if junction.point == pt:
                return junction
        return None

    def calculateStaticMomentX(self):
        """Calculate static moment around X axis."""
        # Sx = int(y*t,du)
        # Sx = 1/2*sum((y[i]+y[i+1])*du[i])*t
        if self.du is None:
            self.calculate_du()
        return np.sum(0.5 * (self.points.y[:-1] + self.points.y[1:]) * self.du) * self.thickness

    def calculateStaticMomentY(self):
        """Calculate static moment around Y axis."""
        # Sy = int(x*t,du)
        # Sy = 1/2*sum((x[i]+x[i+1])*du[i])*t
        if self.du is None:
            self.calculate_du()
        return np.sum(0.5 * (self.points.x[:-1] + self.points.x[1:]) * self.du) * self.thickness

    def calculate_du(self):
        """Calculate differential length elements."""
        dx = np.diff(self.points.x)
        dy = np.diff(self.points.y)
        self.du = np.sqrt(dx**2 + dy**2)

    def calculateMomentOfInertiaX(self):
        """Calculate moment of inertia around X axis."""
        # Jx = int(y**2*t,du)
        # Jx = 1/3*sum((y[i]**2+y[i]*y[i+1]+y[i+1]**2) * t * delta_u[i],i=[0..len(y)])
        if self.du is None:
            self.calculate_du()
        return np.sum(1/3 * (self.points.y[:-1]**2 + self.points.y[:-1]*self.points.y[1:] + 
                            self.points.y[1:]**2) * self.du) * self.thickness

    def calculateMomentOfInertiaY(self):
        """Calculate moment of inertia around Y axis."""
        # Jy = int(x**2*t,du)
        # Jy = 1/3*sum((x[i]**2+x[i]*x[i+1]+x[i+1]**2) * t * delta_u[i],i=[0..len(x)])
        if self.du is None:
            self.calculate_du()
        return np.sum(1/3 * (self.points.x[:-1]**2 + self.points.x[:-1]*self.points.x[1:] + 
                            self.points.x[1:]**2) * self.du) * self.thickness

    def calculateMomentOfInertiaXY(self):
        """Calculate product moment of inertia."""
        # Jxy = int(x*y*t,du)
        # Jxy = 1/6*sum((2*x[i]*y[i]+x[i]*y[i+1]+x[i+1]*y[i]+2*x[i+1]*y[i+1]) * t * delta_u[i])
        if self.du is None:
            self.calculate_du()
        return np.sum(1/6 * (2*self.points.x[:-1]*self.points.y[:-1] + 
                            self.points.x[:-1]*self.points.y[1:] +
                            self.points.x[1:]*self.points.y[:-1] + 
                            2*self.points.x[1:]*self.points.y[1:]) * 
                     self.du) * self.thickness

    def calculateArea(self):
        """Calculate area of the rod."""
        # A = sum(l[i])*t for t=const
        if self.du is None:
            self.calculate_du()
        self.area = np.sum(self.du) * self.thickness
        return self.area

    def display(self, style=''):
        """Display the rod."""
        plt.plot(self.points.x, self.points.y, style)

    def display_Sx(self):
        """Display static moment around X axis."""
        if self.Sx_u:
            S_u_d = self.transform_for_display_u(self.Sx_u)
            plt.plot(self.u, S_u_d)

    def display_Sy(self):
        """Display static moment around Y axis."""
        if self.Sy_u:
            S_u_d = self.transform_for_display_u(self.Sy_u)
            plt.plot(self.u, S_u_d)

    def transform_for_display_u(self, S_u_d):
        """Transform symbolic function for display."""
        return [float(S_u_d.expr.subs(S_u_d.symbols_dict['u'], u_i)) for u_i in self.u]

    def transformTranslate(self, v) -> None:
        """Translate the rod by vector v."""
        self.points.transformTranslate(v)

    def transformRotate(self, alpha: float, pivot: Point2D = Point2D()) -> None:
        """Rotate the rod around pivot point by angle alpha."""
        self.points.transformRotate(alpha, pivot) 