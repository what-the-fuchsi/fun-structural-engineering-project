# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:24:04 2022

@author: Daniel
"""

from straightRod import StraightRod
from geom import Point2D, PointCloud
from factory import ShapeFactory
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:


    structure = ShapeFactory.T_Profile2(20, 80, 2)
    # structure.transformRotate(-np.pi/8)
    # structure.display()

    structure.calculateArea()
    structure.calculateStaticMomentX()
    structure.calculateStaticMomentY()
    structure.calculateCenterOfMass()

    structure.calculate_static_moment_distr()
    # print(structure.center_of_mass.x, structure.center_of_mass.y)


    structure.display()
    # plt.plot(structure.center_of_mass.x, structure.center_of_mass.y, 'gx')
    # plt.show()
    # print(structure.calculateMomentOfInertiaX())
    # print(structure.calculateMomentOfInertiaY())
    # print(structure.calculateMomentOfInertiaXY())
    for member in structure.members:
        member.display_Sx()
    plt.show()

if __name__ == '__main__':
    main()
