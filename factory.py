# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:41:07 2022

@author: Daniel
"""
from geom import *
from straightRod import *
import numpy as np


class ShapeFactory:
    
    @staticmethod
    def Z_Profile(b, h, t):
        point1 = Point2D(-b, h/2, is_free=True)
        point2 = Point2D(0, h/2)
        point3 = Point2D(0, -h/2)
        point4 = Point2D(b, -h/2, is_free=True)
        
        pointCloud1 = PointCloud()
        pointCloud1.addPoints(point1, point2)
        rod1 = StraightRod(pointCloud1, t)
        
        pointCloud2 = PointCloud()
        pointCloud2.addPoints(point2, point3)
        rod2 = StraightRod(pointCloud2, t)
        
        pointCloud3 = PointCloud()
        pointCloud3.addPoints(point3, point4)
        rod3 = StraightRod(pointCloud3, t)
        
        structure = Structure(name="Z-Profile")
        for rod in [rod1, rod2, rod3]:
            structure.add_member(rod)
            
        return structure

    @staticmethod
    def C_Profile(b, h, t):
        point1 = Point2D(b, h/2, is_free=True)
        point2 = Point2D(0, h/2)
        point3 = Point2D(0, -h/2)
        point4 = Point2D(b, -h/2, is_free=True)
        point_list = [point1, point2, point3, point4]

        structure = Structure(name="C-Profile")
        for i in range(3):
            point_cloud = PointCloud()
            point_cloud.addPoints(point_list[i], point_list[i+1])
            rod = StraightRod(point_cloud, t)
            structure.add_member(rod)

        return structure

    @staticmethod
    def T_Profile(b, h, t):
        # not valid!
        point1 = Point2D(-b, 0, is_free=True)
        point2 = Point2D(0, 0)
        point3 = Point2D(b, 0, is_free=True)
        point4 = Point2D(0, -h, is_free=True)

        pc1 = PointCloud()
        pc1.addPoints(point1, point3)
        rod1 = StraightRod(pc1, t)

        pc2 = PointCloud()
        pc2.addPoints(point2, point4)
        rod2 = StraightRod(pc2, t)

        structure = Structure(members=list((rod1, rod2)), name="T-Profile")
        return structure

    @staticmethod
    def T_Profile2(b, h, t):
        point1 = Point2D(-b, 0, is_free=True)
        point2 = Point2D(0, 0)
        point3 = Point2D(b, 0, is_free=True)
        point4 = Point2D(0, -h, is_free=True)

        pc1 = PointCloud()
        pc1.addPoints(point1, point2)
        rod1 = StraightRod(pc1, t, name='top left horz bar')

        pc3 = PointCloud()
        pc3.addPoints(point2, point3)
        rod3 = StraightRod(pc3, t, name='top right horz bar')

        pc2 = PointCloud()
        pc2.addPoints(point2, point4)
        rod2 = StraightRod(pc2, t, name='vertical bar')

        structure = Structure(members=list((rod1, rod3, rod2)), name="T-Profile")
        return structure
