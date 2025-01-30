# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:04:23 2022

@author: Daniel
"""

class Manipulator:
    def __init__(self):
        pass
    
    def transformTranslate(self, v):
        self.xCoords += v[0]
        self.yCoords += v[1]
        
    