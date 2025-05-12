from __future__ import annotations
from enum import Enum
from typing import List, TYPE_CHECKING

from geom import Point2D

if TYPE_CHECKING:
    from structures.base import Structure

class JunctionType(Enum):
    CONT = 1  # Continuous
    CONV = 2  # Convergent
    DIV = 3   # Divergent

class Junction:
    def __init__(self, point: Point2D = None, members: list[Structure] = None):
        self.point: Point2D = point
        self.members: list[Structure] = members if members is not None else []
        self.type: JunctionType = None
        self.resolve_members()
        point.is_junction = True

    def resolve_members(self):
        """Determine the type of junction based on member connections."""
        if not self.members:
            return

        # Count incoming and outgoing members
        incoming = 0
        outgoing = 0
        
        for member in self.members:
            # Check if point is start or end of member
            if self.point == member.points.first_point():
                incoming += 1
            elif self.point == member.points.last_point():
                outgoing += 1

        # Determine junction type
        if incoming == 1 and outgoing == 1:
            self.type = JunctionType.CONT
        elif incoming > outgoing:
            self.type = JunctionType.CONV
        else:
            self.type = JunctionType.DIV

    def add_member(self, member: Structure):
        """Add a member to the junction and update its type."""
        if member not in self.members:
            self.members.append(member)
            self.resolve_members() 