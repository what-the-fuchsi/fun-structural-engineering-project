from geom import *

func = SymbolicFunction('a*x**2+b*x+c', 'x a b c', a=1, b=2, c=-3)
print(func.factor())
