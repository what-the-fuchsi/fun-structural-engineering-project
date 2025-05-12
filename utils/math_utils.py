from enum import Enum
import numpy as np

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


class Axis(Enum):
    X = 'x'
    Y = 'y'
    XY = 'xy'