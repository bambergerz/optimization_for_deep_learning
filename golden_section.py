import numpy as np


def golden_section(func, a, d, error_range=0.0001):
    """

    :param func: the function we are trying to minimize
    :param a: the minimum point from which we are looking for the minima
    :param d: the maximum point to which we are looking for the minima.
    :return: the minima, and the value of the function at the minima.
    """

    tau = (3 - (5**0.5))/2

    if np.linalg.norm(d-a) < error_range:
        return a, func(a)
    else:
        delta = d - a
        b = a + (tau * delta)
        c = d - (tau * delta)
        if func(b) < func(c):
            return golden_section(func, a, c)
        else:
            return golden_section(func, b, d)


if __name__ == "__main__":
    func = lambda x: np.power(x, 2)
    a = np.array([-10])
    d = np.array([10])
    print(golden_section(func, a, d))
