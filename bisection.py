import numpy as np


def bisection(func, deriv, iters, lhs, rhs):
    """

    :param func: Our original function (which takes in a single argument)
     that we are trying to minimize. Argument is an integer.
    :param deriv: The derivative of our initial function (see above).
    Argument is an integer.
    :param iters: An integer representing the number of steps we will take
    towards finding the minima
    :param lhs: the smallest value of x we are considering as the potential minima
    :param rhs: the largest value of x we are considering as the potential minima
    :return: The integer x which yields the minimum output from func, and that supposed
    minimum value.
    """

    for k in range(iters):
        if lhs == rhs:
            return lhs, func(lhs)
        t = (lhs + rhs) / 2
        if deriv(t) > 0:
            rhs = t
        else:
            lhs = t

    loc = (lhs + rhs) / 2
    val = func(loc)
    return "x* = " + str(loc),\
           "f(x*) = " + str(val)


if __name__ == "__main__":
    func = lambda x: x**2
    deriv = lambda x: 2*x
    lhs = -10
    rhs = 10
    print(bisection(func, deriv, 20, lhs, rhs))

