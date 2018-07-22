import numpy as np

from golden_section import golden_section
from bisection import bisection

ONE_DIMENSIONAL_OPTIMIZATION_FUNCS = [golden_section, bisection]


def exact_line_search(func, grad, x_initial, step_func=golden_section, iters=1_000):
    """

    :param func: the function we are trying to minimize
    :param grad: the gradient of the function we are trying to minimize
    :param x_initial: the initial value we think our min will be
    :param step_func: optional argument. A function used to determine the step size
    at a given iteration
    :param iters: the number of iterations we will take.
    :return: the minima of func and its value at that point.
    """

    assert step_func in ONE_DIMENSIONAL_OPTIMIZATION_FUNCS, \
        "Choose valid function to determine step size"

    x = x_initial

    for _ in range(iters):
        direction = grad(x)

        phi = lambda alpha: func(x + np.dot(alpha, grad(x)))

        if step_func == golden_section:
            step_size = step_func(func=phi, a=10**-7, d=10**2)

        elif step_func == bisection:
            step_size = step_func(func=phi, deriv=grad, iters=1000, lhs=x_initial, rhs=x_initial+100)
        else:
            raise NotImplemented

        x_initial = x_initial - np.dot(step_size, direction)

    return x_initial, func(x_initial)


def armijo(func, grad, direction, x_initial, alpha_initial=1, beta=0.2, sigma=10**(-4), eta=10**(-5)):
    """

    vector, point, objective function

    :param func: the function we are trying to minimize
    :param grad: the gradient of the function we are trying to minimize
    :param direction: the direction of descent. a vector
    :param x_initial: our initial guess for the minima of func
    :param alpha_initial: the initial value of our step size. Default is 1
    :param beta: the magnitude of our step back from initial step size
     to ideal step size according to armijo's law.
     Default is 0.2
    :param sigma: the multiplier of the slope of func according to step size.
    The greater this value, the larger our step size is likely to be.
    Default is 10^-4.
    :param eta: the stopping criterial. We stop when the norm of the gradient is less
    than this value.
    :return: the minima of func, and the value of func at that point
    """
    x_k = x_initial
    alpha_k = alpha_initial
    phi = lambda alpha: func(x_k + np.dot(alpha, grad(x_k)))

    while np.linalg.norm(grad(x_k)) >= eta:

        c = np.dot(grad(x_k), direction)

        while phi(alpha_k) - func(x_k) >= sigma * c * alpha_k:
            alpha_k = alpha_k * beta

    return x_k, func(x_k)
