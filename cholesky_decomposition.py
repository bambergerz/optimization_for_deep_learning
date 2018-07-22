import numpy as np

def cholesky(G):
    """

    :param G: a symetric matrix, G
    :return: a vector e of "small" norm and lower triangular matrix L,
    and vector d such that G + Tr(e) is positive definite, and

            G + Tr(e) = L * Tr(d) * L

    Also, calculating pneg such that if G is not positive semi definite, then

            pneg.T * G * pneg < 0
    """

    m, n = np.shape(G)

    """
    gamma, zi, nu, and beta2 are quantities used by the algorithm
    """

    gamma = max(np.trace(G))
    zi = max(max(G - np.trace((np.trace(G)))))
    nu = max([1, pow(n**2-1, 0.5)])
    beta2 = max([gamma, zi/nu, 1e-15])

    c = np.trace(np.trace(G))



