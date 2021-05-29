import operator
from functools import lru_cache
from typing import Tuple, Callable

import numpy as np
import numpy.linalg
import scipy.linalg
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import stats

SolverT = Callable[[np.ndarray, np.ndarray], np.ndarray]
RootResidualT = Tuple[np.ndarray, float]


def plot_normal_distribution(means: float, variance: float, color: str):
    sigma = np.sqrt(variance)
    x = np.linspace(means - 3 * sigma, means + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, means, sigma), color=color)


def integrate_normal_distribution(means: float, variance: float) -> float:
    sigma = np.sqrt(variance)
    y, *_ = integrate.quad(lambda x: stats.norm.pdf(x, means, sigma), means - 3 * sigma, means + 3 * sigma)
    return y


def anal_f(x: float) -> float:
    return 1. / (1. + np.power(x, 2))


def interpolate_x(n: int, start: float = 0., end: float = 1.):
    cur = start
    count = 1
    step = (end - start) / (n - 1)
    while count < n + 1:
        yield cur
        cur += step
        count += 1


def x_powers(x: float):
    cur = 1.
    while True:
        yield cur
        cur *= x


@lru_cache(maxsize=16)
def anal_vec(n: int) -> np.ndarray:
    return np.array([anal_f(x) for x in interpolate_x(n)])


@lru_cache(maxsize=16)
def v_mat(m: int, n: int) -> np.ndarray:
    return np.array([[x_power for x_power, _ in zip(x_powers(x), range(n))] for x in interpolate_x(m)])


@lru_cache(maxsize=16)
def v_square(n: int) -> np.ndarray:
    return v_mat(n, n)


@lru_cache(maxsize=16)
def f_mat(m: int, n: int) -> np.ndarray:
    return np.array([
        [np.sin(j * x * np.pi) if j <= n / 2 else np.cos((j - n / 2) * x * np.pi)
         for j in range(1, n + 1)]
        for x in interpolate_x(m)])


@lru_cache(maxsize=16)
def f_square(n: int) -> np.ndarray:
    return f_mat(n, n)


def solve_and_residual(a_mat: np.ndarray, b: np.ndarray, solver: SolverT) -> RootResidualT:
    x = solver(a_mat, b)
    return x, np.linalg.norm(a_mat @ x - b)


def solve_plu(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
    p_mat, l_mat, u_mat = scipy.linalg.lu(a_mat)
    z = np.linalg.solve(p_mat, b)
    y = np.linalg.solve(l_mat, z)
    return np.linalg.solve(u_mat, y)


def solve_plu_and_residual(a_mat: np.ndarray, b: np.ndarray) -> RootResidualT:
    return solve_and_residual(a_mat, b, solve_plu)


def is_pos_def(a_mat: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(a_mat) > 0)


def solve_cholesky(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
    l_mat = np.linalg.cholesky(a_mat)
    y = np.linalg.solve(l_mat, b)
    return np.linalg.solve(l_mat.transpose(), y)


def solve_cholesky_and_residual(a_mat: np.ndarray, b: np.ndarray) -> RootResidualT:
    return solve_and_residual(a_mat, b, solve_cholesky)


def solve_qr(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
    q_mat, r_mat = np.linalg.qr(a_mat)
    return np.linalg.solve(r_mat, q_mat.transpose() @ b)


@lru_cache(maxsize=16)
def solve_v_qr(m: int, n: int) -> np.ndarray:
    return solve_qr(v_mat(m, n), anal_vec(m))


@lru_cache(maxsize=16)
def solve_f_qr(m: int, n: int) -> np.ndarray:
    return solve_qr(f_mat(m, n), anal_vec(m))


def interpolate_g_v(c: np.ndarray) -> Callable[[float], float]:
    def g_v(x: float) -> float:
        return sum(map(operator.mul, c, x_powers(x)))

    return g_v


def interpolate_g_f(c: np.ndarray) -> Callable[[float], float]:
    def g_f(x: float) -> float:
        N = c.shape[0]
        return sum(c_j * np.sin(j * np.pi * x) if j <= N / 2 else c_j * np.cos((j - N / 2) * np.pi * x) for j, c_j in
                   enumerate(c, 1))

    return g_f
