from core import *

N = 8


def main():
    for name, mat in [('Vandermonde', v_square), ('Fourier', f_square)]:
        c_lu, residual_lu = solve_plu_and_residual(mat(N), anal_vec(N))
        print(f"{name}+LU: c={c_lu}, residual={residual_lu}")
        c_cho, residual_cho = solve_cholesky_and_residual(mat(N).transpose() @ mat(N), mat(N).transpose() @ anal_vec(N))
        print(f"{name}+Cholesky: c={c_cho}, residual={residual_cho}")


if __name__ == '__main__':
    main()
