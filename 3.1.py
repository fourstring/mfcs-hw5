from core import *


def main():
    for name, mat in [('Vandermonde', v_square), ('Fourier', f_square)]:
        for n in [8, 16]:
            c, residual = solve_plu_and_residual(mat(n), anal_vec(n))
            print(f"Solving {name} as coefficients, c={c}, residual={residual}")


if __name__ == '__main__':
    main()
