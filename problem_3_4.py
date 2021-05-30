import pandas as pd

from core import *

N = 8


def main():
    df = pd.DataFrame(
        columns=['LU', 'Cholesky'], index=['Vandermonde', 'Fourier']
    )
    for name, mat in [('Vandermonde', v_square), ('Fourier', f_square)]:
        _, residual_lu = solve_plu_and_residual(mat(N), anal_vec(N))
        _, residual_cho = solve_cholesky_and_residual(mat(N).transpose() @ mat(N), mat(N).transpose() @ anal_vec(N))
        df.loc[name] = [residual_lu, residual_cho]

    return df


if __name__ == '__main__':
    main()
