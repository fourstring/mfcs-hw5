import pandas as pd

from core import *


def main():
    df = pd.DataFrame(
        columns=['c', 'residual']
    )
    for name, mat in [('Vandermonde', v_square), ('Fourier', f_square)]:
        for n in [8, 16]:
            c, residual = solve_plu_and_residual(mat(n), anal_vec(n))
            df.loc[f"{name},N={n}"] = [c, residual]
    return df


if __name__ == '__main__':
    main()
