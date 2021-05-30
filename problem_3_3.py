import pandas as pd

from core import *


def main():
    df = pd.DataFrame(
        np.array([[
            n,
            is_pos_def(v_square(n).transpose() @ v_square(n)),
            is_pos_def(f_square(n).transpose() @ f_square(n)),
            np.linalg.cond(v_square(n)),
            np.linalg.cond(f_square(n))
        ] for n in range(4, 34, 2)]),
        columns=['N', 'isposdef(A_v)', 'isposdef(A_f)', 'cond(V)', 'cond(F)']
    )
    print(df)


if __name__ == '__main__':
    main()
