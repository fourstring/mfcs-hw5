from core import *


def main():
    for name, solve in [('Vandermonde', solve_v_qr), ('Fourier', solve_f_qr)]:
        for n in [4, 8]:
            print(f"{name},M=16,N={n}, c={solve(16, n)}")


if __name__ == '__main__':
    main()
