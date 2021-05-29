from core import *


def main():
    for i, (name, mat) in enumerate([('V', v_square), ('F', f_square)], 1):
        plt.subplot(1, 2, i)
        plt.title(f"N vs Cond({name})")
        x = np.arange(4, 34, 2)
        y = [np.log10(np.linalg.cond(mat(n))) for n in x]
        plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
