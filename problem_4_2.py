from matplotlib.cm import get_cmap

from core import *


def main():
    x = np.linspace(0, 1, num=100)
    b = anal_vec(16)
    interpolated_funcs = {
        'analytic_f': anal_f,
        'g_v_16_16': interpolate_g_v(solve_plu(v_square(16), b)),
        'g_f_16_16': interpolate_g_f(solve_plu(f_square(16), b)),
        'g_v_16_4': interpolate_g_v(solve_v_qr(16, 4)),
        'g_f_16_4': interpolate_g_f(solve_f_qr(16, 4)),
        'g_v_16_8': interpolate_g_v(solve_v_qr(16, 8)),
        'g_f_16_8': interpolate_g_f(solve_f_qr(16, 8)),
    }
    cmap_name = "Dark2"
    cmap = get_cmap(cmap_name)
    colors = cmap.colors
    fig = plt.figure()
    ax0 = fig.add_subplot(421)
    ax0.set_prop_cycle(color=colors)
    for name, f in interpolated_funcs.items():
        ys = [f(x_i) for x_i in x]
        ax0.plot(x, ys, label=name)

    for i, (name, f) in enumerate(interpolated_funcs.items(), 2):
        ax = fig.add_subplot(4, 2, i)
        ys = [f(x_i) for x_i in x]
        ax.plot(x, ys)
        ax.set_title(name)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
