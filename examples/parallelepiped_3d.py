"""3D-визуализация: допустимая коробка ограничений, вписанный axis-aligned box
и повёрнутый параллелепипед.

Строит docs/figures/parallelepiped_3d.png.

    python examples/parallelepiped_3d.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from mlip_parallelepiped.geometry import Problem, identity_basis  # noqa: E402
from mlip_parallelepiped.solver import parallelepiped_search  # noqa: E402
from mlip_parallelepiped.visualization import (  # noqa: E402
    parallelepiped_vertices,
    plot_box_3d,
    set_axes_equal,
)

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    problem = Problem(
        c=[3.0, 2.0, 1.0],
        A=[[2, 1, 1], [1, 2, 0], [0, 1, 2]],
        b=[4.5, 4.0, 3.5],
        l_bounds=[0, 0, 0],
        u_bounds=[3, 3, 3],
        int_idx=[0, 1],
    )
    # gamma > 0 даёт коробкам объём, чтобы 3D-картинка читалась (иначе непрерывные
    # оси схлопываются в точку). Это то же inner box search, но с наградой за ширину.
    tau, gamma = 0.6, 0.3
    box = parallelepiped_search(problem, identity_basis(3), tau=tau, gamma=gamma)
    basis = rot_z(np.radians(25))
    par = parallelepiped_search(problem, basis, tau=tau, gamma=gamma)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # каркас глобальных границ
    plot_box_3d(ax, problem.l_bounds, problem.u_bounds, color="#e8edf7", alpha=0.06, edge="#b7c2d8")

    if box["status"] == "ok":
        l = box["center"] - box["ext"]
        u = box["center"] + box["ext"]
        plot_box_3d(ax, l, u, color="#c9d94d", alpha=0.45, edge="#7c8f00")
        ax.scatter(*box["center"], color="#5c7a00", s=25, label="cube center")

    if par["status"] == "ok":
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        v = parallelepiped_vertices(par["center"], basis, par["ext"])
        faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                 [2, 3, 7, 6], [1, 3, 7, 5], [0, 2, 6, 4]]
        ax.add_collection3d(Poly3DCollection(
            [v[f] for f in faces], facecolors="#b39ddb", edgecolors="#5e35b1",
            linewidths=1.1, alpha=0.5))
        ax.scatter(*par["center"], color="#5e35b1", s=25, label="parallelepiped center")

    ax.set_title("Вписанные коробка (зелёная) и параллелепипед (фиолетовый)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.view_init(elev=20, azim=-60)
    set_axes_equal(ax)
    ax.legend(loc="upper left")

    out = FIG / "parallelepiped_3d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print("cube center_obj:", round(box["center_objective"], 3))
    print("parallelepiped center_obj:", round(par["center_objective"], 3))


if __name__ == "__main__":
    main()
