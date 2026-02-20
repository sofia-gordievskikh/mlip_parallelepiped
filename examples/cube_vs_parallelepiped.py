"""Сравнение куба и параллелепипеда: sweep по tau + картинка.

Строит docs/figures/cube_vs_parallelepiped.png:
- слева допустимая область с вписанным кубом и параллелепипедом;
- справа график center_objective(tau); видно, где куб становится infeasible.

    python examples/cube_vs_parallelepiped.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mcips import inner_box_search, parallelepiped_search  # noqa: E402
from mcips.geometry import Problem, rotation_2d  # noqa: E402
from mcips.visualization import (  # noqa: E402
    plot_box_2d,
    plot_parallelepiped_2d,
    plot_region_2d,
)

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"


def problem() -> Problem:
    return Problem(
        c=[1.0, 1.0],
        A=[[1, -1], [-1, 1], [1, 1]],
        b=[2.0, 0.0, 10.0],
        l_bounds=[0, 0],
        u_bounds=[6, 6],
        int_idx=[0],
    )


def sweep(p: Problem, basis, taus):
    box_obj, par_obj = [], []
    for tau in taus:
        b = inner_box_search(p, tau=tau)
        pr = parallelepiped_search(p, basis, tau=tau)
        box_obj.append(b["center_objective"] if b["status"] == "ok" else np.nan)
        par_obj.append(pr["center_objective"] if pr["status"] == "ok" else np.nan)
    return np.array(box_obj), np.array(par_obj)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p = problem()
    basis = rotation_2d(np.radians(45))
    tau_show = 1.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    plot_region_2d(ax1, p)
    b = inner_box_search(p, tau=tau_show)
    pr = parallelepiped_search(p, basis, tau=tau_show)
    if b["status"] == "ok":
        plot_box_2d(ax1, b["l"], b["u"])
    if pr["status"] == "ok":
        plot_parallelepiped_2d(ax1, pr["center"], basis, pr["ext"])
    ax1.set_aspect("equal")
    ax1.set_title(f"допустимая область, tau={tau_show}")
    ax1.legend(loc="lower right")

    taus = np.linspace(0.2, 2.6, 25)
    box_obj, par_obj = sweep(p, basis, taus)
    ax2.plot(taus, box_obj, "-o", ms=3, color="#e67e22", label="cube")
    ax2.plot(taus, par_obj, "-o", ms=3, color="#8e44ad", label="parallelepiped")
    ax2.set_xlabel("tau (требуемая ширина)")
    ax2.set_ylabel("center objective  c·center")
    ax2.set_title("objective vs tau (NaN = infeasible)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("Inner box search: cube vs parallelepiped")
    fig.tight_layout()
    out = FIG / "cube_vs_parallelepiped.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
