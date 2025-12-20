"""CLI: решить задачу из YAML-конфига коробкой и параллелепипедом.

    python -m mlip_parallelepiped.solve --config configs/demo.yaml
    python -m mlip_parallelepiped.solve --config configs/demo.yaml --plot out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .geometry import Problem, identity_basis, rotation_2d
from .solver import inner_box_search, parallelepiped_search


def load_problem(cfg: dict) -> Problem:
    return Problem(
        c=cfg["c"],
        A=cfg["A"],
        b=cfg["b"],
        l_bounds=cfg["l_bounds"],
        u_bounds=cfg["u_bounds"],
        int_idx=cfg.get("int_idx", []),
    )


def build_basis(cfg: dict, n: int) -> np.ndarray:
    par = cfg.get("parallelepiped", {})
    if "matrix" in par:
        return np.asarray(par["matrix"], dtype=float)
    if "angle_deg" in par and n == 2:
        return rotation_2d(np.radians(par["angle_deg"]))
    return identity_basis(n)


def run(cfg: dict) -> dict:
    problem = load_problem(cfg)
    tau = float(cfg.get("tau", 1.0))
    basis = build_basis(cfg, problem.n)

    box = inner_box_search(problem, tau=tau)
    par = parallelepiped_search(problem, basis, tau=tau)
    return {"problem": problem, "tau": tau, "basis": basis, "box": box, "parallelepiped": par}


def _fmt(res: dict) -> str:
    if res.get("status") != "ok":
        return f"  status: {res.get('status')}"
    cand = np.round(res["candidate"], 3) if res.get("candidate") is not None else None
    return (
        f"  status: ok\n"
        f"  center_objective: {res['center_objective']:.4f}\n"
        f"  candidate: {cand}  (feasible={res['candidate_feasible']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="inner box / parallelepiped search")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--plot", default=None, help="сохранить 2D-сравнение в файл")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = run(cfg)

    print(f"config: {args.config}")
    print(f"dim={out['problem'].n}  tau={out['tau']}  int_idx={out['problem'].int_idx}")
    print("CUBE (axis-aligned box):")
    print(_fmt(out["box"]))
    print("PARALLELEPIPED (rotated basis):")
    print(_fmt(out["parallelepiped"]))

    if out["box"].get("status") == "ok" and out["parallelepiped"].get("status") == "ok":
        gain = out["parallelepiped"]["center_objective"] - out["box"]["center_objective"]
        print(f"objective gain (parallelepiped - cube): {gain:+.4f}")

    if args.plot:
        _plot(out, args.plot)
        print(f"saved plot to {args.plot}")


def _plot(out: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .visualization import plot_box_2d, plot_parallelepiped_2d, plot_region_2d

    problem = out["problem"]
    if problem.n != 2:
        raise SystemExit("--plot поддержан только для 2D задач")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_region_2d(ax, problem)
    if out["box"]["status"] == "ok":
        plot_box_2d(ax, out["box"]["l"], out["box"]["u"])
    if out["parallelepiped"]["status"] == "ok":
        p = out["parallelepiped"]
        plot_parallelepiped_2d(ax, p["center"], out["basis"], p["ext"])
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"cube vs parallelepiped, tau={out['tau']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
