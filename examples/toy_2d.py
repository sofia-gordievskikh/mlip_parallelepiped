"""2D toy case: наклонная полоса, куб vs параллелепипед.

    python examples/toy_2d.py
"""
import numpy as np

from mlip_parallelepiped import inner_box_search, parallelepiped_search
from mlip_parallelepiped.geometry import Problem, rotation_2d


def main() -> None:
    problem = Problem(
        c=[1.0, 1.0],
        A=[[1, -1], [-1, 1], [1, 1]],
        b=[2.0, 0.0, 10.0],
        l_bounds=[0, 0],
        u_bounds=[6, 6],
        int_idx=[0],
    )
    tau = 1.5
    box = inner_box_search(problem, tau=tau)
    par = parallelepiped_search(problem, rotation_2d(np.radians(45)), tau=tau)

    print(f"tau = {tau}")
    print("cube:", box["status"], "center_obj=",
          round(box.get("center_objective", float("nan")), 3) if box["status"] == "ok" else "-")
    print("parallelepiped:", par["status"], "center_obj=",
          round(par["center_objective"], 3), "candidate=", np.round(par["candidate"], 2))


if __name__ == "__main__":
    main()
