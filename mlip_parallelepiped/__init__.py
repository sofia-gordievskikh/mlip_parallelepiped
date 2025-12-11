"""Inner Box Search для MILP: вписываем коробку/параллелепипед в допустимую
область LP-релаксации и берём целочисленный угол как кандидата.

Публичный API:
    solve_box_lp, inner_box_search   - axis-aligned коробка (базовый случай);
    solve_parallelepiped, parallelepiped_search - параллелепипед (повёрнутый базис).
"""
from .geometry import Problem, box_vertices, parallelepiped_vertices
from .solver import (
    build_candidate,
    check_feasible,
    inner_box_search,
    parallelepiped_search,
    solve_box_lp,
    solve_parallelepiped,
)

__all__ = [
    "Problem",
    "box_vertices",
    "parallelepiped_vertices",
    "solve_box_lp",
    "solve_parallelepiped",
    "inner_box_search",
    "parallelepiped_search",
    "build_candidate",
    "check_feasible",
]
