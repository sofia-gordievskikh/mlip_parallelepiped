"""MCIPS - Maximized-Center Inscribed Parallelepiped Search.

Эвристика построения допустимых решений MILP. В допустимую область
LP-релаксации вписывается тело простой формы с максимизированным вдоль c
центром; целочисленная вершина этого тела берётся как кандидат решения.

Два варианта вписываемого тела:
    solve_box_lp, inner_box_search   - axis-aligned коробка (базовый случай, B = I);
    solve_parallelepiped, parallelepiped_search - параллелепипед (аффинный базис B).
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
