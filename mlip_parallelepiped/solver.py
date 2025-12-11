"""Inner Box Search: коробка и параллелепипед.

Идея метода (см. docs/notes.md):
1. решаем вспомогательную LP, которая вписывает широкую коробку/параллелепипед
   в допустимую область LP-релаксации и двигает её центр вдоль c;
2. берём целочисленный угол этой области как кандидата MILP-решения;
3. проверяем кандидата на допустимость.

`solve_box_lp` - базовый axis-aligned случай (перенесён из ноутбука).
`solve_parallelepiped` - обобщение на повёрнутый базис.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog

from .geometry import Problem, parallelepiped_vertices


# --------------------------------------------------------------------- box (cube)
def solve_box_lp(problem: Problem, tau: float = 1.0, method: str = "highs"):
    """Вписать axis-aligned коробку [l, u] в допустимую область.

    Переменные: l (n) и u (n). Максимизируем c·(l+u) (тянем коробку вдоль c).
    Для целочисленных переменных требуем ширину коробки >= tau, чтобы внутри
    гарантированно нашёлся целый уровень.

    Возвращает (l, u, objective) или None, если LP несовместна.
    """
    c, A, b = problem.c, problem.A, problem.b
    lb, ub, int_idx = problem.l_bounds, problem.u_bounds, problem.int_idx
    n = problem.n
    m = 2 * n

    obj = np.zeros(m)
    obj[:n] = -c
    obj[n:] = -c

    A_ub, b_ub = [], []

    # l_i <= u_i
    for i in range(n):
        row = np.zeros(m)
        row[i] = 1.0
        row[n + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # ширина по целочисленным осям: u_i - l_i >= tau
    for i in int_idx:
        row = np.zeros(m)
        row[i] = 1.0
        row[n + i] = -1.0
        A_ub.append(row)
        b_ub.append(-tau)

    # опорная функция коробки: max_{x in box} a_k·x <= b_k
    for k in range(len(b)):
        row = np.zeros(m)
        for i in range(n):
            a = A[k, i]
            if a >= 0:
                row[n + i] += a
            else:
                row[i] += a
        A_ub.append(row)
        b_ub.append(b[k])

    bounds = [(lb[i], ub[i]) for i in range(n)] * 2

    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method=method)
    if not res.success:
        return None
    l = res.x[:n]
    u = res.x[n:]
    return l, u, -res.fun


def build_candidate(problem: Problem, l: np.ndarray, u: np.ndarray) -> Optional[np.ndarray]:
    """Целочисленный угол коробки: для целых переменных берём floor(u)/ceil(l)
    в сторону роста c, для непрерывных - соответствующую границу."""
    c, n = problem.c, problem.n
    int_set = set(problem.int_idx)
    x = np.zeros(n)
    for i in range(n):
        if i in int_set:
            x[i] = np.floor(u[i]) if c[i] >= 0 else np.ceil(l[i])
            if x[i] < l[i] - 1e-9 or x[i] > u[i] + 1e-9:
                return None
        else:
            x[i] = u[i] if c[i] >= 0 else l[i]
    return x


def check_feasible(problem: Problem, x: np.ndarray, tol: float = 1e-7) -> bool:
    return bool(np.all(problem.A @ x <= problem.b + tol))


def inner_box_search(problem: Problem, tau: float = 1.0) -> dict:
    sol = solve_box_lp(problem, tau=tau)
    if sol is None:
        return {"status": "infeasible_box_lp"}
    l, u, obj = sol
    x = build_candidate(problem, l, u)
    feasible = x is not None and check_feasible(problem, x)
    return {
        "status": "ok",
        "shape": "box",
        "l": l,
        "u": u,
        "center": (l + u) / 2,
        "candidate": x,
        "candidate_feasible": feasible,
        "objective": obj,
        "center_objective": float(problem.c @ ((l + u) / 2)),
    }


# --------------------------------------------------------------- parallelepiped
def solve_parallelepiped(
    problem: Problem,
    basis: np.ndarray,
    gamma: float = 0.0,
    tau: float = 0.0,
    method: str = "highs",
):
    """Вписать параллелепипед с фиксированным базисом `basis`.

    Область: {center + basis @ t : t in [-ext, ext]}. Переменные - center (n) и
    полудлины ext (n) >= 0. Максимизируем c·center + gamma·sum(ext).

    Так как базис фиксирован, коэффициенты |a_k·basis_i| - константы, поэтому
    условие «все вершины допустимы» линейно:

        a_k·center + sum_i |a_k·basis_i| * ext_i <= b_k

    Аналогично глобальные границы [l_bounds, u_bounds] по каждой координате.
    При basis = I это ровно опорные ограничения из solve_box_lp.

    `tau` - минимальная ширина области вдоль каждой целочисленной координаты в
    исходном пространстве x (ширина = 2 * sum_j |basis[i,j]| ext_j). Она гарантирует,
    что внутри есть хотя бы один целый уровень. Именно здесь параллелепипед выигрывает
    у куба: в наклонной области требуемую ширину дешевле набрать «вдоль» области, а
    не по осям координат.

    Возвращает (center, ext, objective) или None.
    """
    c, A, b = problem.c, problem.A, problem.b
    lb, ub = problem.l_bounds, problem.u_bounds
    n = problem.n
    B = np.asarray(basis, dtype=float)

    # W[k, i] = |a_k · basis_i|
    AB = A @ B                      # (m, n)
    W = np.abs(AB)
    absB = np.abs(B)               # для глобальных границ по координатам

    obj = np.concatenate([-c, -gamma * np.ones(n)])

    A_ub, b_ub = [], []

    # вершинная допустимость по каждому ограничению
    for k in range(len(b)):
        A_ub.append(np.concatenate([A[k], W[k]]))
        b_ub.append(b[k])

    # параллелепипед внутри глобальной коробки [lb, ub] по каждой координате j
    for j in range(n):
        # center_j + sum_i |B[j,i]| ext_i <= ub_j
        A_ub.append(np.concatenate([_unit(n, j), absB[j]]))
        b_ub.append(ub[j])
        # -center_j + sum_i |B[j,i]| ext_i <= -lb_j
        A_ub.append(np.concatenate([-_unit(n, j), absB[j]]))
        b_ub.append(-lb[j])

    # минимальная ширина tau вдоль целочисленных координат (в пространстве x):
    #   2 * sum_j |basis[i,j]| ext_j >= tau
    if tau > 0:
        for i in problem.int_idx:
            row = np.zeros(2 * n)
            row[n:] = -2.0 * absB[i]
            A_ub.append(row)
            b_ub.append(-tau)

    bounds = [(None, None)] * n + [(0.0, None)] * n

    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method=method)
    if not res.success:
        return None
    center = res.x[:n]
    ext = res.x[n:]
    return center, ext, float(c @ center)


def parallelepiped_search(
    problem: Problem,
    basis: np.ndarray,
    gamma: float = 0.0,
    tau: float = 0.0,
) -> dict:
    sol = solve_parallelepiped(problem, basis, gamma=gamma, tau=tau)
    if sol is None:
        return {"status": "infeasible_parallelepiped_lp"}
    center, ext, obj = sol
    B = np.asarray(basis, dtype=float)

    # угол параллелепипеда, максимизирующий c: сдвигаемся по знаку (B^T c)
    signs = np.sign(B.T @ problem.c)
    signs[signs == 0] = 1.0
    corner = center + B @ (signs * ext)

    # округляем целочисленные координаты угла внутрь и проверяем допустимость
    x = corner.copy()
    for i in problem.int_idx:
        x[i] = np.floor(corner[i]) if problem.c[i] >= 0 else np.ceil(corner[i])
    feasible = check_feasible(problem, x)

    return {
        "status": "ok",
        "shape": "parallelepiped",
        "center": center,
        "ext": ext,
        "vertices": parallelepiped_vertices(center, B, ext),
        "candidate": x,
        "candidate_feasible": feasible,
        "objective": obj,
        "center_objective": float(problem.c @ center),
    }


def _unit(n: int, j: int) -> np.ndarray:
    e = np.zeros(n)
    e[j] = 1.0
    return e
