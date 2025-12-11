"""Геометрия: описание задачи и вершины коробки / параллелепипеда."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Problem:
    """MILP-подобная задача для inner box search.

    Максимизируем c·x на многограннике {x : A x <= b, l_bounds <= x <= u_bounds},
    часть переменных (int_idx) должна быть целой. Метод вписывает коробку или
    параллелепипед в допустимую область и берёт целочисленный угол как кандидата.
    """

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    l_bounds: np.ndarray
    u_bounds: np.ndarray
    int_idx: Sequence[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.l_bounds = np.asarray(self.l_bounds, dtype=float)
        self.u_bounds = np.asarray(self.u_bounds, dtype=float)
        self.int_idx = list(self.int_idx)

    @property
    def n(self) -> int:
        return len(self.c)

    def contains(self, x: np.ndarray, tol: float = 1e-7) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(
            np.all(self.A @ x <= self.b + tol)
            and np.all(x >= self.l_bounds - tol)
            and np.all(x <= self.u_bounds + tol)
        )


def box_vertices(l: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Все 2^n вершин axis-aligned коробки [l, u]."""
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    n = len(l)
    verts = []
    for mask in itertools.product([0, 1], repeat=n):
        verts.append([u[i] if mask[i] else l[i] for i in range(n)])
    return np.array(verts)


def parallelepiped_vertices(center: np.ndarray, basis: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Вершины параллелепипеда {center + basis @ (s * ext) : s in {-1,+1}^n}.

    `basis` - матрица n x n, столбцы задают направления рёбер; `ext` - полудлины
    рёбер вдоль этих направлений. При basis = I получаем обычную коробку.
    """
    center = np.asarray(center, dtype=float)
    basis = np.asarray(basis, dtype=float)
    ext = np.asarray(ext, dtype=float)
    n = len(center)
    verts = []
    for signs in itertools.product([-1, 1], repeat=n):
        t = np.array(signs, dtype=float) * ext
        verts.append(center + basis @ t)
    return np.array(verts)


def rotation_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def identity_basis(n: int) -> np.ndarray:
    return np.eye(n)
