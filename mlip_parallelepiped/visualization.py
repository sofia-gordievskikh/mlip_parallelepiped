"""Визуализация допустимой области и вписанных коробки / параллелепипеда."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import Problem, box_vertices, parallelepiped_vertices


# ----------------------------------------------------------------- 2d helpers
def feasible_polygon_2d(problem: Problem) -> np.ndarray:
    """Многоугольник допустимой области в 2D через клиппинг Сазерленда-Ходжмана.

    Стартуем с прямоугольника глобальных границ и последовательно режем каждым
    полупространством a·x <= b. Возвращает упорядоченные вершины (по контуру).
    """
    lb, ub = problem.l_bounds, problem.u_bounds
    poly = [
        np.array([lb[0], lb[1]]),
        np.array([ub[0], lb[1]]),
        np.array([ub[0], ub[1]]),
        np.array([lb[0], ub[1]]),
    ]
    for a, bk in zip(problem.A, problem.b):
        poly = _clip(poly, a, bk)
        if not poly:
            break
    return np.array(poly)


def _clip(poly, a, bk):
    if not poly:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cur_in = a @ cur <= bk + 1e-9
        nxt_in = a @ nxt <= bk + 1e-9
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            t = (bk - a @ cur) / (a @ (nxt - cur) + 1e-15)
            out.append(cur + t * (nxt - cur))
    return out


def plot_region_2d(ax, problem: Problem, color="#cfe6ff", edge="#3b7db3"):
    poly = feasible_polygon_2d(problem)
    if len(poly):
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.7, zorder=1)
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                color=edge, lw=1.4, zorder=2)


def plot_box_2d(ax, l, u, color="#e67e22", label="cube"):
    l, u = np.asarray(l), np.asarray(u)
    xs = [l[0], u[0], u[0], l[0], l[0]]
    ys = [l[1], l[1], u[1], u[1], l[1]]
    ax.plot(xs, ys, color=color, lw=2, zorder=3, label=label)


def plot_parallelepiped_2d(ax, center, basis, ext, color="#8e44ad", label="parallelepiped"):
    v = parallelepiped_vertices(center, basis, ext)
    order = [0, 1, 3, 2, 0]  # обход контура для 2^2 вершин
    ax.plot(v[order, 0], v[order, 1], color=color, lw=2, zorder=3, label=label)
    ax.scatter([center[0]], [center[1]], color=color, s=20, zorder=4)


# ---------------------------------------------------------------- 3d helpers
def plot_polyhedron_3d(ax, verts, faces, color="#66d9ff", alpha=0.22, edge="#1f9fd1"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection([verts[f] for f in faces], facecolors=color,
                            edgecolors=edge, linewidths=1.0, alpha=alpha)
    ax.add_collection3d(poly)


def plot_box_3d(ax, l, u, color="#c9d94d", alpha=0.55, edge="#7c8f00"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = box_vertices(l, u)
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
             [2, 3, 7, 6], [1, 3, 7, 5], [0, 2, 6, 4]]
    poly = Poly3DCollection([v[f] for f in faces], facecolors=color,
                            edgecolors=edge, linewidths=1.1, alpha=alpha)
    ax.add_collection3d(poly)


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    r = 0.5 * spans.max()
    ax.set_xlim3d([centers[0] - r, centers[0] + r])
    ax.set_ylim3d([centers[1] - r, centers[1] + r])
    ax.set_zlim3d([centers[2] - r, centers[2] + r])
