"""Тесты solver: feasibility, размеры матриц, cube vs parallelepiped."""
import numpy as np
import pytest

from mlip_parallelepiped import (
    Problem,
    inner_box_search,
    parallelepiped_search,
    solve_box_lp,
    solve_parallelepiped,
)
from mlip_parallelepiped.geometry import (
    box_vertices,
    identity_basis,
    parallelepiped_vertices,
    rotation_2d,
)


@pytest.fixture()
def toy_3d():
    return Problem(
        c=[3.0, 2.0, 1.0],
        A=[[2, 1, 1], [1, 2, 0], [0, 1, 2]],
        b=[4.5, 4.0, 3.5],
        l_bounds=[0, 0, 0],
        u_bounds=[3, 3, 3],
        int_idx=[0, 1],
    )


@pytest.fixture()
def slab_2d():
    # тонкая наклонная полоса
    return Problem(
        c=[1.0, 1.0],
        A=[[1, -1], [-1, 1], [1, 1]],
        b=[1.0, 0.0, 10.0],
        l_bounds=[0, 0],
        u_bounds=[6, 6],
        int_idx=[0],
    )


def test_box_feasible_simple(toy_3d):
    res = inner_box_search(toy_3d, tau=1.0)
    assert res["status"] == "ok"
    assert res["candidate_feasible"]
    # кандидат целочислен по int_idx
    for i in toy_3d.int_idx:
        assert res["candidate"][i] == round(res["candidate"][i])


def test_box_vertices_inside_region(toy_3d):
    l, u, _ = solve_box_lp(toy_3d, tau=1.0)
    for v in box_vertices(l, u):
        assert np.all(toy_3d.A @ v <= toy_3d.b + 1e-6)


def test_parallelepiped_vertices_inside_region(toy_3d):
    basis = identity_basis(3)
    center, ext, _ = solve_parallelepiped(toy_3d, basis, gamma=0.2, tau=1.0)
    for v in parallelepiped_vertices(center, basis, ext):
        assert np.all(toy_3d.A @ v <= toy_3d.b + 1e-6)


def test_identity_basis_matches_box(toy_3d):
    # параллелепипед с базисом I и box-LP должны давать одинаковый center objective
    box = inner_box_search(toy_3d, tau=1.0)
    center, ext, obj = solve_parallelepiped(toy_3d, identity_basis(3), gamma=0.0, tau=1.0)
    assert obj == pytest.approx(box["center_objective"], abs=1e-6)


def test_parallelepiped_beats_cube_on_tilted_region(slab_2d):
    # в тонкой наклонной полосе куб не влезает по ширине, а параллелепипед - да
    tau = 1.5
    box = inner_box_search(slab_2d, tau=tau)
    par = parallelepiped_search(slab_2d, rotation_2d(np.radians(45)), tau=tau)
    assert box["status"] != "ok"          # куб infeasible
    assert par["status"] == "ok"
    assert par["candidate_feasible"]


def test_parallelepiped_objective_ge_cube_when_both_feasible():
    p = Problem(
        c=[1.0, 1.0],
        A=[[1, -1], [-1, 1], [1, 1]],
        b=[2.0, 0.0, 10.0],
        l_bounds=[0, 0],
        u_bounds=[6, 6],
        int_idx=[0],
    )
    tau = 1.0
    box = inner_box_search(p, tau=tau)
    par = parallelepiped_search(p, rotation_2d(np.radians(45)), tau=tau)
    assert box["status"] == "ok" and par["status"] == "ok"
    assert par["center_objective"] >= box["center_objective"] - 1e-9


def test_matrix_sizes(slab_2d):
    # проверка размеров: box-LP имеет 2n переменных
    n = slab_2d.n
    l, u, _ = solve_box_lp(slab_2d, tau=0.5)
    assert len(l) == n and len(u) == n
    center, ext, _ = solve_parallelepiped(slab_2d, identity_basis(n), tau=0.5)
    assert len(center) == n and len(ext) == n


def test_infeasible_returns_none():
    # заведомо несовместная область
    p = Problem(c=[1.0], A=[[1.0], [-1.0]], b=[1.0, -5.0],
                l_bounds=[0.0], u_bounds=[10.0], int_idx=[])
    assert solve_box_lp(p, tau=0.1) is None
