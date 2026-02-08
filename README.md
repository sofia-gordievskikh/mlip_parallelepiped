# mlip_parallelepiped

Исследовательский код по методам оптимизации: **inner box search** для MILP.
В допустимую область LP-релаксации вписываем коробку или параллелепипед, двигаем
её центр вдоль `c` и берём целочисленный угол как быстрый кандидат в решение.

Основная идея репозитория - сравнить два варианта вписанной области:
**кубовую (axis-aligned)** и **параллелепипед (повёрнутый базис)**.

![cube vs parallelepiped](docs/figures/cube_vs_parallelepiped.png)

## Математическая постановка

Решаем задачу

```
max  c·x
s.t. A x <= b,   l_bounds <= x <= u_bounds,   x_i целое (i ∈ int_idx)
```

### Кубовая область

Коробка `x ∈ [l, u]` со сторонами вдоль осей. Оптимизируем `l, u`. Условие
«коробка внутри области» - через опорную функцию:

```
sum_i ( a_ki >= 0 ? a_ki·u_i : a_ki·l_i ) <= b_k
```

Для целочисленных осей требуем ширину `u_i - l_i >= tau`. Цель `max c·(l+u)`.

### Параллелепипед

Область `P = { center + B·t : t ∈ [-ext, ext] }` с фиксированным базисом `B`.
Оптимизируем `center` и полудлины `ext`. Так как `B` фиксирован, условие
допустимости всех вершин линейно:

```
a_k·center + sum_i |a_k·B_i|·ext_i <= b_k
```

Переменные, которые оптимизируются, и полный вывод ограничений - в
[docs/notes.md](docs/notes.md).

### Что меняется при параллелепипеде

В наклонной допустимой области требуемую ширину `tau` дешевле набрать вдоль
области, а не по осям координат. Поэтому куб может стать **несовместным**, а
параллелепипед, повёрнутый под наклон, всё ещё вписывается и даёт кандидата с
бОльшим `c·center` (см. график objective(tau) выше).

![parallelepiped 3d](docs/figures/parallelepiped_3d.png)

## Установка

```bash
pip install -r requirements.txt
pip install -e .        # чтобы работали import mlip_parallelepiped и CLI
```

## CLI

```bash
python -m mlip_parallelepiped.solve --config configs/demo.yaml
python -m mlip_parallelepiped.solve --config configs/demo.yaml --plot out.png
```

Параметры задачи (c, A, b, границы, `int_idx`, `tau`, базис параллелепипеда)
лежат в [configs/demo.yaml](configs/demo.yaml).

## Пакет

- `mlip_parallelepiped/geometry.py` - `Problem`, вершины коробки/параллелепипеда;
- `mlip_parallelepiped/solver.py` - `solve_box_lp`, `solve_parallelepiped`,
  `inner_box_search`, `parallelepiped_search`;
- `mlip_parallelepiped/visualization.py` - 2D/3D отрисовка;
- `mlip_parallelepiped/solve.py` - CLI.

## Примеры

```bash
python examples/toy_2d.py                 # 2D наклонная полоса
python examples/cube_vs_parallelepiped.py # sweep по tau + картинка
python examples/parallelepiped_3d.py      # 3D-визуализация
```

Ноутбук `notebooks/experiments.ipynb` вызывает уже вынесенный пакетный код (без
дублирования логики). Исходный черновой ноутбук - `notebooks/mlip_parallelepiped.ipynb`.

## Тесты

```bash
pytest -q
```

Проверяют feasibility, совпадение `B = I` с box-LP, размеры матриц и что
параллелепипед выигрывает у куба на наклонной области.

## Черновик отчёта

Короткий LaTeX-набросок - в [report/report.tex](report/report.tex).
