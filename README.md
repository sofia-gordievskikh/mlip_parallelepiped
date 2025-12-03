# MLIP parallelepiped

Исследовательский код по методам оптимизации: MLIP через параллелепипед вместо
куба.

Основной файл:

- `notebooks/mlip_parallelepiped.ipynb`

Запуск:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/mlip_parallelepiped.ipynb
```

В ноутбуке есть LP-постановка через `scipy.optimize.linprog` и визуализация
области в 3D.
