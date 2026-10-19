"""
Геометрия модели определяется списком смещений для каждой клетки 
"""

from __future__ import annotations

import numpy as np

def ring_offsets(radius: int) -> np.ndarray:
    """
    1d-ring: соседи в окне радиуса radius без клетки 
    возвращает массив формы (n_offsets, 1) 
    """

    if radius < 1: 
        raise ValueError("Радиус должен быть больше единицы")
    return np.array([[d] for d in range(-radius, radius + 1) if d != 0])


def torus_offsets(radius: int, metric: str = "moore") -> np.ndarray:
    """
    2d-torus: 

    metric = "moore" - квадрат 
    metric = "von_neumann" - ромб
    """

    if radius < 1:
        raise ValueError("radius должен быть >= 1")
    offs = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if not (dx == 0 and dy == 0)
    ]
    if metric == "von_neumann":
        offs = [(dx, dy) for dx, dy in offs if abs(dx) + abs(dy) <= radius]
    elif metric != "moore":
        raise ValueError(f"неизвестная метрика: {metric}")
    return np.array(offs, dtype=int)