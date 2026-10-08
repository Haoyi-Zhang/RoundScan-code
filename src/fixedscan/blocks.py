"""Projection helpers for block-diagonal exact-factorization tests."""
from __future__ import annotations
from .model import Box, Instance


def project_coordinate(instance: Instance, coordinate: int) -> Instance:
    if not 0 <= coordinate < instance.dimension:
        raise IndexError(coordinate)
    for matrix in instance.matrices:
        for i in range(instance.dimension):
            for j in range(instance.dimension):
                if i != j and matrix[i][j] != 0:
                    raise ValueError("instance is not diagonal")
    return Instance(
        name=f"{instance.name}-coord-{coordinate}", family=f"{instance.family}-projection",
        width=instance.width, scale=instance.scale, rounding=instance.rounding,
        dimension=1, horizon=instance.horizon, chunk_length=instance.chunk_length,
        association=instance.association, h0=(instance.h0[coordinate],),
        matrices=tuple(((m[coordinate][coordinate],),) for m in instance.matrices),
        boxes=tuple((step[coordinate],) for step in instance.boxes),
        tolerances=tuple((tol[coordinate],) for tol in instance.tolerances),
        metadata={"source": instance.name, "coordinate": coordinate},
    )


def project_prefix(flat_prefix: tuple[int, ...], dimension: int, coordinate: int) -> tuple[int, ...]:
    return tuple(flat_prefix[t * dimension + coordinate] for t in range(len(flat_prefix) // dimension))
