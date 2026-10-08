"""Deterministic development, frozen, and stress instance generation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import random
from typing import Sequence

from .model import Box, Instance
from .oracle import exact_oracle


FAMILIES = ("stable", "unstable", "diagonal", "coupled")
# Preserves the inherited frozen suite's 33 safe / 39 violating balance.
FROZEN_SAFE = (
    True, False, False, False, False, False, False, True,
    False, False, False, True, False, False, False, True,
    True, True, True, True, True, True, True, True,
    True, False, False, False, False, False, False, False,
    True, False, False, False, True, True, True, False,
    True, False, True, True, True, False, True, True,
    True, False, False, False, True, True, True, True,
    False, False, False, False, False, False, False, False,
    False, True, True, True, False, False, True, True,
)


def _matrix_for(family: str, d: int, q: int, width: int, t: int, rng: random.Random) -> tuple[tuple[int, ...], ...]:
    lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    if family == "stable":
        vals = [max(lo, min(hi, q // 2 + ((t + rng.randrange(3)) % 2)))]
        return ((vals[0],),)
    if family == "unstable":
        a = min(hi, q + 1 + ((t + rng.randrange(2)) % 2))
        return ((a,),)
    if family == "diagonal":
        a = max(lo, min(hi, q - 1 - (t % 2)))
        b = max(lo, min(hi, q // 2 + ((t + 1) % 2)))
        return ((a, 0), (0, b))
    if family == "coupled":
        a = max(lo, min(hi, q - 1))
        b = 1 if (t % 2 == 0) else -1
        c = -1 if (t % 2 == 0) else 1
        e = max(lo, min(hi, q // 2 + 1))
        return ((a, b), (c, e))
    raise ValueError(family)


def _base_instance(index: int, family: str, rng: random.Random, split: str) -> Instance:
    combos = ((4, 2), (5, 4), (6, 8), (4, 4), (5, 8), (6, 2))
    width, scale = combos[(index // 4) % len(combos)]
    d = 1 if family in {"stable", "unstable"} else 2
    horizon = 4 if d == 1 else 3
    mode = "floor" if ((index // 4) % 2 == 0) else "rne"
    association = ("left", "balanced", "right")[(index // 4) % 3]
    matrices = tuple(_matrix_for(family, d, scale, width, t, rng) for t in range(horizon))
    word_lo, word_hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    h0 = tuple(rng.randint(-min(2, -word_lo), min(3, word_hi)) for _ in range(d))
    boxes: list[tuple[Box, ...]] = []
    for t in range(horizon):
        row: list[Box] = []
        for i in range(d):
            # All boxes contain exactly three integer codes. A few cases touch a
            # word boundary to exercise clipping while preserving the oracle count.
            selector = (index + 3 * t + i) % 17
            if selector == 0:
                row.append(Box(word_lo, word_lo + 2))
            elif selector == 8:
                row.append(Box(word_hi - 2, word_hi))
            else:
                center = rng.choice((-1, 0, 1))
                center = max(word_lo + 1, min(word_hi - 1, center))
                row.append(Box(center - 1, center + 1))
        boxes.append(tuple(row))
    name = ("case" if split == "frozen" else "dev") + f"-{index:03d}"
    return Instance(
        name=name,
        family=family,
        width=width,
        scale=scale,
        rounding=mode,
        dimension=d,
        horizon=horizon,
        chunk_length=2,
        association=association,
        h0=h0,
        matrices=matrices,
        boxes=tuple(boxes),
        tolerances=tuple(tuple(0 for _ in range(d)) for _ in range(horizon)),
        metadata={"split": split, "generator_seed_family": family},
    )


def _label(instance: Instance, want_safe: bool) -> Instance | None:
    probe = exact_oracle(instance)
    maxima = [[max(abs(lo), abs(hi)) for lo, hi in row] for row in probe.ranges]
    if not want_safe and all(value == 0 for row in maxima for value in row):
        return None
    tolerances = [row[:] for row in maxima]
    if not want_safe:
        chosen = None
        for t, row in enumerate(maxima):
            for i, value in enumerate(row):
                if value > 0:
                    chosen = (t, i, value)
                    break
            if chosen:
                break
        assert chosen is not None
        t, i, value = chosen
        tolerances[t][i] = value - 1
    out = replace(instance, tolerances=tuple(tuple(x for x in row) for row in tolerances))
    final = exact_oracle(out)
    if final.safe != want_safe:
        raise AssertionError("label construction failed")
    md = dict(out.metadata)
    md.update({
        "oracle_safe": final.safe,
        "oracle_earliest_output": final.earliest_output,
        "oracle_witness": list(final.witness) if final.witness is not None else None,
        "oracle_ranges": [[[lo, hi] for lo, hi in row] for row in final.ranges],
        "prefix_evaluations": final.prefix_evaluations,
        "tolerance_rule": "exact-max" if want_safe else "exact-max-minus-one-at-first-positive-coordinate",
    })
    return replace(out, metadata=md)


def generate_collection(count: int, seed: int, split: str) -> list[Instance]:
    if count % 4:
        raise ValueError("collection size must be a multiple of four")
    rng = random.Random(seed)
    result: list[Instance] = []
    for index in range(count):
        family = FAMILIES[index % 4]
        want_safe = FROZEN_SAFE[index] if split == "frozen" else ((index * 5 + 1) % 7 < 3)
        for _attempt in range(100):
            base = _base_instance(index, family, rng, split)
            labeled = _label(base, want_safe)
            if labeled is not None:
                result.append(labeled)
                break
        else:
            raise RuntimeError(f"could not generate discriminating {family} case {index}")
    return result


def generate_stress() -> list[Instance]:
    result: list[Instance] = []
    idx = 0
    for horizon in (8, 16, 32, 64):
        for family, coefficient in (("stable", 2), ("stable", 2), ("unstable", 5), ("unstable", 5)):
            mode = "floor" if idx % 2 == 0 else "rne"
            h0 = (0,) if idx % 4 in (0, 2) else (3,)
            tolerance = (2 if idx % 4 == 0 else 1) if coefficient == 2 else 15
            inst = Instance(
                name=f"stress-{idx:03d}",
                family=family,
                width=4,
                scale=4,
                rounding=mode,
                dimension=1,
                horizon=horizon,
                chunk_length=4,
                association="balanced",
                h0=h0,
                matrices=tuple(((coefficient,),) for _ in range(horizon)),
                boxes=tuple((Box(-1, 1),) for _ in range(horizon)),
                tolerances=tuple((tolerance,) for _ in range(horizon)),
                metadata={
                    "split": "stress",
                    "contract_kind": ("two-code" if tolerance == 2 else "tight-one-code") if coefficient == 2 else "vacuous-word-span",
                    "coefficient_code": coefficient,
                },
            )
            result.append(inst)
            idx += 1
    return result


def write_collections(root: str | Path) -> dict[str, object]:
    root = Path(root)
    specs = {
        "development": generate_collection(24, 73129, "development"),
        "frozen": generate_collection(72, 90173, "frozen"),
        "stress": generate_stress(),
    }
    summary: dict[str, object] = {}
    for split, cases in specs.items():
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.json"):
            old.unlink()
        for case in cases:
            case.save(out / f"{case.name}.json")
        summary[split] = {
            "count": len(cases),
            "names": [x.name for x in cases],
            "prefix_evaluations": sum(int(x.metadata.get("prefix_evaluations", 0)) for x in cases),
            "safe": sum(bool(x.metadata.get("oracle_safe")) for x in cases if "oracle_safe" in x.metadata),
        }
    with (root / "manifest.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary
