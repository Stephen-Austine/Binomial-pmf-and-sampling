"""Minimal reader for GNU Octave's *text* save format.

``scipy.io.loadmat`` only understands MATLAB's binary MAT-files. Octave's default
``save`` format is plain text (``# name: ...`` / ``# type: ...`` blocks), which is
what several ``.mat`` files in this repository use. This reader supports the
three types that appear in them: ``matrix``, ``scalar`` and ``double_range``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Value = float | np.ndarray


def load_octave_text(path: str | Path) -> dict[str, Value]:
    """Load variables from an Octave text-format file.

    Returns:
        Mapping of variable name to ``float`` (scalars) or ``numpy.ndarray``.
        Matrices keep their stored shape; ranges are expanded to 1-D arrays.

    Raises:
        ValueError: If a variable has a type this reader does not support.
    """
    lines = Path(path).read_text().replace("\r\n", "\n").split("\n")
    variables: dict[str, Value] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("# name:"):
            i += 1
            continue

        name = lines[i].split(":", 1)[1].strip()
        vtype = lines[i + 1].split(":", 1)[1].strip()
        i += 2

        if vtype == "matrix":
            rows = int(lines[i].split(":")[1])
            cols = int(lines[i + 1].split(":")[1])
            i += 2
            data = [[float(t) for t in lines[i + r].split()] for r in range(rows)]
            variables[name] = np.array(data).reshape(rows, cols)
            i += rows
        elif vtype == "scalar":
            variables[name] = float(lines[i].strip())
            i += 1
        elif vtype == "double_range":
            # Header comment "# base, limit, increment", then the three numbers.
            base, limit, inc = (float(t) for t in lines[i + 1].split())
            variables[name] = np.arange(base, limit + inc / 2, inc)
            i += 2
        else:
            raise ValueError(
                f"Unsupported Octave type {vtype!r} for variable {name!r}."
            )
    return variables
