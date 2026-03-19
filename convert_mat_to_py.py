from __future__ import annotations

import math
import pathlib
import re
from typing import Any


_NUM_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?|inf|Inf|nan|NaN)$"
)


def convert_matpower_to_pypower(
    matpower_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    function_name: str | None = None,
) -> str:
    """
    Convert a MATPOWER .m case file into a PYPOWER-style .py case file.

    Parameters
    ----------
    matpower_path : str | Path
        Path to MATPOWER case file, e.g. "case1197.m".
    output_path : str | Path | None
        Path to output PYPOWER file, e.g. "case1197.py".
        If None, the function only returns the generated Python source.
    function_name : str | None
        Name of the Python case function. If None, inferred from the MATPOWER
        function name, e.g. `case1197`.

    Returns
    -------
    str
        Generated PYPOWER Python source code.

    Notes
    -----
    This handles the common MATPOWER v2 pattern:
        mpc.version = '2';
        mpc.baseMVA = ...;
        mpc.bus = [ ... ];
        mpc.gen = [ ... ];
        mpc.branch = [ ... ];
        mpc.gencost = [ ... ];

    It also preserves extra literal fields such as areas, bus_name, gentype, etc.

    It does NOT execute arbitrary MATLAB code. So if the .m file builds arrays
    through helper functions or non-literal expressions, use Octave/MATLAB
    `loadcase()` first and then serialize from the loaded structure.
    """
    matpower_path = pathlib.Path(matpower_path)
    text = matpower_path.read_text(encoding="utf-8")
    clean = _strip_matlab_comments(text)

    if function_name is None:
        m = re.search(r"\bfunction\s+\w+\s*=\s*(\w+)\b", clean)
        function_name = m.group(1) if m else matpower_path.stem

    assignments = _parse_assignments(clean)
    case = {k: _parse_expr(v) for k, v in assignments.items()}
    py_code = _generate_pypower_case(case, function_name)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.write_text(py_code, encoding="utf-8")

    return py_code


def _strip_matlab_comments(text: str) -> str:
    """Remove MATLAB '%' comments, respecting single-quoted strings."""
    out_lines = []

    for line in text.splitlines():
        buf = []
        in_str = False
        i = 0

        while i < len(line):
            ch = line[i]

            if ch == "'":
                # MATLAB escapes apostrophe inside strings as ''
                if in_str and i + 1 < len(line) and line[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue

                in_str = not in_str
                buf.append(ch)
                i += 1
                continue

            if ch == "%" and not in_str:
                break

            buf.append(ch)
            i += 1

        out_lines.append("".join(buf))

    return "\n".join(out_lines)


def _parse_assignments(text: str) -> dict[str, str]:
    """
    Extract assignments of the form:
        mpc.field = ...;
    using top-level semicolon matching.
    """
    assignments: dict[str, str] = {}
    i = 0
    n = len(text)

    while i < n:
        m = re.search(r"\bmpc\.(\w+)\s*=", text[i:])
        if not m:
            break

        field = m.group(1)
        start = i + m.end()

        depth_round = 0
        depth_square = 0
        depth_curly = 0
        in_str = False
        j = start

        while j < n:
            ch = text[j]

            if ch == "'":
                if in_str and j + 1 < n and text[j + 1] == "'":
                    j += 2
                    continue
                in_str = not in_str
                j += 1
                continue

            if not in_str:
                if ch == "(":
                    depth_round += 1
                elif ch == ")":
                    depth_round -= 1
                elif ch == "[":
                    depth_square += 1
                elif ch == "]":
                    depth_square -= 1
                elif ch == "{":
                    depth_curly += 1
                elif ch == "}":
                    depth_curly -= 1
                elif ch == ";" and depth_round == depth_square == depth_curly == 0:
                    assignments[field] = text[start:j].strip()
                    j += 1
                    break

            j += 1

        i = j

    return assignments


def _parse_expr(expr: str) -> Any:
    expr = expr.strip()

    if expr.startswith("[") and expr.endswith("]"):
        return _parse_matrix(expr)

    if expr.startswith("{") and expr.endswith("}"):
        return _parse_cell_array(expr)

    if expr.startswith("'") and expr.endswith("'"):
        return expr[1:-1].replace("''", "'")

    if _is_number(expr):
        return _parse_number(expr)

    # Fallback: keep raw MATLAB expression as a string
    return expr


def _parse_matrix(expr: str) -> list[list[Any]]:
    """Parse MATLAB numeric/string matrix literal [ ... ]."""
    body = expr[1:-1].strip()
    if not body:
        return []

    rows: list[list[str]] = []
    row_tokens: list[str] = []
    token: list[str] = []

    depth_round = 0
    depth_square = 0
    depth_curly = 0
    in_str = False
    i = 0

    def flush_token() -> None:
        nonlocal token, row_tokens
        t = "".join(token).strip()
        if t:
            row_tokens.append(t)
        token = []

    def flush_row() -> None:
        nonlocal row_tokens, rows
        flush_token()
        if row_tokens:
            rows.append(row_tokens)
        row_tokens = []

    while i < len(body):
        ch = body[i]

        if ch == "'":
            token.append(ch)
            if in_str and i + 1 < len(body) and body[i + 1] == "'":
                token.append("'")
                i += 2
                continue
            in_str = not in_str
            i += 1
            continue

        if not in_str:
            if ch == "(":
                depth_round += 1
                token.append(ch)
            elif ch == ")":
                depth_round -= 1
                token.append(ch)
            elif ch == "[":
                depth_square += 1
                token.append(ch)
            elif ch == "]":
                depth_square -= 1
                token.append(ch)
            elif ch == "{":
                depth_curly += 1
                token.append(ch)
            elif ch == "}":
                depth_curly -= 1
                token.append(ch)
            elif depth_round == depth_square == depth_curly == 0 and ch in ", \t":
                flush_token()
            elif depth_round == depth_square == depth_curly == 0 and ch == ";":
                flush_row()
            elif depth_round == depth_square == depth_curly == 0 and ch in "\r\n":
                flush_row()
            else:
                token.append(ch)
        else:
            token.append(ch)

        i += 1

    flush_row()

    parsed_rows: list[list[Any]] = []
    for row in rows:
        parsed_row: list[Any] = []
        for tok in row:
            tok = tok.strip()
            if tok.startswith("'") and tok.endswith("'"):
                parsed_row.append(tok[1:-1].replace("''", "'"))
            elif _is_number(tok):
                parsed_row.append(_parse_number(tok))
            else:
                parsed_row.append(tok)
        parsed_rows.append(parsed_row)

    return parsed_rows


def _parse_cell_array(expr: str) -> list[Any]:
    """Parse simple MATLAB cell array literal { ... }."""
    body = expr[1:-1].strip()
    if not body:
        return []

    parts: list[str] = []
    start = 0
    depth_round = 0
    depth_square = 0
    depth_curly = 0
    in_str = False
    i = 0

    while i < len(body):
        ch = body[i]

        if ch == "'":
            if in_str and i + 1 < len(body) and body[i + 1] == "'":
                i += 2
                continue
            in_str = not in_str
        elif not in_str:
            if ch == "(":
                depth_round += 1
            elif ch == ")":
                depth_round -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
            elif ch == "{":
                depth_curly += 1
            elif ch == "}":
                depth_curly -= 1
            elif ch in ";," and depth_round == depth_square == depth_curly == 0:
                parts.append(body[start:i].strip())
                start = i + 1

        i += 1

    parts.append(body[start:].strip())

    out: list[Any] = []
    for p in parts:
        if p.startswith("'") and p.endswith("'"):
            out.append(p[1:-1].replace("''", "'"))
        elif _is_number(p):
            out.append(_parse_number(p))
        else:
            out.append(p)

    return out


def _is_number(tok: str) -> bool:
    return bool(_NUM_RE.match(tok.strip()))


def _parse_number(tok: str) -> int | float:
    tok = tok.strip().replace("D", "e").replace("d", "e")

    if tok.lower() == "inf":
        return float("inf")
    if tok.lower() == "nan":
        return float("nan")

    val = float(tok)
    if math.isfinite(val) and val.is_integer():
        return int(val)
    return val


def _py_repr(x: Any) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return 'float("nan")'
        if math.isinf(x):
            return 'float("inf")' if x > 0 else '-float("inf")'
        return repr(float(x))
    return repr(x)


def _format_matrix(rows: list[list[Any]], indent: str = "        ") -> str:
    lines = ["array(["]
    for row in rows:
        lines.append(f"{indent}{_py_repr(row)},")
    lines.append("])")
    return "\n".join(lines)


def _generate_pypower_case(case: dict[str, Any], function_name: str) -> str:
    """
    Emit a PYPOWER case file as Python source.
    """
    preferred_order = ["version", "baseMVA", "bus", "gen", "branch", "gencost", "areas"]

    lines: list[str] = [
        "from numpy import array",
        "",
        f"def {function_name}():",
        '    """Auto-converted from a MATPOWER case file."""',
        "",
        "    ppc = {}",
    ]

    version = str(case.get("version", "2"))
    lines.append(f"    ppc['version'] = {_py_repr(version)}")
    lines.append("")

    for key in preferred_order:
        if key == "version" or key not in case:
            continue
        _append_field(lines, key, case[key])

    for key, value in case.items():
        if key in preferred_order:
            continue
        _append_field(lines, key, value)

    lines.append("    return ppc")
    lines.append("")
    return "\n".join(lines)


def _append_field(lines: list[str], key: str, value: Any) -> None:
    if isinstance(value, list) and (not value or isinstance(value[0], list)):
        lines.append(f"    ppc[{key!r}] = {_format_matrix(value)}")
    else:
        lines.append(f"    ppc[{key!r}] = {_py_repr(value)}")
    lines.append("")
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, required=True, help="Case name")
    args = parser.parse_args()
    
    case_name = args.case_name
    
    convert_matpower_to_pypower(matpower_path=f"{case_name}.m",
                                output_path=f"{case_name}.py",
                                )