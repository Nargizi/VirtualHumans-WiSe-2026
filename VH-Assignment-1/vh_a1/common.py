"""Shared helpers: formative checks and a summary table.

The checks are *formative*: they tell students whether their code reproduces reference values.
They are not the grade (instructors read the code and the written answers).
Expected values are stored as plain numbers in the check cells, never as reference code.
"""
from __future__ import annotations

import html
from typing import Callable

import numpy as np

try:  # IPython is present in Colab/Jupyter; the package must also import without it (tests, CLI)
    from IPython.display import HTML, display
except Exception:  # pragma: no cover
    HTML = None

    def display(*a, **k):  # type: ignore
        print(*a)

# status -> (label, colour)
_STATUS = {"pass": ("passed", "#2e7d32"), "fail": ("FAILED", "#c62828"), "todo": ("not implemented", "#757575")}
_RESULTS: dict[str, dict[str, str]] = {}  # part -> {check name -> status}


def _emit(rows_html: str):
    if HTML is not None:
        display(HTML(rows_html))
    else:  # plain-text fallback
        print(rows_html)


class Checker:
    """Run small tests after a coding cell and print one line per check.

    A check is "not implemented" if the student function returns ``None`` or raises ``NotImplementedError``;
    it FAILED if it raises anything else or returns a wrong value.
    """

    def __init__(self, part: str):
        self.part = part
        _RESULTS.setdefault(part, {})

    # ---- core -----------------------------------------------------------------------
    def _record(self, name: str, status: str, detail: str = "", hint: str = ""):
        _RESULTS[self.part][name] = status
        label, col = _STATUS[status]
        extra = ""
        if status == "fail" and detail:
            extra += f"<div style='margin-left:1.2em;font-family:monospace;font-size:.85em;opacity:.85'>{html.escape(detail)}</div>"
        if status == "fail" and hint:
            extra += f"<div style='margin-left:1.2em;font-size:.88em'>Hint: {html.escape(hint)}</div>"
        _emit(f"<div style='font-family:system-ui,sans-serif;margin:.1em 0'>"
              f"<span class='vh-st' data-s='{status}' style='color:{col};font-weight:600;display:inline-block;min-width:9.5em'>{label}</span>"
              f"{html.escape(name)}{extra}</div>")

    def test(self, name: str, fn: Callable[[], object], hint: str = ""):
        """`fn` returns True/False, or (bool, detail), or None (= not implemented)."""
        try:
            out = fn()
        except NotImplementedError:
            return self._record(name, "todo")
        except (TypeError, AttributeError) as e:
            # a stub that still returns None makes the *check* fail on "None @ ..." etc. -> not implemented yet
            if "NoneType" in str(e):
                return self._record(name, "todo")
            return self._record(name, "fail", f"{type(e).__name__}: {e}", hint)
        except Exception as e:  # noqa: BLE001  (students' code can raise anything)
            return self._record(name, "fail", f"{type(e).__name__}: {e}", hint)
        if out is None:
            return self._record(name, "todo")
        ok, detail = (out if isinstance(out, tuple) else (bool(out), ""))
        self._record(name, "pass" if ok else "fail", detail, hint)

    def close(self, name: str, fn: Callable[[], object], expected, atol: float = 1e-6, hint: str = ""):
        """`fn()` must return an array close to `expected` (same shape)."""

        def run():
            got = fn()
            if got is None:
                return None
            got = np.asarray(got, dtype=float)
            exp = np.asarray(expected, dtype=float)
            if got.shape != exp.shape:
                return False, f"shape {got.shape}, expected {exp.shape}"
            if np.allclose(got, exp, atol=atol):
                return True, ""
            return False, f"got {np.array2string(got, precision=4, suppress_small=True)}  expected {np.array2string(exp, precision=4, suppress_small=True)}"

        self.test(name, run, hint)

    # ---- summary ----------------------------------------------------------------------
    def summary(self):
        progress(self.part)


def progress(part: str | None = None):
    """Table of check results so far (formative)."""
    parts = [part] if part else list(_RESULTS)
    rows = "<tr><th align='left' style='padding:2px 12px'>checks</th><th style='padding:2px 12px'>passed</th><th style='padding:2px 12px'>failed</th><th style='padding:2px 12px'>not implemented</th></tr>"
    for p in parts:
        res = _RESULTS.get(p, {})
        n = lambda s: sum(v == s for v in res.values())
        rows += (f"<tr><td style='padding:2px 12px'>{html.escape(p)}</td><td align='center'>{n('pass')}</td>"
                 f"<td align='center'>{n('fail')}</td><td align='center'>{n('todo')}</td></tr>")
    _emit(f"<table style='font-family:system-ui,sans-serif;border-collapse:collapse'>{rows}</table>")
    return {p: dict(_RESULTS.get(p, {})) for p in parts}


def results() -> dict[str, dict[str, str]]:
    """Machine-readable copy of the check results (used by the test-suite)."""
    return {p: dict(r) for p, r in _RESULTS.items()}


def reset_results():
    _RESULTS.clear()


def ready(*values) -> bool:
    """True if none of the values is None (students' stub functions return None until implemented)."""
    return all(v is not None for v in values)
