#!/usr/bin/env python3
"""Run a minimal smoke suite for critical operational paths."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"

CRITICAL_SCRIPTS = [
    ROOT_DIR / "scripts" / "rotation_target_050_live.py",
    ROOT_DIR / "scripts" / "telegram_bot.py",
    ROOT_DIR / "scripts" / "operation_watchdog.py",
    ROOT_DIR / "scripts" / "pairwise_regime_live.py",
    ROOT_DIR / "scripts" / "pairwise_regime_mixture_shadow_live.py",
    ROOT_DIR / "scripts" / "search_core_champion.py",
]


def run_check(name: str, command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
    )
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "command": command,
    }


def main() -> None:
    checks = [
        run_check("py_compile_critical_scripts", [str(PYTHON_BIN), "-m", "py_compile", *[str(path) for path in CRITICAL_SCRIPTS]]),
        run_check(
            "fractal_self_check",
            [str(PYTHON_BIN), str(ROOT_DIR / "scripts" / "verify_fractal_genome_cells.py"), "--mode", "self-check"],
        ),
    ]
    payload = {
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
