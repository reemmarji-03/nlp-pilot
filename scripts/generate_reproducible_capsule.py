import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.run_metadata import run_metadata


def pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line for line in out.splitlines() if line.strip()]


def main() -> int:
    out_dir = ROOT / "reproducible_capsule"
    out_dir.mkdir(exist_ok=True)
    payload = run_metadata({"pip_freeze": pip_freeze()})
    (out_dir / "environment.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# NLP Pilot Reproducible Capsule\n\n"
        "This capsule records Python, platform, package versions, redacted settings, "
        "and reproducibility notes for the current environment.\n\n"
        "Regenerate it with:\n\n"
        "```powershell\n"
        "python scripts/generate_reproducible_capsule.py\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
