"""Render docs/assets/bench.png from a measured `qtensor bench --json FILE` result.

Usage: python scripts/plot_bench.py [docs/assets/bench.json] [docs/assets/bench.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def plot(data: dict, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    b = data["brick"]
    chi = [r["chi"] for r in b]
    axes[0].semilogx(chi, [r["true_fidelity"] for r in b], "o-", base=2, label="true fidelity")
    axes[0].semilogx(chi, [r["estimated_fidelity"] for r in b], "x--", base=2, label="estimate Π(1−ε)")
    axes[0].set_xlabel("bond dimension χ")
    axes[0].set_ylabel("fidelity with exact state")
    axes[0].set_title("Haar brick-wall n=16, depth 12")
    ax2 = axes[0].twinx()
    ax2.loglog(chi, [r["mps_bytes"] for r in b], "g:", base=2)
    ax2.axhline(b[0]["dense_bytes"], color="g", lw=0.8)
    ax2.set_ylabel("MPS bytes (green; line = dense)")
    axes[0].legend(loc="center right")
    t = data["tfim"]
    labels = [f"TEBD χ={r['chi']}" for r in t["tebd"]] + [f"RBM α={r['alpha']}" for r in t.get("rbm", [])]
    errs = [r["rel_err"] for r in t["tebd"]] + [r["rel_err"] for r in t.get("rbm", [])]
    axes[1].bar(labels, errs, color="tab:purple")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative energy error")
    axes[1].set_title(f"Critical TFIM ground state, n={t['n']}")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("qtensor bench — seed 0, CPU (measured)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/bench.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/assets/bench.png")
    print(plot(json.loads(src.read_text()), dst))
