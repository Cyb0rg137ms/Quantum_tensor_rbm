"""``qtensor`` command line: ``bench``, ``ghz``, ``tfim``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from qtensor.benchmarks import BenchConfig, qudit_ghz, run_all, tfim_ground_state


def _fmt(r: dict) -> str:
    out = [
        "## MPS vs dense state vector (untruncated)",
        "| n | d | gates | fidelity | max entropy err | max bond |",
        "|---|---|---|---|---|---|",
    ]
    out += [
        f"| {a['n']} | {a['d']} | {a['gates']} | {a['fidelity']:.15f} | {a['max_entropy_err']:.1e} | {a['max_bond']} |"
        for a in r["agreement"]
    ]
    out += [
        "",
        "## Qudit GHZ (d=10)",
        "| n | seconds | MPS bytes | dense bytes | S per bond / ln d |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {g['n']} | {g['seconds']:.4f} | {g['mps_bytes']} | 10^{g['dense_bytes_log10']:.1f} | "
        f"{g['entropy_min'] / g['log_d']:.6f} |"
        for g in r["ghz"]
    ]
    out += [
        "",
        "## Brick-wall Haar circuit: fidelity vs bond dimension",
        "| chi | true fidelity | estimate prod(1-eps) | mid-chain S | seconds | MPS bytes |",
        "|---|---|---|---|---|---|",
    ]
    out += [
        f"| {b['chi']} | {b['true_fidelity']:.4f} | {b['estimated_fidelity']:.4f} | {b['mid_entropy']:.3f} | "
        f"{b['seconds']:.3f} | {b['mps_bytes']} |"
        for b in r["brick"]
    ]
    t = r["tfim"]
    out += [
        "",
        f"## TFIM ground state (n={t['n']}, h=J=1), exact E0 = {t['exact_energy']:.10f}",
        "| method | energy | rel. error | seconds |",
        "|---|---|---|---|",
    ]
    out += [
        f"| imag-TEBD chi={x['chi']} | {x['energy']:.10f} | {x['rel_err']:.2e} | {x['seconds']:.2f} |"
        for x in t["tebd"]
    ]
    out += [
        f"| RBM alpha={x['alpha']} ({x['params']} params) | {x['energy']:.10f} | {x['rel_err']:.2e} | "
        f"{x['seconds']:.2f} |"
        for x in t["rbm"]
    ]
    out += ["", f"total {r['total_s']:.1f} s"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qtensor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--json")
    g = sub.add_parser("ghz")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--d", type=int, default=10)
    t = sub.add_parser("tfim")
    t.add_argument("--n", type=int, default=8)
    a = ap.parse_args(argv)
    np.set_printoptions(precision=6)
    if a.cmd == "bench":
        r = run_all(BenchConfig(seed=a.seed, quick=a.quick))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(r, fh, indent=2)
        print(_fmt(r))
    elif a.cmd == "ghz":
        print(json.dumps(qudit_ghz(a.n, a.d), indent=2))
    else:
        print(json.dumps(tfim_ground_state(a.n, True, 0), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
