"""Synthetic regression check: paired inputs, plotting, and embedded fonts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pymupdf

with tempfile.TemporaryDirectory(prefix="eldr-plot-check-") as directory:
    root = Path(directory)
    (root / "inputs").mkdir()
    (root / "results").mkdir()
    for workload, domains in (("task", ["code", "math", "medical", "legal"]),
                              ("language", ["English", "Chinese", "Russian", "French"])):
        pairs = [[f"synthetic {domain} {i}", domain] for domain in domains for i in range(80)]
        raw = json.dumps(pairs).encode()
        (root / "inputs" / f"{workload}.json").write_bytes(raw)
        for model, layers in (("qwen", 48), ("gptoss", 36), ("gemma", 30)):
            counts = np.random.default_rng(0).integers(0, 8, (len(pairs), layers, 128), dtype=np.int32)
            np.savez_compressed(
                root / "results" / f"{model}-{workload}-full.npz",
                prefill_counts=counts, decode_counts=counts,
                request_ids=np.asarray([hashlib.sha256(p.encode()).hexdigest() for p, _ in pairs]),
                metadata=np.asarray(json.dumps(dict(model=model, training_sha256=hashlib.sha256(raw).hexdigest()))),
            )
    command = [sys.executable, str(Path(__file__).with_name("plot_motivation.py")),
               str(root), "--fresh", "--output", str(root / "preview"),
               "--save-data", str(root / "motivation.json")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = root / "preview/03_motivation"
    summary = json.loads((output / "summary.json").read_text())
    assert all(abs(row["r"] - 1) < 1e-12 for row in summary["pooled_correlation"].values())
    compact_command = [sys.executable, str(Path(__file__).with_name("plot_motivation.py")),
                       str(root / "motivation.json"), "--output", str(root / "compact")]
    result = subprocess.run(compact_command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert summary == json.loads((root / "compact/03_motivation/summary.json").read_text())
    for name in ("domain_2x3_layer", "pd_scatter"):
        with pymupdf.open(output / f"{name}.pdf") as doc:
            assert len(doc) == 1
            for xref, _, kind, *_ in doc[0].get_fonts():
                assert kind != "Type3" and doc.extract_font(xref)[3]
            with pymupdf.open(root / "compact/03_motivation" / f"{name}.pdf") as compact:
                assert doc[0].get_pixmap().samples == compact[0].get_pixmap().samples
    path = root / "results/qwen-task-full.npz"
    with np.load(path, allow_pickle=False) as data:
        changed = dict(data)
    changed["request_ids"] = changed["request_ids"][::-1]
    np.savez_compressed(path, **changed)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "alignment mismatch" in result.stderr
print("PASS: r=1; compact redraw identical; fonts embedded; mismatched requests rejected.")
