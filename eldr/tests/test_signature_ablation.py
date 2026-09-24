"""CPU validation of paired calibration, greedy masks and probability transport."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybase64
from aiohttp import web

from eldr.experiments.signature_ablation import (
    fit_signature,
    greedy_mask,
    read_paired_capture,
)
from eldr.serving.clustering import build_signature, load_centroid_file
from eldr.serving.proxy import decode_signature


def paired_bytes(counts, probability, decode):
    output = bytearray()
    for c, p, d in zip(counts, probability, decode):
        fields = (
            c.astype("<i2"),
            np.log(p).astype("<f4"),
            p.astype("<f4"),
            d.astype("<i2"),
        )
        output.extend(struct.pack("<4I", *(row.nbytes for row in fields)))
        for row in fields:
            output.extend(row.tobytes())
    return bytes(output)


class SignatureAblationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rng = np.random.default_rng(58)
        self.counts = rng.integers(0, 4, size=(32, 3, 8), dtype=np.int16)
        self.probability = rng.random((32, 3, 8), dtype=np.float32) + 0.01
        self.probability /= self.probability.sum(-1, keepdims=True)
        self.decode = rng.integers(0, 4, size=(32, 3, 8), dtype=np.int16)
        self.source = self.root / "paired.bin"
        self.raw = paired_bytes(self.counts, self.probability, self.decode)
        self.source.write_bytes(self.raw)

    def test_parser_retains_every_record_and_rejects_corruption(self):
        actual = read_paired_capture(self.source, 3, 8)
        for got, expected in zip(actual, (self.counts, self.probability, self.decode)):
            np.testing.assert_array_equal(got, expected)
        for bad in (b"", self.raw[:-1], b"bad!" + self.raw[4:]):
            self.source.write_bytes(bad)
            with self.assertRaises(ValueError):
                read_paired_capture(self.source, 3, 8)
        bad_probability = self.probability.copy()
        for value in (-1, float("nan"), float("inf")):
            bad_probability[0, 0, 0] = value
            with np.errstate(invalid="ignore"):
                self.source.write_bytes(
                    paired_bytes(self.counts, bad_probability, self.decode)
                )
            with self.assertRaises(ValueError):
                read_paired_capture(self.source, 3, 8)

    def test_greedy_and_both_fits_are_reproducible_and_non_mutating(self):
        original = self.probability.copy()
        self.assertEqual(
            greedy_mask(self.probability, self.decode, pairs=1000),
            greedy_mask(self.probability, self.decode, pairs=1000, seed=1),
        )
        np.testing.assert_array_equal(self.probability, original)
        with patch.dict(
            "eldr.experiments.signature_ablation.MODEL_GEOMETRY", {"test": (3, 8, 16)}
        ):
            for variant in ("count_idf", "gate_prob_all"):
                path = self.root / f"{variant}.json"
                result = fit_signature(self.source, path, "test", variant, k=2)
                loaded = load_centroid_file(path)
                self.assertEqual(result["n_fit"], 32)
                self.assertEqual(result["seed"], 1)
                signal = self.counts if variant == "count_idf" else self.probability
                vectors = build_signature(loaded, signal)
                expected = (
                    signal[:, result["layer_mask"]].astype(np.float32).reshape(32, -1)
                )
                if variant == "count_idf":
                    expected *= np.asarray(result["sig_idf"], dtype=np.float32)
                expected /= np.linalg.norm(expected, axis=1, keepdims=True)
                np.testing.assert_array_equal(vectors, expected)
                np.testing.assert_allclose(
                    np.linalg.norm(loaded["centroid_matrix"], axis=1), 1, atol=1e-6
                )
                with self.assertRaises(ValueError):
                    fit_signature(self.source, path, "test", variant, k=2)
        self.assertEqual(self.source.read_bytes(), self.raw)

    def test_gate_wire_format_and_metadata_fail_closed(self):
        metadata = dict(
            L=3,
            e=8,
            k=1,
            variant="gate_prob_all",
            sig_dtype="<f4",
            layer_mask=[1],
            centroids=[[1.0] + [0.0] * 7],
        )
        path = self.root / "gate.json"
        path.write_text(json.dumps(metadata))
        app = {"centroid_data": load_centroid_file(path)}
        probability = self.probability[0]
        encoded = pybase64.b64encode(probability.astype("<f4")).decode()
        np.testing.assert_array_equal(decode_signature(app, encoded), probability)
        for payload in (
            pybase64.b64encode(self.counts[0].astype("<i2")).decode(),
            pybase64.b64encode(np.full((3, 8), np.nan, dtype="<f4")).decode(),
        ):
            with self.assertRaises(web.HTTPInternalServerError):
                decode_signature(app, payload)
        for change in (dict(sig_dtype="<i2"), dict(sig_idf=[1] * 8)):
            path.write_text(json.dumps(metadata | change))
            with self.assertRaises(ValueError):
                load_centroid_file(path)


if __name__ == "__main__":
    unittest.main()
