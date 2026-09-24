import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eldr.serving.clustering import load_centroid_file
from eldr.serving.routing import route_jsq


class TestFittingArtifact(unittest.TestCase):
    def _artifact(self, path: Path, *, metadata_k: int = 2) -> None:
        path.write_text(
            json.dumps(
                {
                    "k": metadata_k,
                    "L": 1,
                    "e": 2,
                    "layer_mask": [0],
                    "centroids": [[1.0, 0.0], [0.0, 1.0]],
                }
            ),
            encoding="utf-8",
        )

    def test_metadata_count_must_match_centroid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            self._artifact(path, metadata_k=3)
            with self.assertRaisesRegex(ValueError, "metadata k=3"):
                load_centroid_file(path)

    def test_static_route_requires_one_centroid_per_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            self._artifact(path)
            centroid_data = load_centroid_file(path)
            signature = np.asarray([[1, 0]], dtype=np.int16)
            with self.assertRaisesRegex(ValueError, "one centroid per decoder"):
                route_jsq(centroid_data, signature, [0])


if __name__ == "__main__":
    unittest.main()
