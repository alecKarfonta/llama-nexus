"""Unit tests for MTP benchmark helpers (scripts/mtp_bench_lib.py)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LIB_PATH = _REPO / "scripts" / "mtp_bench_lib.py"


def _load_lib():
    spec = importlib.util.spec_from_file_location("mtp_bench_lib", _LIB_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtp_bench_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestMtpBenchLib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = _load_lib()

    def test_infer_model_family_qwen(self):
        self.assertEqual(self.lib.infer_model_family("Qwen3.6-27B-Q6_K.gguf"), "qwen3.6")

    def test_infer_model_family_glm(self):
        self.assertEqual(self.lib.infer_model_family("GLM-4.6-MTP-Q4_K_M.gguf"), "glm")

    def test_expand_env_path_default(self):
        path = self.lib.expand_env_path("${MODELS_DIR:-./models}/foo.gguf")
        self.assertTrue(path.endswith("/foo.gguf") or path.endswith("\\foo.gguf"))

    def test_parse_mtp_log_line(self):
        line = "draft acceptance rate = 0.76482 ( 3483 accepted / 4554 generated)"
        stats = self.lib.parse_mtp_stats_from_log_line(line)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertAlmostEqual(stats["acceptance_rate"], 0.76482, places=4)
        self.assertEqual(stats["tokens_accepted"], 3483)

    def test_iter_matrix_cases_counts(self):
        matrix = {
            "models": [{"id": "m1", "path": "/tmp/a.gguf", "family": "qwen3.6"}],
            "parallel_slots": [1],
            "mtp_draft_n_max": [2, 3],
            "mtp_draft_p_min": [0.5, 0.75],
            "cache_types": ["q8_0"],
            "run_baseline": True,
            "repetitions": 1,
        }
        cases = self.lib.iter_matrix_cases(matrix)
        # baseline 1 + mtp 2*2 = 5
        self.assertEqual(len(cases), 5)
        self.assertFalse(cases[0]["mtp_enabled"])
        self.assertTrue(cases[1]["mtp_enabled"])


if __name__ == "__main__":
    unittest.main()
