import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.synth_resume import build_signature


class TestSynthResume(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _write_jsonl(self, path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _run_generate(self, config_path: Path):
        subprocess.run(
            [sys.executable, "-m", "src.data.generate_synthetic", "--config", str(config_path)],
            cwd=self.repo_root,
            check=True,
        )

    def test_resume_appends_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "raw.jsonl"
            output_path_template = str(tmp_path / "out_{run_id}.jsonl")
            checkpoint_path = tmp_path / "resume.json"

            text = "The Lancer RPG is a game about mechs and pilots. " * 20
            chunks = [
                {"page": 1, "text": text, "source": "book"},
                {"page": 2, "text": text, "source": "book"},
            ]
            self._write_jsonl(input_path, chunks)

            config = {
                "project_name": "lancer",
                "dataset_tag": "test",
                "ingest": {"raw_output_path": str(input_path)},
                "output": {
                    "path": output_path_template,
                    "run_id": "auto",
                    "append": False,
                    "flush_every": 1,
                },
                "generation": {"shuffle": False, "shuffle_seed": 1337},
                "limits": {"enforce_max_samples": False},
                "resume": {
                    "enabled": True,
                    "checkpoint_path": str(checkpoint_path),
                    "allow_mismatch": False,
                    "force_restart": False,
                },
                "task_types": ["rules_qa"],
                "llm": {"repair_invalid_json": True},
            }

            run_id = "resume"
            output_path = Path(output_path_template.format(run_id=run_id))
            initial_record = {
                "instruction": "old",
                "input": "",
                "output": "old",
                "task_type": "rules_qa",
                "source_page": 1,
            }
            self._write_jsonl(output_path, [initial_record, initial_record])

            signature = build_signature(
                config,
                str(input_path),
                str(output_path),
                ["rules_qa"],
                False,
                1337,
            )
            checkpoint = {
                "run_id": run_id,
                "signature": signature,
                "next_index": 1,
                "stats": {
                    "samples_written": 2,
                    "long_samples": 0,
                    "skipped_low_signal": 0,
                    "table_like_pages": 0,
                    "processed_pages": 1,
                    "requested_questions": 2,
                },
            }
            checkpoint_path.write_text(json.dumps(checkpoint))

            config_path = tmp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            self._run_generate(config_path)

            lines = output_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 4)
            first = json.loads(lines[0])
            self.assertEqual(first["instruction"], "old")

    def test_signature_mismatch_restarts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "raw.jsonl"
            output_path_template = str(tmp_path / "out_{run_id}.jsonl")
            checkpoint_path = tmp_path / "resume.json"

            text = "The Lancer RPG is a game about mechs and pilots. " * 20
            chunks = [{"page": 1, "text": text, "source": "book"}]
            self._write_jsonl(input_path, chunks)

            config = {
                "project_name": "lancer",
                "dataset_tag": "test",
                "ingest": {"raw_output_path": str(input_path)},
                "output": {
                    "path": output_path_template,
                    "run_id": "fixed",
                    "append": False,
                    "flush_every": 1,
                },
                "generation": {"shuffle": False, "shuffle_seed": 1337},
                "limits": {"enforce_max_samples": False},
                "resume": {
                    "enabled": True,
                    "checkpoint_path": str(checkpoint_path),
                    "allow_mismatch": False,
                    "force_restart": False,
                },
                "task_types": ["rules_qa"],
                "llm": {"repair_invalid_json": True},
            }

            output_path = Path(output_path_template.format(run_id="fixed"))
            old_record = {
                "instruction": "old",
                "input": "",
                "output": "old",
                "task_type": "rules_qa",
                "source_page": 1,
            }
            self._write_jsonl(output_path, [old_record, old_record])

            checkpoint = {
                "run_id": "fixed",
                "signature": "mismatch",
                "next_index": 1,
                "stats": {"samples_written": 2},
            }
            checkpoint_path.write_text(json.dumps(checkpoint))

            config_path = tmp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            self._run_generate(config_path)

            lines = output_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertNotEqual(first["instruction"], "old")

    def test_signature_changes_with_rag_ingest(self):
        config = {
            "project_name": "lancer",
            "dataset_tag": "test",
            "ingest": {"raw_output_path": "raw.jsonl"},
            "output": {"path": "out.jsonl"},
            "task_types": ["rules_qa"],
            "generation": {"shuffle": False, "shuffle_seed": 1337},
            "limits": {"enforce_max_samples": False},
            "rag_mode": {"enabled": False},
            "rag_ingest": {"enabled": False},
        }
        sig_a = build_signature(
            config,
            "raw.jsonl",
            "out.jsonl",
            ["rules_qa"],
            False,
            1337,
        )
        config["rag_ingest"] = {"enabled": True, "chunks_output_path": "chunks.jsonl"}
        sig_b = build_signature(
            config,
            "raw.jsonl",
            "out.jsonl",
            ["rules_qa"],
            False,
            1337,
        )
        self.assertNotEqual(sig_a, sig_b)


if __name__ == "__main__":
    unittest.main()
