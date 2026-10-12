import json
import tempfile
import unittest
from pathlib import Path

from data.prepare_dataset import content_hash, normalize_text, prepare_dataset, split_for_hash


class DatasetPipelineTests(unittest.TestCase):
    def test_normalization_is_stable(self):
        raw = "A\r\nB   \r\n\r\n\r\n\r\nC\n"
        self.assertEqual(normalize_text(raw), "A\nB\n\n\nC")

    def test_split_is_deterministic(self):
        digest = content_hash("same document")
        self.assertEqual(split_for_hash(digest, 42), split_for_hash(digest, 42))

    def test_prepare_dataset_deduplicates_documents(self):
        with tempfile.TemporaryDirectory() as raw_tmp, tempfile.TemporaryDirectory() as out_tmp:
            raw_dir = Path(raw_tmp)
            output_dir = Path(out_tmp)

            repeated = "This is a sufficiently long document. " * 20
            unique = "This is another sufficiently long document. " * 20

            (raw_dir / "a.txt").write_text(repeated, encoding="utf-8")
            (raw_dir / "b.txt").write_text(repeated, encoding="utf-8")
            (raw_dir / "c.txt").write_text(unique, encoding="utf-8")

            stats = prepare_dataset(raw_dir, output_dir, min_chars=10, seed=42)

            self.assertEqual(stats["input_documents"], 3)
            self.assertEqual(stats["kept_documents"], 2)
            self.assertEqual(stats["duplicates_removed"], 1)

            rows = []
            for split in ("train", "validation", "test"):
                path = output_dir / f"{split}.jsonl"
                self.assertTrue(path.exists())
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))

            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["content_hash"] for row in rows}), 2)

    def test_jsonl_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as raw_tmp, tempfile.TemporaryDirectory() as out_tmp:
            raw_dir = Path(raw_tmp)
            output_dir = Path(out_tmp)
            payload = {
                "id": "doc-1",
                "source": "unit-test",
                "language": "sk",
                "license": "CC0-1.0",
                "text": "Slovenský testovací dokument. " * 20,
            }
            (raw_dir / "docs.jsonl").write_text(
                json.dumps(payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            prepare_dataset(raw_dir, output_dir, min_chars=10, seed=42)
            rows = []
            for split in ("train", "validation", "test"):
                for line in (output_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["doc_id"], "doc-1")
            self.assertEqual(rows[0]["language"], "sk")
            self.assertEqual(rows[0]["license"], "CC0-1.0")


if __name__ == "__main__":
    unittest.main()
