import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Flux1" / "scripts" / "eval_multi.sh"
DATA_ROOT = ROOT / "data"


def parse_targets(script_text, erase_type):
    match = re.search(
        rf'targets_map\["{re.escape(erase_type)}"\]="\\\n(.*?)\\\n"',
        script_text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError(f"missing targets_map entry for {erase_type}")
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class Flux1EvalMultiContractTest(unittest.TestCase):
    def test_targets_match_csv_erase_concepts_in_order(self):
        script_text = SCRIPT.read_text()
        for erase_type in ("10_celebrity", "50_celebrity", "100_celebrity"):
            rows = csv_rows(DATA_ROOT / f"{erase_type}.csv")
            concepts = []
            for row in rows:
                if row["type"] == "erase" and row["concept"] not in concepts:
                    concepts.append(row["concept"])
            self.assertEqual(parse_targets(script_text, erase_type), concepts)

    def test_celebrity_csv_files_have_no_embedded_header_rows(self):
        for erase_type in ("10_celebrity", "50_celebrity", "100_celebrity"):
            rows = csv_rows(DATA_ROOT / f"{erase_type}.csv")
            bad_rows = [row for row in rows if row["id"] == "id" or row["type"] == "type"]
            self.assertEqual(bad_rows, [])


if __name__ == "__main__":
    unittest.main()
