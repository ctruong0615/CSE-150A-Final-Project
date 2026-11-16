"""Synthetic edge cases, not Steam observations or project results."""
import csv
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

from preprocess import STATES, convert, mapped_columns, owner_midpoint, preprocess, split_rows

BASE = {"AppID": "10", "Name": "Synthetic game", "Price": "9.99", "Release date": "Jan 1, 2020",
        "Positive": "80", "Negative": "20", "Estimated owners": "0 - 20000", "Peak CCU": "100",
        "Genres": "Indie,RPG", "About the game": "<p>A fictional example.</p>"}
COLUMNS = mapped_columns(BASE)


class PreprocessingTests(unittest.TestCase):
    def convert(self, **changes):
        return convert({**BASE, **changes}, COLUMNS, 100, 0.8, date(2026, 11, 15))

    def test_label_threshold_and_genres(self):
        row, reason = self.convert()
        self.assertIsNone(reason)
        self.assertEqual((row["success"], row["is_indie"], row["is_rpg"]), ("yes", "yes", "yes"))
        self.assertEqual(self.convert(Positive="79", Negative="21")[0]["success"], "no")
        self.assertEqual(self.convert(Positive="79", Negative="20")[1], "below_minimum_reviews")

    def test_invalid_and_missing_values(self):
        for value in ("nan", "inf", "-1", ""):
            self.assertEqual(self.convert(Price=value)[1], "invalid_price")
        self.assertEqual(self.convert(Positive="2.5")[1], "invalid_review_counts")
        self.assertEqual(self.convert(**{"Release date": "Jan 1, 2030"})[1], "unreleased_as_of_snapshot")
        self.assertEqual(self.convert(**{"Release date": "Coming soon"})[1], "invalid_release_date")

    def test_optional_missing_is_unknown(self):
        row, _ = self.convert(**{"Estimated owners": "", "Peak CCU": "", "Genres": "", "About the game": ""})
        for field in ("owners_band", "peak_ccu_band", "is_indie", "is_rpg", "description_length"):
            self.assertEqual(row[field], "unknown")

    def test_price_boundaries(self):
        for value, expected in (("0", "free"), ("9.99", "under_10"), ("10", "10_to_30"), ("30", "30_plus")):
            self.assertEqual(self.convert(Price=value)[0]["price_band"], expected)

    def test_owner_ranges(self):
        self.assertEqual(owner_midpoint("20,000 - 50,000"), 35000)
        self.assertIsNone(owner_midpoint("50000 - 20000"))

    def test_required_schema(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            mapped_columns(["AppID", "Price", "Release date", "Negative"])

    def test_split_is_deterministic_disjoint_and_stratified(self):
        rows = [{"app_id": i, "success": "yes" if i % 2 else "no"} for i in range(100)]
        train, test = split_rows(rows, 0.2, 42)
        self.assertEqual((train, test), split_rows(list(reversed(rows)), 0.2, 42))
        self.assertEqual(len(test), 20)
        self.assertEqual(sum(r["success"] == "yes" for r in test), 10)
        self.assertFalse({r["app_id"] for r in train} & {r["app_id"] for r in test})

    def test_end_to_end_audit_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.csv"
            rows = [{**BASE, "AppID": str(i+1), "Positive": "90" if i % 2 else "50",
                     "Negative": "10" if i % 2 else "50"} for i in range(20)]
            rows.extend([rows[0], {**BASE, "AppID": "999", "Price": "-1"}])
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(BASE))
                writer.writeheader()
                writer.writerows(rows)
            meta = preprocess(source, root / "one")
            preprocess(source, root / "two")
            self.assertEqual(meta["eligible_rows"], 20)
            self.assertEqual(meta["audit"]["duplicate_app_id"], 1)
            self.assertEqual(meta["audit"]["invalid_price"], 1)
            self.assertEqual(sum(v for k, v in meta["audit"].items() if k != "input_rows") + 20, 22)
            with (root / "one/train.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(STATES))
                for row in reader:
                    self.assertTrue(all(row[k] in STATES[k] for k in STATES))
            for name in ("train.csv", "test.csv", "train_ids.csv", "test_ids.csv", "metadata.json"):
                self.assertEqual((root / "one" / name).read_bytes(), (root / "two" / name).read_bytes())
            json.loads((root / "one/metadata.json").read_text())

    def test_empty_eligible_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "empty.csv"
            source.write_text(",".join(BASE) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No eligible"):
                preprocess(source, Path(tmp) / "out")

    def test_known_merged_header_is_repaired_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "merged.csv"
            original = list(BASE)
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(original[:3] + ["DiscountDLC count"] + original[3:])
                for i in range(10):
                    row = {**BASE, "AppID": str(i+1)}
                    values = [row[k] for k in original]
                    writer.writerow(values[:3] + ["0", "2"] + values[3:])
            meta = preprocess(source, root / "out")
            self.assertEqual(meta["eligible_rows"], 10)
            self.assertEqual(meta["class_counts"]["all"], {"yes": 10})
            self.assertEqual(len(meta["header_repairs"]), 1)

    def test_unexpected_row_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "broken.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(BASE)
                writer.writerow(list(BASE.values()) + ["extra unexpected field"])
            with self.assertRaisesRegex(ValueError, "fields; expected"):
                preprocess(source, root / "out")


if __name__ == "__main__":
    unittest.main()
