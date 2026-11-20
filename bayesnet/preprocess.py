"""Convert the Kaggle Steam CSV into discrete Bayesian-network observations.

Standard-library only. Run `python preprocess.py --help` for usage.
The thresholds are provisional project choices, not Steam's rating rules.
"""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import sys

STATES = {
    "price_band": ["free", "under_10", "10_to_30", "30_plus"],
    "release_era": ["2010_or_earlier", "2011_to_2015", "2016_to_2020", "2021_plus"],
    "owners_band": ["under_20k", "20k_to_100k", "100k_plus", "unknown"],
    "peak_ccu_band": ["under_100", "100_to_1000", "1000_plus", "unknown"],
    "description_length": ["under_100_words", "100_to_300_words", "300_plus_words", "unknown"],
    "is_indie": ["no", "yes", "unknown"],
    "is_rpg": ["no", "yes", "unknown"],
    "success": ["no", "yes"],
}
ALIASES = {
    "app_id": ["appid", "steam_appid", "app_id"],
    "name": ["name"],
    "price": ["price"],
    "release_date": ["release_date"],
    "positive": ["positive", "positive_ratings"],
    "negative": ["negative", "negative_ratings"],
    "owners": ["estimated_owners", "owners"],
    "peak_ccu": ["peak_ccu"],
    "genres": ["genres", "genre"],
    "description": ["about_the_game", "detailed_description", "short_description"],
}
REQUIRED = {"app_id", "price", "release_date", "positive", "negative"}


def normalized(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def number(value):
    try:
        result = float(str(value).strip().replace(",", ""))
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def integer(value):
    result = number(value)
    return int(result) if result is not None and result.is_integer() else None


def release_date(value):
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%b %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def owner_midpoint(value):
    # Kaggle uses strings such as '0 - 20000'; preserve the estimate's uncertainty.
    text = str(value).strip().replace(",", "")
    match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", text)
    if match:
        lo, hi = map(int, match.groups())
        return (lo + hi) / 2 if hi >= lo else None
    return number(text)


def mapped_columns(fieldnames):
    source = {normalized(name): name for name in fieldnames or []}
    mapping = {key: next((source[a] for a in aliases if a in source), None)
               for key, aliases in ALIASES.items()}
    missing = sorted(key for key in REQUIRED if mapping[key] is None)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return mapping


def convert(row, columns, min_reviews, positive_threshold, as_of):
    def get(key):
        return row.get(columns[key], "") or "" if columns[key] else ""

    app_id = integer(get("app_id"))
    positive, negative = integer(get("positive")), integer(get("negative"))
    price, released = number(get("price")), release_date(get("release_date"))
    if app_id is None or app_id <= 0:
        return None, "invalid_app_id"
    if positive is None or negative is None:
        return None, "invalid_review_counts"
    if price is None:
        return None, "invalid_price"
    if released is None:
        return None, "invalid_release_date"
    if released > as_of:
        return None, "unreleased_as_of_snapshot"
    total = positive + negative
    if total < min_reviews:
        return None, "below_minimum_reviews"
    owners, ccu = owner_midpoint(get("owners")), integer(get("peak_ccu"))
    description = html.unescape(re.sub(r"<[^>]*>", " ", get("description")))
    words = len(description.split())
    genres = {part.strip().lower() for part in re.split(r"[,;]", get("genres")) if part.strip()}
    ratio = positive / total
    features = {
        "price_band": "free" if price == 0 else "under_10" if price < 10 else "10_to_30" if price < 30 else "30_plus",
        "release_era": "2010_or_earlier" if released.year <= 2010 else "2011_to_2015" if released.year <= 2015 else "2016_to_2020" if released.year <= 2020 else "2021_plus",
        "owners_band": "unknown" if owners is None else "under_20k" if owners < 20000 else "20k_to_100k" if owners < 100000 else "100k_plus",
        "peak_ccu_band": "unknown" if ccu is None else "under_100" if ccu < 100 else "100_to_1000" if ccu < 1000 else "1000_plus",
        "description_length": "unknown" if not words else "under_100_words" if words < 100 else "100_to_300_words" if words < 300 else "300_plus_words",
        "is_indie": "unknown" if not genres else "yes" if "indie" in genres else "no",
        "is_rpg": "unknown" if not genres else "yes" if "rpg" in genres else "no",
        "success": "yes" if ratio >= positive_threshold else "no",
    }
    return {"app_id": app_id, "name": get("name"), "genres": get("genres"), "price": price,
            "release_date": released.isoformat(), "total_reviews": total,
            "positive_fraction": ratio, "owners_midpoint": owners,
            "peak_ccu": ccu, "description_words": words, **features}, None


def split_rows(rows, test_fraction, seed):
    """Stratify by target; keep singleton classes in training and report counts."""
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["app_id"]):
        groups[row["success"]].append(row)
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(groups):
        group = groups[label][:]
        rng.shuffle(group)
        n = max(1, min(len(group) - 1, round(len(group) * test_fraction))) if len(group) > 1 else 0
        test.extend(group[:n])
        train.extend(group[n:])
    return sorted(train, key=lambda r: r["app_id"]), sorted(test, key=lambda r: r["app_id"])


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_chart(path, title, counts):
    """Portable SVG bar chart; no plotting dependency needed."""
    entries = list(counts.items())
    height = 90 + len(entries) * 32
    maximum = max(counts.values(), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="780" height="{height}" viewBox="0 0 780 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="24" y="32" font-family="sans-serif" font-size="19">{html.escape(title)}</text>']
    for i, (label, count) in enumerate(entries):
        y = 65 + i * 32
        width = round(400 * count / maximum)
        parts.extend([f'<text x="24" y="{y+15}" font-family="sans-serif" font-size="13">{html.escape(str(label))}</text>',
                      f'<rect x="240" y="{y}" width="{width}" height="21" fill="#2563eb"/>',
                      f'<text x="{250+width}" y="{y+15}" font-family="sans-serif" font-size="13">{count:,}</text>'])
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def preprocess(input_path, output_dir, min_reviews=100, positive_threshold=0.8,
               test_fraction=0.2, seed=42, as_of=date(2026, 11, 15)):
    if min_reviews < 1 or not 0 < positive_threshold <= 1 or not 0 < test_fraction < 1:
        raise ValueError("Require min_reviews >= 1, 0 < positive_threshold <= 1, 0 < test_fraction < 1")
    # Steam descriptions can exceed Python's conservative default CSV field limit.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    counts, seen, rows = Counter(), set(), []
    header_repairs = []
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        original_fields = next(reader, [])
        fields = []
        for field in original_fields:
            if field == "DiscountDLC count":
                fields.extend(["Discount", "DLC count"])
                header_repairs.append("Split merged 'DiscountDLC count' into 'Discount' and 'DLC count'.")
            else:
                fields.append(field)
        if len({normalized(field) for field in fields}) != len(fields):
            raise ValueError("Duplicate normalized column names in CSV header")
        columns = mapped_columns(fields)
        for values in reader:
            if not values:
                counts["blank_records"] += 1
                continue
            if len(values) != len(fields):
                raise ValueError(f"CSV record ending at line {reader.line_num} has {len(values)} fields; expected {len(fields)}. Check the source header before proceeding.")
            raw = dict(zip(fields, values))
            counts["input_rows"] += 1
            identifier = integer(raw.get(columns["app_id"]))
            # First occurrence wins even when invalid: avoid cherry-picking duplicate records.
            if identifier is not None and identifier in seen:
                counts["duplicate_app_id"] += 1
                continue
            if identifier is not None:
                seen.add(identifier)
            cleaned, reason = convert(raw, columns, min_reviews, positive_threshold, as_of)
            if reason:
                counts[reason] += 1
            else:
                rows.append(cleaned)
    if not rows:
        raise ValueError("No eligible games remain. Check the input schema and thresholds.")
    train, test = split_rows(rows, test_fraction, seed)
    if not test:
        raise ValueError("Not enough eligible games for a nonempty test set.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cleaned_games.csv", rows, list(rows[0]))
    for name, subset in (("train", train), ("test", test)):
        write_csv(output_dir / f"{name}.csv", subset, list(STATES))
        write_csv(output_dir / f"{name}_ids.csv", subset, ["app_id"])
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "source": "https://www.kaggle.com/datasets/fronkongames/steam-games-dataset",
        "input_file": input_path.name, "sha256": digest.hexdigest(),
        "settings": {"min_reviews": min_reviews, "positive_threshold": positive_threshold,
                     "test_fraction": test_fraction, "seed": seed, "as_of": as_of.isoformat()},
        "column_mapping": columns, "header_repairs": header_repairs,
        "original_columns": original_fields, "effective_columns": fields,
        "states": STATES, "audit": dict(counts),
        "eligible_rows": len(rows), "train_rows": len(train), "test_rows": len(test),
        "class_counts": {name: dict(Counter(r["success"] for r in subset))
                         for name, subset in (("all", rows), ("train", train), ("test", test))},
        "warnings": ["Thresholds are provisional; review before final experiments.",
                     "Owners and CCU are post-release observations; results are associational.",
                     "Low-review games are excluded, so conclusions apply to the eligible population.",
                     "Missing optional variables use an explicit unknown state."]
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    for feature in STATES:
        histogram = Counter(row[feature] for row in rows)
        write_chart(figures / f"{feature}.svg", f"Eligible games: {feature}",
                    {state: histogram[state] for state in STATES[feature]})
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Extracted Kaggle games.csv")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-reviews", type=int, default=100)
    parser.add_argument("--positive-threshold", type=float, default=0.8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 11, 15))
    args = parser.parse_args()
    try:
        metadata = preprocess(args.input, args.output, args.min_reviews, args.positive_threshold,
                              args.test_fraction, args.seed, args.as_of)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")
    print(json.dumps({k: metadata[k] for k in ("audit", "eligible_rows", "train_rows", "test_rows", "class_counts")}, indent=2))


if __name__ == "__main__":
    main()
