"""Download a pinned public Kaggle dataset version without storing credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=59)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.version < 1:
        parser.error("Version must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "games.csv"
    if destination.exists():
        parser.error("games.csv already exists; preserve it or select another output directory")
    url = f"https://www.kaggle.com/api/v1/datasets/download/fronkongames/steam-games-dataset?datasetVersionNumber={args.version}"
    archive = args.output / "steam_download.zip.part"
    request = urllib.request.Request(url, headers={"User-Agent": "CSE150A-course-project/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if not zipfile.is_zipfile(archive):
        raise ValueError("Kaggle did not return a ZIP archive. Download manually from the dataset page.")
    with zipfile.ZipFile(archive) as dataset:
        candidates = [entry for entry in dataset.infolist() if Path(entry.filename).name.lower() == "games.csv"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one games.csv; archive contains {dataset.namelist()}")
        entry = candidates[0]
        with dataset.open(entry) as source, destination.open("wb") as stream:
            shutil.copyfileobj(source, stream, length=1024 * 1024)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"dataset": "fronkongames/steam-games-dataset", "version": args.version,
                "download_url": url, "sha256": digest.hexdigest(), "bytes": destination.stat().st_size}
    (args.output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    archive.unlink()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
