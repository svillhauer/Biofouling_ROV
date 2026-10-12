#!/usr/bin/env python3
"""
Unpacks a GBIF Darwin Core Archive (the .zip you get emailed) and downloads
every StillImage into data/images/<out-subdir>/, writing a metadata CSV
alongside it. Use --out-subdir to keep multiple species separate.

Usage:
    python scripts/data/extract_gbif_images.py /path/to/gbif_download.zip
    python scripts/data/extract_gbif_images.py /path/to/already_unzipped_folder
    python scripts/data/extract_gbif_images.py aa --out-subdir aa   # second species
"""
import argparse
import csv
import hashlib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from biofouling.gbif import recover_url_and_id
from biofouling.paths import DATA_DIR, IMAGES_DIR


def unpack(zip_path: Path, tag: str) -> Path:
    raw_dir = DATA_DIR / "raw_gbif" / tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return raw_dir


def load_multimedia_rows(raw_dir: Path):
    """Read multimedia.txt as raw rows (list of field lists), bypassing pandas'
    column alignment - see the note on ragged rows in biofouling.gbif."""
    for name in ("multimedia.txt", "Multimedia.txt", "verbatim_multimedia.txt"):
        p = raw_dir / name
        if p.exists():
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                next(reader)  # header - unreliable column count for ragged rows, not used
                return [row for row in reader if row]
    raise FileNotFoundError(
        f"Could not find multimedia.txt in {raw_dir}. Files present: "
        f"{[p.name for p in raw_dir.iterdir()]}"
    )


def load_occurrence(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "occurrence.txt"
    if not p.exists():
        raise FileNotFoundError(f"Could not find occurrence.txt in {raw_dir}")
    return pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, on_bad_lines="skip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path, help="Path to the GBIF download .zip, or an already-unzipped folder")
    ap.add_argument("--limit", type=int, default=None, help="Only download the first N images (for a quick test run)")
    ap.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    ap.add_argument("--out-subdir", default="all",
                    help="Subfolder under data/images/ to save into - use a distinct name per species")
    args = ap.parse_args()

    images_dir = IMAGES_DIR / args.out_subdir
    meta_csv = images_dir / "metadata.csv"

    if args.zip_path.is_dir():
        print(f"{args.zip_path} is a directory, using it directly as the raw DwC-A folder.")
        raw_dir = args.zip_path
    else:
        raw_dir = unpack(args.zip_path, args.out_subdir)
        print(f"Unpacked {args.zip_path} -> {raw_dir}")

    raw_rows = load_multimedia_rows(raw_dir)
    records = []
    for idx, fields in enumerate(raw_rows):
        url, gbif_id = recover_url_and_id(fields, f"row{idx}")
        if url:
            records.append({"identifier": url, "gbifID": gbif_id})
    media = pd.DataFrame(records)
    if len(media) < len(raw_rows):
        print(f"{len(raw_rows) - len(media)}/{len(raw_rows)} multimedia rows had no recoverable image URL and were "
              f"skipped (kept {len(media)}). This is common with messy/ragged GBIF exports.")

    occ = load_occurrence(raw_dir)
    occ_cols = [c for c in ["gbifID", "scientificName", "locality", "country",
                             "eventDate", "recordedBy", "decimalLatitude",
                             "decimalLongitude", "basisOfRecord"] if c in occ.columns]
    occ = occ[occ_cols]

    merged = media.merge(occ, on="gbifID", how="left")
    if args.limit:
        merged = merged.head(args.limit)

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(merged)} images to {images_dir} with {args.workers} workers")

    def download_one(row):
        url = row["identifier"]
        gbif_id = row["gbifID"]
        ext = ".jpg"
        if isinstance(url, str) and "." in url.rsplit("/", 1)[-1]:
            suffix = "." + url.rsplit(".", 1)[-1].split("?")[0].lower()
            if suffix in (".jpg", ".jpeg", ".png"):
                ext = suffix
        # Stable hash of the URL (not a positional row index) so the same image
        # always gets the same filename across re-runs, regardless of ordering
        # or how many other rows were recovered - this is what makes the
        # fpath.exists() dedup check below actually work on a second run.
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        fname = f"{gbif_id}_{url_hash}{ext}"
        fpath = images_dir / fname
        meta = {"filename": fname, "gbifID": gbif_id, **{c: row.get(c) for c in occ_cols}}
        if fpath.exists():
            return meta
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            fpath.write_bytes(resp.content)
            return meta
        except Exception as e:
            tqdm.write(f"FAILED {url}: {e}")
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row) for _, row in merged.iterrows()]
        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result is not None:
                rows.append(result)

    out = pd.DataFrame(rows)
    if meta_csv.exists():
        prior = pd.read_csv(meta_csv, dtype=str)
        out = pd.concat([prior, out]).drop_duplicates(subset="filename", keep="last")
    out.to_csv(meta_csv, index=False)
    print(f"Done. {len(out)} images on record. Metadata -> {meta_csv}")


if __name__ == "__main__":
    main()
