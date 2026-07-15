#!/usr/bin/env python3
"""
Unpacks a GBIF Darwin Core Archive (the .zip you get emailed) and downloads
every StillImage into data/images/<out-subdir>/, writing a metadata CSV
alongside it. Use --out-subdir to keep multiple species separate.

Usage:
    python scripts/01_extract_gbif_images.py /path/to/gbif_download.zip
    python scripts/01_extract_gbif_images.py /path/to/already_unzipped_folder
    python scripts/01_extract_gbif_images.py aa --out-subdir aa   # second species
"""
import sys
import csv
import re
import hashlib
import zipfile
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# Some GBIF multimedia.txt exports are ragged: a subset of rows (often from
# one particular source dataset) carry extra fields not declared in the
# header - e.g. an inserted datasetKey - which throws off column position for
# just those rows, and can even make pandas.read_csv silently absorb the
# extra leading fields into a MultiIndex. Rather than trust column position
# at all, recover the image URL and gbifID by pattern-matching every raw
# field in the row, regardless of where it landed.
IMAGE_URL_RE = re.compile(r"https?://\S+\.(?:jpg|jpeg|png|tif|tiff)\b", re.IGNORECASE)
GBIF_ID_RE = re.compile(r"^\d{6,12}$")


def recover_url_and_id(fields, fallback_id: str):
    """Find an image URL and gbifID anywhere in a raw TSV row, tolerant of column shift."""
    url = None
    for v in fields:
        m = IMAGE_URL_RE.search(v)
        if m:
            url = m.group(0)
            break
    gbif_id = next((v for v in fields if GBIF_ID_RE.match(v)), fallback_id)
    return url, gbif_id


def unpack(zip_path: Path, tag: str) -> Path:
    raw_dir = ROOT / "data" / "raw_gbif" / tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return raw_dir


def load_multimedia_rows(raw_dir: Path):
    """Read multimedia.txt as raw rows (list of field lists), bypassing pandas'
    column alignment - see the note on ragged rows above."""
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

    images_dir = ROOT / "data" / "images" / args.out_subdir
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
