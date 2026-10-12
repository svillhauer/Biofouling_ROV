"""Parsing helpers for GBIF Darwin Core Archive exports."""
import re

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
