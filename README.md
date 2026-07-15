# Biofouling ROV detector — YOLOv11-seg pipeline

Goal: an instance-segmentation model that spots biofouling organisms on a
harbor sea wall in ROV video, and identifies **which** organism it is. This
project is adapted from the earlier single-species `Rugolopteryx okamurae`
pipeline (see `../MRSS/Rugolopteryx okamurae/`), generalized to a multi-class
setup covering four organisms:

```
0: rugulopteryx_okamurae     (invasive brown alga)
1: asparagopsis_armata       (invasive red alga, visual look-alike to #0)
2: botrylloides_violaceus    (invasive colonial tunicate)
3: hildenbrandia_rubra       (crustose encrusting red alga)
```

(`lithophyllum_incrustans` was considered and dropped before labeling
started - not needed for this project. `hymeniacidon_perlevis` was trained
briefly as class 3 but removed - what was thought to be encrusting sponge
in the ROV photos was actually botrylloides. Its ~148 labeled images are
archived, not deleted, in `data/labels_yolo_seg_archived_hp/` in case it's
ever worth reintroducing as a class with clearer evidence it's present at
the site.)

## Status as of setup

- **Classes 0 and 1** (Rugulopteryx, Asparagopsis) start with the labeled
  data carried over from the original project: ~190 and ~228 images
  respectively. Their class IDs were kept identical on purpose so the
  existing label files didn't need to be touched at all.
- **Classes 2-3** (Botrylloides, Hildenbrandia) started with **0 labels** -
  extract + label them the same way as the others (see Workflow below).
  Both are now underway.
- A handful of real ROV photos are being labeled directly (via
  `--source rov_frames`) and folded into training alongside the GBIF pool,
  specifically for botrylloides - see "Closing the ROV domain gap" below.
- The raw photo pools for classes 0/1 (`data/images/all`, `data/images/aa`)
  and the SAM checkpoint (`sam_b.pt`) are **symlinked** from the original
  project rather than copied, to save disk space (they're large, read-only,
  and unlikely to change). Labels, weights, and scripts are real copies -
  the two projects are otherwise fully independent from here on.
- This project reuses the original project's Python environment directly
  (`../MRSS/Rugolopteryx okamurae/.venv`) rather than creating a new one -
  the machine is missing `python3-venv` system-wide (`sudo apt install
  python3.10-venv` would fix that if a fully separate env is ever wanted).

## Setup

```
source "../MRSS/Rugolopteryx okamurae/.venv/bin/activate"
```

(Same environment as the original project - already has ultralytics, opencv,
pandas, requests, tqdm, pyyaml, matplotlib, Pillow installed.)

## Workflow

### 1. Get each new species's images out of its GBIF download

```
python scripts/01_extract_gbif_images.py /path/to/botrylloides_violaceus.zip --out-subdir bv
python scripts/01_extract_gbif_images.py /path/to/hildenbrandia_rubra.zip --out-subdir h
```

Unchanged from the original project - unzips the Darwin Core Archive, reads
`multimedia.txt` + `occurrence.txt`, downloads images into
`data/images/<out-subdir>/`. Use `--limit 50` for a quick trial run first.

### 2. Label

```
python scripts/02_label_tool.py --source images/bv --class-id 2
python scripts/02_label_tool.py --source images/h --class-id 3
```

Same SAM-assisted point-and-click tool as before (left-click=positive point,
right-click=negative point, SPACE=predict, s=accept, b=confirmed-background,
q=quit/autosave). **The class ID must match the class list in `data.yaml`** -
double-check the HUD in the label window shows the right class before you
start a session.

To add more Rugulopteryx/Asparagopsis labels, same as the original project:
```
python scripts/02_label_tool.py --source images/all --class-id 0
python scripts/02_label_tool.py --source images/aa --class-id 1
```

### 3. Split into train/val/test

```
python scripts/03_split_dataset.py
```

Rewritten for multi-class: splits **per class independently** (grouped by
GBIF record ID to avoid near-duplicate leakage) rather than globally, so a
class with very little data still gets some val/test representation instead
of a global shuffle potentially leaving it with none. Confirmed-background
images (the `b` key) are pooled across all species and split the same way -
they're useful negatives for every class, not tied to whichever folder they
came from.

Unlike the original project, there's **no "hard negative" species handling**
here - every class is a real, trained class. (If you later want to test
specificity against some *other* confusable organism that's deliberately
never trained, that's a reasonable thing to add back - ask for it.)

### 4. Train

```
python scripts/04_train.py --weights yolo11s-seg.pt --epochs 150 --batch 16
```

Same as before - YOLOv11s-seg from COCO-pretrained weights, boosted HSV
augmentation for the underwater color cast. Since this is now genuinely
multi-class, **ultralytics' own validation confusion matrix**
(`runs/segment/<name>/confusion_matrix.png` and `..._normalized.png`,
generated automatically) becomes your primary tool for seeing species-vs-
species confusion - this is exactly the gap the original project had to
build bespoke tooling to fill for its single "real" class vs. one
suppressed look-alike; with 4 real classes it comes for free.

### Closing the ROV domain gap (botrylloides)

The GBIF training photos are sharp, well-lit, in-air/tidepool macro shots,
but the actual ROV footage is blurry, hazy, color-cast, and much
lower-detail - a real domain gap, most visible on botrylloides, whose
small orange instances in ROV photos went completely undetected by an
early model that only ever saw large, sharp, well-lit colonies. Two things
help close it, used together:

```
python scripts/06_degrade_domain_match.py --class-ids 2
```

Generates blurred/hazy/color-cast/noised duplicates of the botrylloides
training images (added alongside the clean originals, train split only -
run after `03_split_dataset.py`, since it writes into `data/dataset/`
directly). This is a simulation of the domain gap, not the real thing, so
it's a partial fix on its own - it doesn't teach the model what a *small,
distant* instance looks like, only a blurry *same-scale* one.

The stronger fix is labeling a few real ROV photos directly and folding
them into training, which is what `data/rov_frames/` +
`python scripts/02_label_tool.py --source rov_frames --class-id 2` is for
- see the workflow above. Keep some ROV photos out of this entirely if you
want a genuinely unseen set for post-training analysis.

### 5. Scan ROV photos for biofouling

```
python scripts/05_scan_for_biofouling.py --weights runs/segment/train/weights/best.pt --source rov_photos --save
```

Reports, per image: whether biofouling was detected at all, and which
species with what confidence, plus an overall summary (how many of the
scanned images had fouling, broken down by species). This is the tool
built specifically for the "look at the sea wall, tell me if there's
biofouling" use case. `--save` writes annotated copies; large outlier
images are automatically downscaled first to avoid a GPU OOM during mask
rendering (a real issue hit in the original project - one 41-megapixel
photo crashed inference until this was added).

## Layout

```
data/
  images/all/        <- symlinked: Rugulopteryx photos (from original project)
  images/aa/          <- symlinked: Asparagopsis photos (from original project)
  images/bv/          <- Botrylloides photos (add once downloaded)
  images/hp/          <- Hymeniacidon photos (unused - class removed, kept in case it's revisited)
  images/h/           <- Hildenbrandia photos (add once downloaded)
  rov_frames/        <- drop your own ROV/underwater frames here (also used for labeling
                        a handful directly into training - see "Closing the ROV domain gap")
  labels_yolo_seg/   YOLO-seg polygon labels, keyed by image stem (any source dir)
  labels_yolo_seg_archived_hp/  archived hymeniacidon labels, not used by the split script
  dataset/           final train/val/test split in YOLO layout
scripts/
  01_extract_gbif_images.py
  02_label_tool.py
  03_split_dataset.py       (rewritten: multi-class, per-class stratified split)
  04_train.py
  05_scan_for_biofouling.py (new: presence + per-species scan tool)
  06_degrade_domain_match.py (new: ROV-like degraded copies for domain-gap classes)
data.yaml            ultralytics dataset config: 4 classes
runs/                training outputs
sam_b.pt             symlinked from the original project
```
