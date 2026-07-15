# Biofouling ROV Detector

YOLOv11 instance segmentation for detecting biofouling organisms on a harbor
sea wall in ROV video, and identifying which organism is present.

## Classes

Trained as four separate classes, not one generic "biofouling" label. This
gives species identity in addition to plain presence/absence (presence is
just "did any class fire").

```
0: rugulopteryx_okamurae   (invasive brown alga)
1: asparagopsis_armata     (invasive red alga)
2: botrylloides_violaceus  (invasive colonial tunicate)
3: hildenbrandia_rubra     (encrusting red alga)
```

## Setup

```
pip install ultralytics opencv-python numpy pandas pyyaml requests tqdm pillow matplotlib
```

## Workflow

### 1. Extract images from a GBIF download

```
python scripts/01_extract_gbif_images.py /path/to/species.zip --out-subdir <name>
```

Unzips a Darwin Core Archive and downloads the images listed in
`multimedia.txt` / `occurrence.txt` into `data/images/<name>/`. Use
`--limit 50` for a quick test run.

### 2. Label

```
python scripts/02_label_tool.py --source images/<name> --class-id <id>
```

SAM-assisted point-and-click labeling. Left-click: positive point.
Right-click: negative point. SPACE: predict mask. s: accept. b: confirmed
background. q: quit and autosave. The class ID must match `data.yaml`.

### 3. Split into train/val/test

```
python scripts/03_split_dataset.py
```

Splits each class independently (grouped by GBIF record ID to avoid
near-duplicate leakage), so low-data classes still get val/test coverage.
Confirmed-background images are pooled across all classes.

### 4. Train

```
python scripts/04_train.py --weights yolo11s-seg.pt --epochs 150 --batch 16
```

Fine-tunes YOLOv11s-seg from COCO-pretrained weights, with boosted HSV
augmentation for the underwater color cast. Check
`runs/segment/<name>/confusion_matrix.png` for species-vs-species confusion.

### 5. Close the ROV domain gap (botrylloides)

GBIF photos are sharp, well-lit macro shots. ROV footage is blurry, hazy,
and color-cast. That gap is most visible on botrylloides, whose small
instances went undetected until this was addressed. Two steps close it:

```
python scripts/06_degrade_domain_match.py --class-ids 2
```

Generates blurred, hazy, color-cast, noised duplicates of the botrylloides
training images (train split only, added alongside the originals). Run
after step 3.

```
python scripts/02_label_tool.py --source rov_frames --class-id 2
```

Labels a handful of real ROV photos directly and folds them into training.
This is the stronger fix: the degraded copies only simulate blur at the
same scale, not a small, distant instance.

### 6. Scan photos for biofouling

```
python scripts/05_scan_for_biofouling.py --weights runs/segment/train-4/weights/best.pt --source rov_photos --save
```

Reports per image whether biofouling was detected, which species, and
confidence, plus an overall summary. `--save` writes annotated copies.
Oversized images are automatically downscaled before mask rendering to
avoid a GPU out-of-memory error.

### 7. Live scan from a BlueOS video stream

```
python scripts/07_live_scan.py --source rtsp://192.168.2.2:8554/video --weights runs/segment/train-4/weights/best.pt
```

Runs the model over a live RTSP stream, meant to run topside on a GPU
machine rather than on the ROV's own companion computer. Shows an
annotated preview window and logs every detection (timestamp, species,
confidence) to a CSV. Use `--source 0` to test against a local webcam
first, and `--no-display` for a headless run. Press q in the preview
window (or Ctrl+C in the terminal) to stop.

Find the stream URL in the BlueOS web UI under Video Manager: set up an
RTSP (or other cv2-readable) output for the camera you want, then pass
that URL as `--source`.

## Layout

```
data/
  images/<name>/     source photos per species, from GBIF
  rov_frames/        real ROV/underwater frames for direct labeling
  labels_yolo_seg/   YOLO-seg polygon labels
  dataset/           final train/val/test split, YOLO layout
scripts/
  01_extract_gbif_images.py
  02_label_tool.py
  03_split_dataset.py
  04_train.py
  05_scan_for_biofouling.py
  06_degrade_domain_match.py
  07_live_scan.py
data.yaml   dataset config, 4 classes
runs/       training outputs
```
