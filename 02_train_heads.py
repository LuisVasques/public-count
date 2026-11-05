import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
from ultralytics import YOLO

URL = "https://huggingface.co/datasets/ShantyCam/crowdhuman/resolve/main/"
DATASET_DIR = Path("/home/luis/projetos/datasets/crowdhuman-heads")


def download(name, destination):
    archive = destination / name
    if archive.exists():
        return archive
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {name}...")
    temp_file = archive.with_suffix(archive.suffix + ".partial")
    urlretrieve(URL + name, temp_file)
    temp_file.rename(archive)
    return archive


def extract(archive, destination):
    marker = destination / f".{archive.stem}.ok"
    if marker.exists():
        return
    print(f"Extraindo {archive.name}...")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.filename.lower().endswith(".jpg"):
                with z.open(member) as source, open(destination / Path(member.filename).name, "wb") as output:
                    shutil.copyfileobj(source, output)
    marker.touch()


def convert_annotations(annotations, images, split, limit=None, seed=0):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    lines = [json.loads(l) for l in annotations.read_text().splitlines()]
    lines = [l for l in lines if (images / f"{l['ID']}.jpg").exists()]
    random.Random(seed).shuffle(lines)
    if limit:
        lines = lines[:limit]

    total = 0
    for item in lines:
        source = images / f"{item['ID']}.jpg"
        height, width = cv2.imread(str(source)).shape[:2]
        labels = []
        for box in item["gtboxes"]:
            if box.get("tag") != "person" or "hbox" not in box:
                continue
            if box.get("head_attr", {}).get("ignore", 0) or box.get("extra", {}).get("ignore", 0):
                continue
            x, y, w, h = box["hbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            labels.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        destination = image_dir / source.name
        if not destination.exists():
            destination.symlink_to(source)
        (label_dir / f"{source.stem}.txt").write_text("\n".join(labels))
        total += len(labels)
    print(f"{split}: {len(lines)} imagens, {total} cabeças")


def prepare(parts, val_limit):
    raw_dir = DATASET_DIR / "raw"
    download("annotation_train.odgt", raw_dir)
    download("annotation_val.odgt", raw_dir)
    for part in parts:
        extract(download(f"CrowdHuman_{part}.zip", raw_dir), raw_dir / "train")
    extract(download("CrowdHuman_val.zip", raw_dir), raw_dir / "val")

    convert_annotations(raw_dir / "annotation_train.odgt", raw_dir / "train", "train")
    convert_annotations(raw_dir / "annotation_val.odgt", raw_dir / "val", "val", val_limit)

    yaml = DATASET_DIR / "heads.yaml"
    yaml.write_text(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: head\n")
    return yaml


def main():
    parser = argparse.ArgumentParser(description="Treina um detector de cabeças YOLO no CrowdHuman")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--parts", nargs="+", default=["train01"], choices=["train01", "train02", "train03"])
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    yaml = prepare(args.parts, args.val_limit)
    if args.prepare_only:
        return

    model = YOLO(args.model)
    model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        max_det=1000,
        mosaic=1.0,
        scale=0.5,
        close_mosaic=5,
        patience=15,
        workers=8,
        project=str(Path("runs").resolve()),
        name="heads",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
