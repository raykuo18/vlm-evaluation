#!/usr/bin/env python
"""Stage Flickr30k Entities (Plummer et al., ICCV 2015) into the local evaluation store.

Two sources, because they live in different places:

  * **Annotations** -- `annotations.zip` (29 MB) from github.com/BryanPlummer/flickr30k_entities.
    PASCAL-VOC-style `Annotations/<image_id>.xml` (boxes, keyed by coreference chain id) plus
    `Sentences/<image_id>.txt` (captions with `[/EN#<chain>/<type> words]` bracketing).
  * **Images** -- the canonical route is a form at Illinois (`forms.illinois.edu/sec/229675`), which
    cannot be automated. The HF mirror `nlphuji/flickr30k` is ungated and serves the identical
    `flickr30k-images/<image_id>.jpg` names the annotations key on, so we use that.

    python scripts/download_flickr_entities_from_hf.py --root_dir /path/to/vlm_eval_data

Run on a COMPUTE node (login1 SIGKILLs sustained downloads; the image zip is 4.4 GB).

SCORING CONVENTIONS (decided here, consumed by vlm_eval/tasks/harnesses/flickr_entities.py):

  * A phrase whose coreference chain carries SEVERAL boxes is stored with all of them. The harness
    reports both the standard merged-box protocol (union of the chain's boxes -- what MDETR and
    KOSMOS-2 report) and an any-box variant, from the same predictions.
  * Phrases with chain id `0` (`notvisual`) are EXCLUDED, not scored as misses -- there is no box to
    predict. Same for chains flagged `scene` or `nobndbox`.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download, list_repo_files
from torch.hub import download_url_to_file

ANNOTATIONS_URL = "https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip"
# The split lists are loose files in the repo ROOT, not members of annotations.zip.
SPLIT_URL = "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/{split}.txt"
IMAGES_REPO = "nlphuji/flickr30k"
PHRASE_RE = re.compile(r"\[/EN#(\d+)(?:/[^\s\]]+)*\s+([^\]]+)\]")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_dir", type=Path,
                    default=Path("/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data"))
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--skip_images", action="store_true", help="annotations only (for a metadata-only rebuild)")
    return ap.parse_args()


def fetch_annotations(out_dir: Path) -> Path:
    ann_dir = out_dir / "annotations"
    if (ann_dir / "Annotations").is_dir() and (ann_dir / "Sentences").is_dir():
        print(f"[flickr] annotations already at {ann_dir}")
        return ann_dir
    ann_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "annotations.zip"
    if not zip_path.exists():
        print(f"[flickr] downloading {ANNOTATIONS_URL}")
        download_url_to_file(ANNOTATIONS_URL, str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ann_dir)
    print(f"[flickr] extracted annotations -> {ann_dir}")
    return ann_dir


def fetch_split_lists(ann_dir: Path, splits: List[str]) -> None:
    for split in splits:
        dest = ann_dir / f"{split}.txt"
        if dest.exists():
            continue
        url = SPLIT_URL.format(split=split)
        print(f"[flickr] downloading {url}")
        download_url_to_file(url, str(dest))


def fetch_images(out_dir: Path) -> Path:
    img_dir = out_dir / "flickr30k-images"
    if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) > 30000:
        print(f"[flickr] images already at {img_dir}")
        return img_dir
    files = list_repo_files(IMAGES_REPO, repo_type="dataset")
    zips = [f for f in files if f.endswith(".zip") and "image" in f.lower()]
    if not zips:
        raise RuntimeError(f"no image archive in {IMAGES_REPO}; saw {files[:20]}")
    print(f"[flickr] downloading {zips[0]} from {IMAGES_REPO} (~4.4 GB)")
    local = hf_hub_download(repo_id=IMAGES_REPO, filename=zips[0], repo_type="dataset")
    with zipfile.ZipFile(local) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg") and not m.startswith("__MACOSX")]
        print(f"[flickr] extracting {len(members)} images -> {out_dir}")
        zf.extractall(out_dir, members=members)
    if not img_dir.is_dir():                      # archive may not carry the top-level folder
        img_dir = out_dir
    return img_dir


def boxes_for_image(xml_path: Path) -> tuple[Dict[str, List[List[float]]], int, int]:
    """chain id -> [ [x1,y1,x2,y2], ... ]; plus image width/height. Skips scene/nobndbox chains."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)
    boxes: Dict[str, List[List[float]]] = defaultdict(list)
    for obj in root.findall("object"):
        if obj.find("bndbox") is None:            # `scene` / `nobndbox` -- nothing to localize
            continue
        bb = obj.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        for name in obj.findall("name"):
            boxes[name.text.strip()].append(box)
    return boxes, width, height


def main() -> int:
    args = parse_args()
    out_dir = args.root_dir / "download" / "flickr-entities"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = fetch_annotations(out_dir)
    fetch_split_lists(ann_dir, args.splits)
    if not args.skip_images:
        fetch_images(out_dir)

    for split in args.splits:
        split_file = ann_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"missing split list {split_file}")
        image_ids = [ln.strip() for ln in split_file.read_text().split() if ln.strip()]

        records, n_notvisual, n_nobox, n_multi = [], 0, 0, 0
        for image_id in image_ids:
            xml_path = ann_dir / "Annotations" / f"{image_id}.xml"
            sent_path = ann_dir / "Sentences" / f"{image_id}.txt"
            if not xml_path.exists() or not sent_path.exists():
                continue
            chain_boxes, width, height = boxes_for_image(xml_path)
            seen = set()
            for line in sent_path.read_text().splitlines():
                for chain_id, phrase in PHRASE_RE.findall(line):
                    phrase = " ".join(phrase.split())
                    if chain_id == "0":                      # notvisual
                        n_notvisual += 1
                        continue
                    if chain_id not in chain_boxes:          # scene / nobndbox
                        n_nobox += 1
                        continue
                    key = (image_id, chain_id, phrase.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    boxes = chain_boxes[chain_id]
                    if len(boxes) > 1:
                        n_multi += 1
                    records.append({
                        "image_id": image_id,
                        "chain_id": chain_id,
                        "phrase": phrase,
                        "boxes_xyxy": boxes,
                        "image_width": width,
                        "image_height": height,
                    })

        dest = out_dir / f"examples-{split}.jsonl"
        with open(dest, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"[flickr] {split}: {len(image_ids)} images -> {len(records)} phrase-box pairs "
              f"({n_multi} multi-box, excluded {n_notvisual} notvisual + {n_nobox} scene/nobndbox)")
        print(f"[flickr] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
