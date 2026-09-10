"""
flickr_entities.py

Task runner + scorer for Flickr30k Entities (Plummer et al., ICCV 2015) phrase grounding.

WHY THIS BENCHMARK. Every grounding number we have (RefCOCO/+/g, and OCID-Ref's tabletop scenes) is
COCO imagery, which the LLaVA-1.5 instruction mixture also draws on. Flickr30k is genuinely non-COCO,
so it is the control that says whether grounding ability transfers or whether we are reading
familiarity with COCO.

SHAPE. One (image, phrase) pair per example; the model emits a single box; Acc @ 0.5 IoU. That is the
same `refer` route as RefCOCO -- same prompt builder, same parser, same geometry -- so it needs no new
trained format and no new scoring machinery.

THE MULTI-BOX DECISION, MADE EXPLICIT. A phrase's coreference chain can carry several boxes ("two
men" -> one box per man). The model is asked for one box, so a convention is required. We report BOTH,
from the same predictions, and neither is free:

  * `accuracy__Flickr-merged`  -- IoU against the UNION of the chain's boxes. This is what MDETR and
    KOSMOS-2 report, so it is the comparable number, and it is the headline here. It is also
    unforgiving for scattered chains, where the union covers mostly background.
  * `accuracy__Flickr-any`     -- correct if IoU >= 0.5 with any INDIVIDUAL box of the chain OR with
    their union. Including the union is what makes this a strict superset of the merged protocol, so
    `any >= merged` always holds and the GAP between them is the quantity of interest: how often the
    model grounds a single instance instead of the whole group. (Individual boxes alone are not a
    superset -- a prediction covering a scattered group matches the union and no single box.)
    More generous than merged, and not comparable to published numbers.

Single-box phrases (the large majority) score identically under both, so the gap between the two is a
direct read-out of how much the multi-box convention is doing.

EXCLUSIONS, not failures. Phrases whose chain id is `0` (`notvisual`) and chains flagged
`scene`/`nobndbox` have no box to predict; the stager drops them rather than scoring them as misses.

COORDINATE FRAME. Ground truth goes through `normalize_bbox_for_transform`, which projects into the
centered pad-to-square frame the model actually emits -- see
docs/reference/GROUNDING_COORDINATE_FRAME.md. Do not hand it raw-image-normalized boxes.
"""
import json
import os
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from accelerate import PartialState
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.harnesses.geometry import normalize_bbox_for_transform
from vlm_eval.tasks.harnesses.refcoco import compute_iou, parse_bbox
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

overwatch = initialize_overwatch(__name__)


def _union(boxes: List[List[float]]) -> List[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


# === Index Builder ===
def build_flickr_entities_indices(
    root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]] = None, seed: int = 21
) -> List[Path]:
    """Build BOTH splits. `BUILDER_DISPATCH` calls builders as (root_dir, slim_sizes, seed) with no
    split argument, and the family registers one dataset id per split, so one call has to cover both."""
    out: List[Path] = []
    for split in DATASET_REGISTRY["flickr-entities"]["splits"]:
        out.extend(_build_split_indices(root_dir, slim_dataset_sizes, seed, split))
    return out


def _build_split_indices(
    root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int, split: str
) -> List[Path]:
    paths = DATASET_REGISTRY["flickr-entities"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    index_files = [dataset_dir / f"metadata-full-{split}.json"] + (
        [dataset_dir / f"metadata-slim-{k}-{split}.json" for k in slim_dataset_sizes] if slim_dataset_sizes else []
    )
    if all(p.exists() for p in index_files):
        return index_files

    examples_file = root_dir / paths["download_dir"] / f"examples-{split}.jsonl"
    if not examples_file.exists():
        raise FileNotFoundError(
            f"Flickr30k Entities source not found at {examples_file}. Stage it first with\n"
            f"    sbatch exp_scripts/evaluation/stage_flickr_entities.slurm"
        )

    images_rel = Path(paths["images"])
    index: Dict[int, Dict] = {}
    with open(examples_file, "r") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            boxes = [[float(c) for c in b] for b in row["boxes_xyxy"]]
            index[i] = {
                "example_id": i,                       # positional: stable, and NOT Python hash()
                "ref_expression": row["phrase"],
                "img_path": str(images_rel / f"{row['image_id']}.jpg"),
                "image_id": row["image_id"],
                "chain_id": row["chain_id"],
                # `bbox_xyxy` is the MERGED box -- the field the shared refer-scorer geometry expects.
                "bbox_xyxy": _union(boxes),
                "all_boxes_xyxy": boxes,
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
                "split": split,
            }

    with open(index_files[0], "w") as f:
        json.dump(index, f)

    if slim_dataset_sizes is not None:
        all_ids = sorted(index.keys())
        Random(seed).shuffle(all_ids)
        for k, slim_path in zip(slim_dataset_sizes, index_files[1:], strict=True):
            with open(slim_path, "w") as f:
                json.dump({eid: index[eid] for eid in all_ids[:k]}, f)

    return index_files


class FlickrEntitiesIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, List[float]]:
        ex = self.examples[idx]
        return ex["example_id"], ex["ref_expression"], Path(self.root_dir / ex["img_path"]), ex["bbox_xyxy"]

    def __len__(self) -> int:
        return len(self.examples)


class FlickrEntitiesMapDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = self.prompt_fn(ex["ref_expression"])

        from torchvision.transforms import Compose

        image = Image.open(self.root_dir / ex["img_path"]).convert("RGB")
        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            pixel_values = self.image_processor(image)
        else:
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        return ex["example_id"], prompt, pixel_values, ex["ref_expression"]

    def __len__(self) -> int:
        return len(self.examples)


def _flickr_collate(batch):
    example_ids, prompts, pixel_values, phrases = zip(*batch)
    if isinstance(pixel_values[0], torch.Tensor):
        pixel_values = torch.stack(pixel_values, dim=0)
    elif isinstance(pixel_values[0], dict):
        pixel_values = {k: torch.stack([pv[k] for pv in pixel_values], dim=0) for k in pixel_values[0]}
    else:
        raise TypeError(f"Unsupported Flickr30k pixel_values type: {type(pixel_values[0])}")
    return torch.tensor(example_ids, dtype=torch.long), list(prompts), pixel_values, list(phrases)


class FlickrEntitiesTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        self.task_results_dir, self.model_id = task_results_dir, model_id
        self.prompt_fn, self.image_processor = prompt_fn, image_processor

        self.distributed_state = PartialState()
        os.makedirs(self.task_results_dir, exist_ok=True)
        self.dataset = FlickrEntitiesMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.distributed_state.num_processes,
            rank=self.distributed_state.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(
            self.dataset,
            batch_size=device_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=_flickr_collate,
        )

        result_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for example_ids, prompts, pixel_values, phrases in tqdm(
                dataloader, desc="=>> Evaluating", disable=not self.distributed_state.is_main_process
            ):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(self.distributed_state.device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.to(self.distributed_state.device) for k, v in pixel_values.items()}
                else:
                    raise ValueError(f"Unexpected `pixel_values` type = {type(pixel_values)}")

                gen_texts = vlm.generate_answer(pixel_values, prompts)
                for eid, gen, phrase in zip(example_ids, gen_texts, phrases, strict=True):
                    eid = int(eid.item())
                    result_pairs[eid] = {"example_id": eid, "ref_exp": phrase, "model_output": gen}
        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_pairs, f, indent=2)

        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


class FlickrEntitiesScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_sent_bbox_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "test",
        **_: str,
    ) -> None:
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_sent_bbox_pairs = full_result_sent_bbox_pairs
        self.transform_info = self._load_transform_info()
        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)

    def _load_transform_info(self) -> Dict[str, str]:
        transform_file = self.task_results_dir / "image_transform.json"
        if transform_file.exists():
            with open(transform_file, "r") as f:
                return json.load(f)
        raise FileNotFoundError(
            f"Missing `{transform_file}`. Re-run evaluation so preprocessing metadata can be recorded."
        )

    def score(self, model_id: str) -> Dict[str, float]:
        n_total = n_merged = n_any = n_invalid = n_multi = 0

        for example_id, example in tqdm(self.full_result_sent_bbox_pairs.items(), "=> Scoring Box Predictions:"):
            ann = self.annotations[str(example_id)]
            image_wh = (int(ann["image_width"]), int(ann["image_height"]))
            n_total += 1

            pred = parse_bbox(example["model_output"])
            if pred is None:
                n_invalid += 1
                continue

            merged_gt = normalize_bbox_for_transform(ann["bbox_xyxy"], image_wh, self.transform_info)
            example["ground_truth_bbox"] = merged_gt
            if compute_iou(pred, merged_gt) >= 0.5:
                n_merged += 1

            # The lenient protocol must be a SUPERSET of the merged one, or the pair cannot be read as
            # "how much does the model lose by grounding one instance instead of the group".
            # Testing only the individual boxes is NOT a superset: a prediction that correctly covers a
            # scattered group matches the union but has low IoU with every single box, so it scored
            # merged=hit / any=miss. Measured on one run: 359 such cases, ALL multi-box. So the
            # candidate set is the individual boxes PLUS the union.
            boxes = ann.get("all_boxes_xyxy") or [ann["bbox_xyxy"]]
            if len(boxes) > 1:
                n_multi += 1
            candidates = list(boxes) + [ann["bbox_xyxy"]]
            if any(
                compute_iou(pred, normalize_bbox_for_transform(b, image_wh, self.transform_info)) >= 0.5
                for b in candidates
            ):
                n_any += 1

        accuracies = {
            "accuracy__Flickr-merged": n_merged / n_total if n_total else 0.0,
            "accuracy__Flickr-any": n_any / n_total if n_total else 0.0,
            "invalid_fraction__Flickr": n_invalid / n_total if n_total else 0.0,
            "multibox_fraction__Flickr": n_multi / n_total if n_total else 0.0,
        }
        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Flickr30k Acc@0.5 (merged box, comparable): {accuracies['accuracy__Flickr-merged']:.3f}\n"
            f"          => Flickr30k Acc@0.5 (any box):                {accuracies['accuracy__Flickr-any']:.3f}\n"
            f"          => unparseable model outputs:                  {accuracies['invalid_fraction__Flickr']:.3f}\n"
            f"          => multi-box phrases:                          {accuracies['multibox_fraction__Flickr']:.3f}"
        )
        return accuracies
