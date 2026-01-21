"""
vsibench.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the VSI-Bench dataset.
VSI-Bench is a video-based VQA benchmark with both multiple-choice and numeric-answer question types.
"""
import json
import math
import os
import re
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

overwatch = initialize_overwatch(__name__)

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


def _read_jsonl(jsonl_path: Path) -> List[Dict]:
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_pruned_ids(pruned_ids_path: Path) -> set:
    with open(pruned_ids_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-Z]\s*[\.\)]\s*", "", option).strip()


def _select_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return [0]
    if num_frames <= 1:
        return [total_frames // 2]
    if total_frames >= num_frames:
        return [
            int(round(i * (total_frames - 1) / (num_frames - 1)))
            for i in range(num_frames)
        ]
    indices = list(range(total_frames))
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return indices


def _load_video_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    try:
        import decord
    except ImportError as exc:
        raise ImportError(
            "VSI-Bench evaluation requires `decord` for video decoding. "
            "Install it in the eval environment (e.g., `pip install decord`)."
        ) from exc

    decord.bridge.set_bridge("numpy")
    vr = decord.VideoReader(str(video_path))
    indices = _select_frame_indices(len(vr), num_frames)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def _compose_grid(frames: List[Image.Image], grid_cols: Optional[int] = None) -> Image.Image:
    if len(frames) == 1:
        return frames[0]

    base_w, base_h = frames[0].size
    normalized = [
        frame if frame.size == (base_w, base_h) else frame.resize((base_w, base_h))
        for frame in frames
    ]

    cols = grid_cols or int(math.ceil(math.sqrt(len(normalized))))
    rows = int(math.ceil(len(normalized) / cols))

    grid = Image.new("RGB", (cols * base_w, rows * base_h))
    for idx, frame in enumerate(normalized):
        row = idx // cols
        col = idx % cols
        grid.paste(frame, (col * base_w, row * base_h))
    return grid


# === Dataset Indexing / Building Utilities ===
def build_vsibench_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    paths = DATASET_REGISTRY["vsi-bench"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    index_files = [dataset_dir / "metadata-full.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    jsonl_path = root_dir / paths["annotations"]
    pruned_ids = _read_pruned_ids(root_dir / paths["pruned_ids"])
    raw_examples = _read_jsonl(jsonl_path)

    index: Dict[int, Dict] = {}
    for example in tqdm(raw_examples, desc="=> Processing VSI-Bench Dataset:", leave=False):
        qid = int(example["id"])
        dataset_name = example["dataset"]
        scene_name = example["scene_name"]
        video_rel_path = Path("download/vsi-bench") / dataset_name / f"{scene_name}.mp4"

        options = example.get("options")
        if options is not None and isinstance(options, list) and len(options) == 0:
            options = None

        index[qid] = {
            "question_id": qid,
            "question": example["question"],
            "answer": example["ground_truth"],
            "options": options,
            "dataset": dataset_name,
            "scene_name": scene_name,
            "question_type": example["question_type"],
            "video_path": str(video_rel_path),
            "pruned": str(qid) in pruned_ids,
        }

    all_qids = list(index.keys())
    Random(seed).shuffle(all_qids)

    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids}, f)
        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))
            n_slim = min(n_slim, len(all_qids))
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids[:n_slim]}, f)
        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ==
class VsiBenchIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, str]:
        ex = self.examples[idx]
        return ex["question_id"], ex["question"], Path(self.root_dir / ex["video_path"]), ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === Map Dataset Declarations ===
class VsiBenchMapDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        prompt_fn: Callable[..., str],
        image_processor: ImageProcessor = None,
        num_frames: int = 4,
        grid_cols: Optional[int] = None,
    ) -> None:
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file
        self.num_frames, self.grid_cols = num_frames, grid_cols

        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        options = ex.get("options")
        if options:
            choices = [_strip_option_prefix(opt) for opt in options]
            question_prompt = self.prompt_fn(ex["question"], choices=choices)
        else:
            question_prompt = self.prompt_fn(ex["question"])

        video_path = self.root_dir / ex["video_path"]
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing VSI-Bench video at {video_path}. "
                "Download and extract the VSI-Bench videos before evaluation."
            )

        frames = _load_video_frames(video_path, self.num_frames)
        grid_image = _compose_grid(frames, self.grid_cols)

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            pixel_values = self.image_processor(grid_image)
        else:
            pixel_values = self.image_processor(grid_image, return_tensors="pt")["pixel_values"][0]

        return (
            ex["question_id"],
            question_prompt,
            pixel_values,
            ex["question"],
            ex["answer"],
            ex["question_type"],
            ex.get("pruned", False),
        )

    def __len__(self) -> int:
        return len(self.examples)


# === VSI-Bench Task Runner ===
class VsiBenchTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[..., str],
        image_processor: ImageProcessor,
    ) -> None:
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        from accelerate import PartialState

        self.distributed_state = PartialState()
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"VSI-Bench Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            exit(0)

        num_frames = int(os.getenv("VSI_NUM_FRAMES", "4"))
        grid_cols_env = os.getenv("VSI_GRID_COLS", "").strip()
        grid_cols = int(grid_cols_env) if grid_cols_env else None
        overwatch.info(
            f"VSI-Bench video sampling: num_frames={num_frames}, grid_cols={grid_cols or 'auto'}",
            ctx_level=1,
        )

        overwatch.info(f"Assembling VSI-Bench Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = VsiBenchMapDataset(
            self.root_dir,
            self.index_file,
            self.prompt_fn,
            self.image_processor,
            num_frames=num_frames,
            grid_cols=grid_cols,
        )

    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.distributed_state.num_processes,
            rank=self.distributed_state.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(self.dataset, batch_size=device_batch_size, sampler=sampler, num_workers=num_workers)

        result_qa_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for (
                question_ids,
                question_prompts,
                pixel_values,
                questions,
                answers,
                question_types,
                pruned_flags,
            ) in tqdm(dataloader, desc="=>> Evaluating", disable=not self.distributed_state.is_main_process):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(self.distributed_state.device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.to(self.distributed_state.device) for k, v in pixel_values.items()}
                else:
                    raise ValueError(f"Unexpected `pixel_values` type = {type(pixel_values)}")

                gen_answers = vlm.generate_answer(pixel_values, question_prompts)
                for question_id, gen_answer, question, answer, q_type, pruned in zip(
                    question_ids,
                    gen_answers,
                    questions,
                    answers,
                    question_types,
                    pruned_flags,
                    strict=True,
                ):
                    qid = int(question_id.item()) if hasattr(question_id, "item") else int(question_id)
                    pruned_value = bool(pruned.item()) if hasattr(pruned, "item") else bool(pruned)
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "model_output": gen_answer,
                        "ground_truth_answer": answer,
                        "question_type": q_type,
                        "pruned": pruned_value,
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


def _fuzzy_matching(pred: str) -> str:
    return pred.split(" ")[0].rstrip(".").strip()


def _exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.lower() == target.lower() else 0.0


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _mean_relative_accuracy(pred: Optional[float], target: Optional[float], start: float, end: float, interval: float) -> float:
    if pred is None or target is None:
        return 0.0
    if target == 0:
        return 0.0
    num_pts = int((end - start) / interval + 2)
    step = (end - start) / (num_pts - 1) if num_pts > 1 else 0
    conf_intervs = [start + i * step for i in range(num_pts)]
    abs_dist_norm = abs(pred - target) / target
    accuracy = [1.0 if abs_dist_norm <= 1 - conf else 0.0 for conf in conf_intervs]
    return sum(accuracy) / len(accuracy)


def _aggregate_records(records: List[Dict]) -> Dict[str, object]:
    per_type: Dict[str, List[float]] = {}
    for record in records:
        question_type = record["question_type"]
        pred = _fuzzy_matching(record["model_output"])
        target = record["ground_truth_answer"]

        if question_type in MCA_QUESTION_TYPES:
            score = _exact_match(pred, target)
            per_type.setdefault(question_type, []).append(score)
        elif question_type in NA_QUESTION_TYPES:
            score = _mean_relative_accuracy(
                _to_float(pred),
                _to_float(target),
                start=0.5,
                end=0.95,
                interval=0.05,
            )
            per_type.setdefault(question_type, []).append(score)
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    output: Dict[str, float] = {}
    for question_type, scores in per_type.items():
        if question_type in MCA_QUESTION_TYPES:
            output[f"{question_type}_accuracy"] = sum(scores) / len(scores)
        else:
            output[f"{question_type}_MRA:.5:.95:.05"] = sum(scores) / len(scores)

    if (
        "object_rel_direction_easy_accuracy" in output
        and "object_rel_direction_medium_accuracy" in output
        and "object_rel_direction_hard_accuracy" in output
    ):
        output["object_rel_direction_accuracy"] = (
            output.pop("object_rel_direction_easy_accuracy")
            + output.pop("object_rel_direction_medium_accuracy")
            + output.pop("object_rel_direction_hard_accuracy")
        ) / 3.0

    overall = sum(output.values()) / len(output) if output else 0.0
    output["overall"] = overall

    ordered = {
        "overall": overall * 100.0,
    }
    for question_type in [
        "object_counting",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation",
        "object_rel_distance",
        "object_rel_direction",
        "route_planning",
        "obj_appearance_order",
    ]:
        for metric in ["accuracy", "MRA:.5:.95:.05"]:
            key = f"{question_type}_{metric}"
            if key in output:
                ordered[key] = output[key] * 100.0

    tabulated_keys = ", ".join(ordered.keys())
    tabulated_results = ", ".join([f"{value:.3f}" for value in ordered.values()])
    ordered["tabulated_keys"] = tabulated_keys
    ordered["tabulated_results"] = tabulated_results
    return ordered


# === VSI-Bench Scorer ===
class VsiBenchScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "test",
    ) -> None:
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_qa_pairs = full_result_qa_pairs
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[int, Dict]:
        metadata_path = self.annotations_file
        if not metadata_path.exists():
            return {}

        if metadata_path.suffix == ".jsonl":
            examples = _read_jsonl(metadata_path)
        else:
            with open(metadata_path, "r") as f:
                payload = json.load(f)
            examples = list(payload.values()) if isinstance(payload, dict) else payload

        return {int(ex["question_id"]): ex for ex in examples}

    def _hydrate_records(self) -> List[Dict]:
        records = []
        for key, value in self.full_result_qa_pairs.items():
            qid = int(value.get("question_id", key))
            meta = self.metadata.get(qid, {})
            record = {
                "question_id": qid,
                "model_output": value.get("model_output", ""),
                "ground_truth_answer": value.get("ground_truth_answer")
                or value.get("answer")
                or meta.get("answer")
                or meta.get("ground_truth"),
                "question_type": value.get("question_type") or meta.get("question_type"),
                "pruned": value.get("pruned", meta.get("pruned", False)),
            }
            records.append(record)
        return records

    def score(self, model_id: str) -> Dict[str, object]:
        records = self._hydrate_records()
        full_records = records
        debiased_records = [rec for rec in records if not rec["pruned"]]
        pruned_records = [rec for rec in records if rec["pruned"]]

        summary: Dict[str, object] = {
            "n_full": len(full_records),
            "n_debiased": len(debiased_records),
            "n_pruned": len(pruned_records),
        }

        subset_metrics = {
            "full": _aggregate_records(full_records),
            "debiased": _aggregate_records(debiased_records),
            "pruned": _aggregate_records(pruned_records),
        }

        summary["overall"] = subset_metrics["full"].get("overall")
        for prefix, metrics in subset_metrics.items():
            for key, value in metrics.items():
                summary[f"{prefix}__{key}"] = value

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Overall (Full): {subset_metrics['full'].get('overall', 0.0):.3f}\n"
            f"          => Overall (Debiased): {subset_metrics['debiased'].get('overall', 0.0):.3f}\n"
            f"          => Overall (Pruned): {subset_metrics['pruned'].get('overall', 0.0):.3f}"
        )

        return summary
