"""
cvbench.py

Task runner + scorer for CV-Bench (Cambrian-1, NeurIPS 2024; HF `nyu-visionx/CV-Bench`).

2,638 multiple-choice items over four vision-centric sub-tasks:

    2D  Count     788   |  2D  Relation  650      (images: ADE20K val 633 + COCO val2017 805)
    3D  Depth     600   |  3D  Distance  600      (images: Omni3D -- non-COCO)

Scored the same way as every other closed-set task in this harness: the first generated token's
softmax restricted to the single-token option letters, renormalised, argmax. No instruction-following
and no LLM grader required.

TWO THINGS THAT MAKE THIS DIFFERENT FROM `ai2d.py`, AND WHY:

1. **Variable option count.** CV-Bench items have 2, 4, 5 or 6 choices (measured: 1850/493/156/139).
   `ai2d.py` hard-codes `range(4)`, which would silently truncate the 5- and 6-option items and score
   them against the wrong candidate set. We group each batch by `len(choices)` and score each group
   with its own letter set -- the `vsi.py` pattern.

2. **The official metric is not plain accuracy.** CV-Bench weights its two 2D image sources equally
   and then weights 2D against 3D equally:

       CV-Bench = ( (acc_ADE20K + acc_COCO) / 2 + acc_Omni3D ) / 2

   Plain micro-accuracy over all 2,638 items is reported too (`accuracy__CVBench-micro`) but it is NOT
   the number the paper quotes.

CAVEAT worth carrying into any writeup: ~30% of CV-Bench imagery is COCO, which the LLaVA-1.5
instruction mixture also draws on. It is a measurement instrument, never a contamination argument.
"""
import json
import os
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import PartialState
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

overwatch = initialize_overwatch(__name__)

# 2D and 3D ship as separate parquet shards, each with its own `idx` starting at 0. Namespace them so
# example ids are globally unique WITHOUT using Python's `hash()` -- that is PYTHONHASHSEED-dependent
# and silently invalidates every stored result when the seed changes (see the note on pope.py).
TYPE_OFFSET = {"2D": 0, "3D": 1_000_000}

# The official average groups sources, not tasks.
SOURCE_2D = ("ADE20K", "COCO")
SOURCE_3D = ("Omni3D",)


def _answer_to_index(answer: str, n_choices: int) -> int:
    """CV-Bench stores the gold answer as a parenthesised letter, e.g. "(C)"."""
    letter = answer.strip().strip("()").strip()
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError(f"Unparseable CV-Bench answer {answer!r}")
    idx = ord(letter.upper()) - ord("A")
    if not 0 <= idx < n_choices:
        raise ValueError(f"CV-Bench answer {answer!r} out of range for {n_choices} choices")
    return idx


# === Index Builder ===
def build_cvbench_indices(
    root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]] = None, seed: int = 21
) -> List[Path]:
    """Materialise `metadata-full.json` (and any slim variants) from the staged `examples.jsonl`."""
    paths = DATASET_REGISTRY["cv-bench"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    index_files = [dataset_dir / "metadata-full.json"] + (
        [dataset_dir / f"metadata-slim-{k}.json" for k in slim_dataset_sizes] if slim_dataset_sizes else []
    )
    if all(p.exists() for p in index_files):
        return index_files

    examples_file = root_dir / paths["examples"]
    if not examples_file.exists():
        raise FileNotFoundError(
            f"CV-Bench source not found at {examples_file}. Stage it first with\n"
            f"    sbatch exp_scripts/evaluation/stage_cvbench.slurm"
        )

    index: Dict[int, Dict] = {}
    with open(examples_file, "r") as f:
        for line in f:
            row = json.loads(line)
            choices = list(row["choices"])
            example_id = TYPE_OFFSET[row["type"]] + int(row["idx"])
            if example_id in index:
                raise ValueError(f"duplicate CV-Bench example_id {example_id}")
            index[example_id] = {
                "example_id": example_id,
                "question": row["question"],
                "answer_choices": choices,
                "answer_label": _answer_to_index(row["answer"], len(choices)),
                "img_path": str(Path(paths["download_dir"]) / row["filename"]),
                "task": row["task"],          # Count | Relation | Depth | Distance
                "cv_type": row["type"],       # 2D | 3D
                "source": row["source"],      # ADE20K | COCO | Omni3D
            }

    n_expected = DATASET_REGISTRY["cv-bench"]["expected_examples"]
    assert len(index) == n_expected, f"CV-Bench: expected {n_expected} examples, built {len(index)}"

    with open(index_files[0], "w") as f:
        json.dump(index, f)

    # Slim variants nest: shuffle once, then take prefixes.
    if slim_dataset_sizes is not None:
        all_ids = sorted(index.keys())
        Random(seed).shuffle(all_ids)
        for k, slim_path in zip(slim_dataset_sizes, index_files[1:], strict=True):
            with open(slim_path, "w") as f:
                json.dump({eid: index[eid] for eid in all_ids[:k]}, f)

    return index_files


# === Index (metadata-only) Dataset ===
class CVBenchIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, List[str], int]:
        ex = self.examples[idx]
        return (
            ex["example_id"],
            ex["question"],
            Path(self.root_dir / ex["img_path"]),
            ex["answer_choices"],
            ex["answer_label"],
        )

    def __len__(self) -> int:
        return len(self.examples)


# === Map Dataset (materialises pixel values) ===
class CVBenchMapDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        prompt_fn: Callable[[str, List[str]], str],
        image_processor: ImageProcessor,
    ) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        # Re-render the options through the harness's own MC prompt builder ("A. ...") rather than
        # using CV-Bench's pre-rendered `prompt` field ("(A) ..."): the "A." form plus the trailing
        # "Answer with the option's letter..." string is byte-identical to the A-OKVQA format the
        # model was instruction-tuned on.
        question_prompt = self.prompt_fn(ex["question"], ex["answer_choices"])

        from PIL import Image
        from torchvision.transforms import Compose

        image = Image.open(self.root_dir / ex["img_path"]).convert("RGB")
        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            pixel_values = self.image_processor(image)
        else:
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        return (
            ex["example_id"],
            question_prompt,
            pixel_values,
            ex["question"],
            ex["answer_choices"],
            ex["answer_label"],
        )

    def __len__(self) -> int:
        return len(self.examples)


def _cvbench_collate(batch):
    """Custom collate: `answer_choices` are variable-length, so the default collate cannot stack them."""
    example_ids, prompts, pixel_values, questions, choices, labels = zip(*batch)
    if isinstance(pixel_values[0], torch.Tensor):
        pixel_values = torch.stack(pixel_values, dim=0)
    elif isinstance(pixel_values[0], dict):
        pixel_values = {k: torch.stack([pv[k] for pv in pixel_values], dim=0) for k in pixel_values[0]}
    else:
        raise TypeError(f"Unsupported CV-Bench pixel_values type: {type(pixel_values[0])}")
    return (
        torch.tensor(example_ids, dtype=torch.long),
        list(prompts),
        pixel_values,
        list(questions),
        [list(c) for c in choices],
        torch.tensor(labels, dtype=torch.long),
    )


def _slice_pixel_values(pixel_values, indices: List[int]):
    if not indices:
        return pixel_values
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.index_select(0, torch.tensor(indices, device=pixel_values.device))
    device = next(iter(pixel_values.values())).device
    index_tensor = torch.tensor(indices, device=device)
    return {k: v.index_select(0, index_tensor) for k, v in pixel_values.items()}


# === Task Runner ===
class CVBenchTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str, List[str]], str],
        image_processor: ImageProcessor,
    ) -> None:
        self.root_dir, self.index_file = root_dir, index_file
        self.task_results_dir, self.model_id = task_results_dir, model_id
        self.prompt_fn, self.image_processor = prompt_fn, image_processor

        self.distributed_state = PartialState()
        os.makedirs(self.task_results_dir, exist_ok=True)
        self.dataset = CVBenchMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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
            collate_fn=_cvbench_collate,
        )

        result_qa_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for example_ids, prompts, pixel_values, questions, choices, labels in tqdm(
                dataloader, desc="=>> Evaluating", disable=not self.distributed_state.is_main_process
            ):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(self.distributed_state.device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.to(self.distributed_state.device) for k, v in pixel_values.items()}
                else:
                    raise ValueError(f"Unexpected `pixel_values` type = {type(pixel_values)}")

                # Group by option count -- CV-Bench mixes 2/4/5/6-way items in one batch, and the
                # candidate letter set has to match each item's own option count.
                for n_choices in sorted({len(c) for c in choices}):
                    group = [i for i, c in enumerate(choices) if len(c) == n_choices]
                    return_strings = [chr(ord("A") + i) for i in range(n_choices)]
                    gen_probabilities = vlm.generate_answer(
                        _slice_pixel_values(pixel_values, group),
                        [prompts[i] for i in group],
                        return_string_probabilities=return_strings,
                    )
                    for i, probs in zip(group, gen_probabilities, strict=True):
                        eid = int(example_ids[i].item())
                        result_qa_pairs[eid] = {
                            "example_id": eid,
                            "question": questions[i],
                            "choices": choices[i],
                            "mc_probabilities": probs,
                            "ground_truth_answer": int(labels[i].item()),
                        }
        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Scorer ===
class CVBenchScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "test",
        **_: str,
    ) -> None:
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_qa_pairs = full_result_qa_pairs
        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)

    def score(self, model_id: str) -> Dict[str, float]:
        correct_task, total_task = {}, {}
        correct_source, total_source = {}, {}
        n_correct = n_total = 0

        for example in self.full_result_qa_pairs.values():
            ann = self.annotations[str(example["example_id"])]
            task, source = ann["task"], ann["source"]
            hit = int(np.argmax(example["mc_probabilities"])) == example["ground_truth_answer"]

            correct_task[task] = correct_task.get(task, 0) + int(hit)
            total_task[task] = total_task.get(task, 0) + 1
            correct_source[source] = correct_source.get(source, 0) + int(hit)
            total_source[source] = total_source.get(source, 0) + 1
            n_correct += int(hit)
            n_total += 1

        accuracies = {f"accuracy__CVBench-{t}": correct_task[t] / total_task[t] for t in sorted(total_task)}
        accuracies.update(
            {f"accuracy__CVBench-src-{s}": correct_source[s] / total_source[s] for s in sorted(total_source)}
        )
        accuracies["accuracy__CVBench-micro"] = n_correct / n_total if n_total else 0.0

        # The Cambrian-1 reporting scheme, reproduced exactly so our numbers are comparable to their
        # Table 1. It is UNWEIGHTED at both levels:
        #
        #   CV-Bench_2D  = (acc_ADE20K + acc_COCO) / 2     <- source-level, NOT item-weighted, so
        #                                                     ADE20K (633) counts as much as COCO (805)
        #   CV-Bench_3D  = acc_Omni3D                       <- all 1,200 3D items are Omni3D; equal
        #                                                     Depth/Distance counts (600/600) make the
        #                                                     per-task mean identical, so no ambiguity
        #   CV-Bench     = (CV-Bench_2D + CV-Bench_3D) / 2 <- so 1,438 2D items count as much as 1,200 3D
        #
        # Plain micro-accuracy over all 2,638 items is NOT this number; it is reported separately as
        # `accuracy__CVBench-micro` and should not be compared against published CV-Bench figures.
        acc_2d = [correct_source[s] / total_source[s] for s in SOURCE_2D if total_source.get(s)]
        acc_3d = [correct_source[s] / total_source[s] for s in SOURCE_3D if total_source.get(s)]
        if acc_2d:
            accuracies["accuracy__CVBench-2D"] = sum(acc_2d) / len(acc_2d)
        if acc_3d:
            accuracies["accuracy__CVBench-3D"] = sum(acc_3d) / len(acc_3d)
        if acc_2d and acc_3d:
            accuracies["accuracy__CVBench-final"] = (sum(acc_2d) / len(acc_2d) + sum(acc_3d) / len(acc_3d)) / 2

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id}\n"
            + "\n".join(f"          => {k.replace('accuracy__', '')}: {v:.3f}" for k, v in accuracies.items())
        )
        return accuracies
