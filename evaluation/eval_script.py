"""
eval_script.py — Ragas evaluation runner.

Loads golden Q&A pairs, runs each through the CRAG pipeline, and evaluates
with Ragas metrics: context_precision, faithfulness, answer_relevancy.
Results are printed and saved to evaluation/ragas_results.json.
"""

import json
import os
import sys
import logging
from pathlib import Path

# Ensure project root is on sys.path for imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
)

from core_logic.graph import run_crag_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_golden_dataset(path: str) -> list[dict]:
    """
    Load the golden evaluation dataset from a JSON file.

    Args:
        path: Path to golden_dataset.json.

    Returns:
        List of dicts with keys: question, ground_truth, contexts.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d golden Q&A pairs from %s", len(data), path)
    return data


def run_evaluation(golden_path: str | None = None) -> dict:
    """
    Run the full Ragas evaluation pipeline.

    1. Load golden dataset
    2. Run each question through the CRAG pipeline
    3. Build Ragas Dataset
    4. Evaluate with context_precision, faithfulness, answer_relevancy
    5. Print and save results

    Args:
        golden_path: Optional path override for the golden dataset JSON.

    Returns:
        Dict of metric name → score.
    """
    if golden_path is None:
        golden_path = os.path.join(
            os.path.dirname(__file__), "golden_dataset.json"
        )

    golden_data = load_golden_dataset(golden_path)

    # ── Run CRAG pipeline for each question ──────────────────────────────
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, item in enumerate(golden_data):
        question = item["question"]
        logger.info(
            "[%d/%d] Processing: '%s'", i + 1, len(golden_data), question[:80]
        )

        try:
            result = run_crag_pipeline(question)
            answer = result.get("answer", "")

            # Extract context texts from retrieved chunks
            retrieved_chunks = result.get("retrieved_chunks", [])
            context_texts = [
                chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")
            ]

            # Fall back to golden contexts if no chunks were retrieved
            if not context_texts:
                context_texts = item.get("contexts", ["No context retrieved."])

        except Exception as exc:
            logger.error("Pipeline failed for question %d: %s", i + 1, exc)
            answer = "Error: pipeline failed"
            context_texts = item.get("contexts", ["Error"])

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(item.get("ground_truth", ""))

    # ── Build Ragas Dataset ──────────────────────────────────────────────
    logger.info("Building Ragas evaluation dataset…")
    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    logger.info("Running Ragas evaluation with 3 metrics…")
    try:
        ragas_result = evaluate(
            dataset=eval_dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
            ],
        )
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        raise RuntimeError(f"Ragas evaluation error: {exc}") from exc

    # ── Format and print results ─────────────────────────────────────────
    scores = {k: float(v) for k, v in ragas_result.items() if isinstance(v, (int, float))}

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Score':>10}")
    print("-" * 60)
    for metric_name, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric_name:<30} {score:>10.4f}  {bar}")
    print("=" * 60 + "\n")

    # ── Save results ─────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    logger.info("Results saved to %s", output_path)

    return scores


if __name__ == "__main__":
    print("=" * 60)
    print("  Legal & Financial CRAG — Ragas Evaluation")
    print("=" * 60)

    try:
        results = run_evaluation()
        print("✅ Evaluation complete.")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)
