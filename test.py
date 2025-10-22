import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import pipeline

try:
    import sacrebleu  # type: ignore
except Exception:  # pragma: no cover
    sacrebleu = None
def read_lines_with_optional_refs(input_path: Path, limit: Optional[int] = None) -> Tuple[List[str], List[Optional[str]]]:
    """Read dataset file. Supports two formats:
    - Single column: English source per line (no references available)
    - TSV with >=7 columns: last two columns are English source and Malayalam reference
    Returns sources list and parallel references list (None where not available).
    """
    sources: List[str] = []
    references: List[Optional[str]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                # Assume last two fields are English then Malayalam reference
                eng = parts[-2].strip()
                ml = parts[-1].strip()
                if eng:
                    sources.append(eng)
                    references.append(ml if ml else None)
            else:
                # Fallback: treat entire line as English source
                sources.append(line)
                references.append(None)
            if limit is not None and len(sources) >= limit:
                break
    return sources, references


def batched(iterable: List[str], batch_size: int) -> List[List[str]]:
    return [iterable[i : i + batch_size] for i in range(0, len(iterable), batch_size)]


def run_translations(sources: List[str], model_name: str, device: str, batch_size: int = 16) -> List[str]:
    device_id = 0 if device == "cuda" else -1
    translator = pipeline("translation", model=model_name, device=device_id)
    outputs: List[str] = []
    for chunk in tqdm(batched(sources, batch_size), total=(len(sources) + batch_size - 1) // batch_size, desc=f"{model_name}"):
        preds = translator(chunk)
        for item in preds:
            outputs.append(item["translation_text"])
    return outputs


def compute_metrics(hypotheses: List[str], references: List[Optional[str]]) -> dict:
    metrics = {}
    if sacrebleu is None:
        return metrics
    # Keep only examples with references
    paired_refs = [r for r in references if r is not None]
    paired_hyps = [h for h, r in zip(hypotheses, references) if r is not None]
    if not paired_refs:
        return metrics
    bleu = sacrebleu.corpus_bleu(paired_hyps, [paired_refs])
    chrf = sacrebleu.corpus_chrf(paired_hyps, [paired_refs])
    metrics.update({
        "bleu": getattr(bleu, "score", None),
        "chrf": getattr(chrf, "score", None),
        # Additional common fields when available across sacrebleu versions
        "bleu_bp": getattr(bleu, "bp", None),
        "bleu_precisions": getattr(bleu, "precisions", None),
        "chrf_beta": getattr(chrf, "beta", None),
        "num_scored": len(paired_refs),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EN→ML translation models and compare outputs")
    parser.add_argument("--input", type=str, required=True, help="Path to input text/TSV file")
    parser.add_argument("--out_csv", type=str, default="translation_eval.csv", help="Where to save side-by-side CSV")
    parser.add_argument("--metrics_json", type=str, default="translation_metrics.json", help="Where to save metrics JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for a quick run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for pipeline calls")
    parser.add_argument("--report_txt", type=str, default="translation_report.txt", help="Where to save a human-readable text report")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_csv = Path(args.out_csv)
    metrics_json = Path(args.metrics_json)
    report_txt = Path(args.report_txt)

    sources, references = read_lines_with_optional_refs(input_path, limit=args.limit)
    if not sources:
        raise SystemExit("No inputs found in the provided file")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    new_model = "facebook/nllb-200-1.3B"
    baseline_model = "Helsinki-NLP/opus-mt-en-ml"

    print(f"Running on {len(sources)} examples (device={device})…")

    preds_new = run_translations(sources, new_model, device=device, batch_size=args.batch_size)
    preds_base = run_translations(sources, baseline_model, device=device, batch_size=args.batch_size)

    # Write side-by-side CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "english", "reference_ml", "new_model_ml", "baseline_ml"])
        for i, (src, ref, hyp_n, hyp_b) in enumerate(zip(sources, references, preds_new, preds_base)):
            writer.writerow([i, src, ref or "", hyp_n, hyp_b])

    # Compute metrics when references exist
    metrics = {
        "new_model": compute_metrics(preds_new, references),
        "baseline": compute_metrics(preds_base, references),
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV → {out_csv}")
    print(f"Saved metrics → {metrics_json}")
    if sacrebleu is None:
        print("Note: sacrebleu not installed; metrics skipped. Install and re-run for BLEU/chrF.")

    # Write human-readable text report
    report_txt.parent.mkdir(parents=True, exist_ok=True)
    sample_n = min(20, len(sources))
    with report_txt.open("w", encoding="utf-8") as rf:
        rf.write("EN→ML Translation Evaluation Report\n")
        rf.write("=" * 40 + "\n\n")
        rf.write(f"Total examples processed: {len(sources)}\n")
        if metrics.get("new_model"):
            rf.write("\nNew model metrics:\n")
            for k, v in metrics["new_model"].items():
                rf.write(f"  - {k}: {v}\n")
        if metrics.get("baseline"):
            rf.write("\nBaseline model metrics:\n")
            for k, v in metrics["baseline"].items():
                rf.write(f"  - {k}: {v}\n")
        if sacrebleu is None:
            rf.write("\nNote: sacrebleu not installed; metrics omitted.\n")

        rf.write("\nSample comparisons (first {sample_n}):\n\n")
        for i in range(sample_n):
            rf.write(f"Index: {i}\n")
            rf.write(f"EN: {sources[i]}\n")
            if references[i]:
                rf.write(f"REF: {references[i]}\n")
            rf.write(f"NEW: {preds_new[i]}\n")
            rf.write(f"BASE: {preds_base[i]}\n")
            rf.write("-" * 40 + "\n")

    print(f"Saved text report → {report_txt}")


if __name__ == "__main__":
    main()

