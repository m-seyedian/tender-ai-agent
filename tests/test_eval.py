import json
from pathlib import Path
from eval.evaluate_rag import DATASET_PATH, RESULTS_MD_PATH, run_benchmark

def test_golden_dataset_has_20_valid_questions():
    raw = json.loads(DATASET_PATH.read_text(encoding='utf-8-sig'))
    assert len(raw['questions']) == 20

def test_rag_evaluation_benchmark_meets_quality_thresholds(tmp_path: Path):
    report_path = tmp_path / 'eval_results.md'
    _, hybrid_metrics = run_benchmark(output_path=report_path)
    assert report_path.exists()
    assert RESULTS_MD_PATH.exists()
    assert hybrid_metrics.hit_at_1 >= 90.0
    assert hybrid_metrics.hit_at_3 >= 95.0
    assert hybrid_metrics.mrr >= 0.95
