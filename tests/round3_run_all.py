"""
Round 3: Comprehensive Benchmark - Master Runner
Runs all 3 tests sequentially and consolidates results.
"""
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, r"E:\git\zx-work-rag\tests")

from round3_test1_chat import run_round3_test1
from round3_test2_embedding import run_round3_test2
from round3_test3_rag import run_round3_test3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            r"E:\git\zx-work-rag\tests\round3_master.log", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    start_time = datetime.now()
    logger.info(f"=== Round 3 Comprehensive Benchmark Started at {start_time} ===")

    all_results = {
        "round": 3,
        "start_time": start_time.isoformat(),
        "tests": {},
    }

    # Test 1: Deep Chat Model Comparison
    logger.info("=" * 60)
    logger.info("Starting Test 1: Deep Chat Model Comparison")
    logger.info("=" * 60)
    try:
        t1_results = run_round3_test1()
        all_results["tests"]["test1_chat"] = {
            "status": "completed",
            "models_tested": t1_results["models_tested"],
            "model_summaries": t1_results["model_summaries"],
        }
    except Exception as e:
        all_results["tests"]["test1_chat"] = {"status": "failed", "error": str(e)}
        logger.error(f"Test 1 failed: {e}")

    # Test 2: Embedding Model Comprehensive Benchmark
    logger.info("=" * 60)
    logger.info("Starting Test 2: Embedding Model Comprehensive Benchmark")
    logger.info("=" * 60)
    try:
        t2_results = run_round3_test2()
        all_results["tests"]["test2_embedding"] = {
            "status": "completed",
            "models_tested": t2_results["models_tested"],
            "model_summary": t2_results["model_summary"],
        }
    except Exception as e:
        all_results["tests"]["test2_embedding"] = {"status": "failed", "error": str(e)}
        logger.error(f"Test 2 failed: {e}")

    # Test 3: Full RAG Pipeline
    logger.info("=" * 60)
    logger.info("Starting Test 3: Full RAG Pipeline")
    logger.info("=" * 60)
    try:
        t3_results = run_round3_test3()
        all_results["tests"]["test3_rag"] = {
            "status": "completed",
            "comparison": t3_results["comparison"],
        }
    except Exception as e:
        all_results["tests"]["test3_rag"] = {"status": "failed", "error": str(e)}
        logger.error(f"Test 3 failed: {e}")

    end_time = datetime.now()
    all_results["end_time"] = end_time.isoformat()
    all_results["total_duration_sec"] = (end_time - start_time).total_seconds()

    # Save consolidated results
    with open(
        r"E:\git\zx-work-rag\tests\round3_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"=== Round 3 Complete in {all_results['total_duration_sec']:.1f}s ===")
    logger.info(f"Results saved to round3_results.json")

    # Print quick summary
    print("\n" + "=" * 60)
    print("ROUND 3 QUICK SUMMARY")
    print("=" * 60)

    if "test1_chat" in all_results["tests"] and all_results["tests"]["test1_chat"]["status"] == "completed":
        summaries = all_results["tests"]["test1_chat"]["model_summaries"]
        print("\n--- Chat Model Comparison ---")
        for model, s in summaries.items():
            print(f"  {model}: quality={s['avg_quality_score']}, chinese={s['avg_chinese_quality']}, "
                  f"tps={s['avg_tokens_per_second']}, hallucinations={s['hallucination_count']}")

    if "test2_embedding" in all_results["tests"] and all_results["tests"]["test2_embedding"]["status"] == "completed":
        summaries = all_results["tests"]["test2_embedding"]["model_summary"]
        print("\n--- Embedding Model Benchmark ---")
        for s in summaries:
            model = s["model"]
            if s.get("loaded_ok"):
                print(f"  {model}: dim={s.get('dimensions')}, "
                      f"discrimination={s.get('discrimination_gap', 'N/A')}, "
                      f"hit_rate={s.get('retrieval_hit_rate', 'N/A')}, "
                      f"MRR={s.get('retrieval_mrr', 'N/A')}, "
                      f"throughput={s.get('throughput_texts_per_sec', 'N/A')} t/s")
            else:
                print(f"  {model}: FAILED TO LOAD")

    if "test3_rag" in all_results["tests"] and all_results["tests"]["test3_rag"]["status"] == "completed":
        comparison = all_results["tests"]["test3_rag"]["comparison"]
        print("\n--- RAG Pipeline Results ---")
        for key, val in comparison.items():
            print(f"  {key}: quality={val['overall_answer_quality']}, "
                  f"retrieval_acc={val['overall_retrieval_accuracy']}")


if __name__ == "__main__":
    main()
