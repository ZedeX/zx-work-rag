"""
Round 2 Test 2: New Embedding Models Test
Tests 5 NEW embedding models for vector quality, dimensions, similarity, and throughput.
Gracefully handles OOM errors for models too large for 6GB VRAM.
"""
import time
import json
import logging
import numpy as np
from pathlib import Path
from openai import OpenAI

TESTS_DIR = Path(__file__).parent

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

NEW_EMBEDDING_MODELS = [
    "text-embedding-qwen3-embedding-4b",
    "text-embedding-qwen3-embedding-8b",
    "text-embedding-embeddinggemma-300m",
    "text-embedding-embeddinggemma-300m-qat",
]

# Also include baseline for comparison
BASELINE_MODEL = "text-embedding-qwen3-embedding-0.6b"

CHINESE_TEXTS = [
    "数字化转型是企业发展的必由之路",
    "融资租赁行业在2023年规模达到8万亿",
    "腾讯云中登产品提供账户管理和资金清算功能",
    "人工智能技术正在改变金融行业的运作方式",
    "数据治理是数字化转型的核心组成部分",
]

SIMILARITY_PAIRS = [
    {
        "text1": "数字化转型方案",
        "text2": "企业数字化改造计划",
        "expected": "similar",
        "description": "语义相似对",
    },
    {
        "text1": "数字化转型方案",
        "text2": "今天天气很好",
        "expected": "dissimilar",
        "description": "语义不相似对",
    },
]

BATCH_THROUGHPUT_SIZE = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            str(TESTS_DIR / "round2_test2_embedding.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cosine_similarity(v1: list, v2: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def is_oom_error(error_msg: str) -> bool:
    """Check if error is OOM related."""
    oom_keywords = ["out of memory", "OOM", "CUDA", "VRAM", "memory", "alloc", "capacity"]
    return any(kw.lower() in error_msg.lower() for kw in oom_keywords)


def test_batch_embedding(client: OpenAI, model: str) -> dict:
    """Test batch embedding with 5 Chinese texts."""
    result = {
        "model": model,
        "test": "batch_embedding",
        "vector_dimensions": 0,
        "num_vectors": 0,
        "time_sec": 0,
        "oom": False,
        "error": None,
    }
    try:
        start = time.time()
        response = client.embeddings.create(model=model, input=CHINESE_TEXTS)
        elapsed = time.time() - start

        vectors = [item.embedding for item in response.data]
        result["num_vectors"] = len(vectors)
        result["vector_dimensions"] = len(vectors[0]) if vectors else 0
        result["time_sec"] = round(elapsed, 2)
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        logger.info(
            f"[{model}] Batch: {result['num_vectors']} vectors, "
            f"dim={result['vector_dimensions']}, time={elapsed:.2f}s"
        )
    except Exception as e:
        result["error"] = str(e)
        result["oom"] = is_oom_error(str(e))
        logger.error(f"[{model}] Batch embedding error: {e}")
    return result


def test_semantic_similarity(client: OpenAI, model: str) -> list:
    """Test semantic similarity with paired texts."""
    results = []
    for pair in SIMILARITY_PAIRS:
        r = {
            "model": model,
            "test": "semantic_similarity",
            "text1": pair["text1"],
            "text2": pair["text2"],
            "expected": pair["expected"],
            "description": pair["description"],
            "similarity": 0,
            "verdict": "",
            "oom": False,
            "error": None,
        }
        try:
            response = client.embeddings.create(
                model=model, input=[pair["text1"], pair["text2"]]
            )
            v1 = response.data[0].embedding
            v2 = response.data[1].embedding
            sim = cosine_similarity(v1, v2)
            r["similarity"] = round(sim, 4)

            if pair["expected"] == "similar":
                r["verdict"] = "PASS" if sim > 0.7 else "FAIL"
            else:
                r["verdict"] = "PASS" if sim < 0.5 else "FAIL"

            # Compute discrimination gap
            r["discrimination_gap"] = None  # filled later

            logger.info(
                f"[{model}][{pair['description']}] "
                f"similarity={sim:.4f}, verdict={r['verdict']}"
            )
        except Exception as e:
            r["error"] = str(e)
            r["oom"] = is_oom_error(str(e))
            logger.error(f"[{model}] Similarity test error: {e}")
        results.append(r)
    return results


def test_throughput(client: OpenAI, model: str, num_texts: int = BATCH_THROUGHPUT_SIZE) -> dict:
    """Measure embedding throughput with batch of texts."""
    result = {
        "model": model,
        "test": "throughput",
        "num_texts": num_texts,
        "total_time_sec": 0,
        "texts_per_second": 0,
        "oom": False,
        "error": None,
    }
    try:
        texts = CHINESE_TEXTS * (num_texts // len(CHINESE_TEXTS) + 1)
        texts = texts[:num_texts]

        start = time.time()
        response = client.embeddings.create(model=model, input=texts)
        elapsed = time.time() - start

        result["total_time_sec"] = round(elapsed, 2)
        result["texts_per_second"] = round(num_texts / elapsed, 2) if elapsed > 0 else 0
        result["total_tokens"] = response.usage.total_tokens

        logger.info(
            f"[{model}] Throughput: {num_texts} texts in {elapsed:.2f}s = "
            f"{result['texts_per_second']} texts/s"
        )
    except Exception as e:
        result["error"] = str(e)
        result["oom"] = is_oom_error(str(e))
        logger.error(f"[{model}] Throughput test error: {e}")
    return result


def run_round2_test2() -> dict:
    """Run Round 2 Test 2: New embedding model tests."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    all_models = NEW_EMBEDDING_MODELS + [BASELINE_MODEL]
    results = {
        "test_name": "Round 2 Test 2: New Embedding Models",
        "new_models": NEW_EMBEDDING_MODELS,
        "baseline_model": BASELINE_MODEL,
        "batch_results": [],
        "similarity_results": [],
        "throughput_results": [],
        "model_summary": [],
    }

    for model in all_models:
        logger.info(f"=== Testing embedding model: {model} ===")
        model_ok = True

        # Batch test
        r = test_batch_embedding(client, model)
        results["batch_results"].append(r)
        if r.get("oom"):
            model_ok = False
            logger.warning(f"[{model}] OOM detected, skipping remaining tests")

        time.sleep(1)

        if model_ok:
            # Similarity test
            sim_results = test_semantic_similarity(client, model)
            results["similarity_results"].extend(sim_results)
            # Check if similarity test had OOM
            if any(sr.get("oom") for sr in sim_results):
                model_ok = False

        time.sleep(1)

        if model_ok:
            # Throughput test
            t = test_throughput(client, model)
            results["throughput_results"].append(t)
            if t.get("oom"):
                model_ok = False
        else:
            results["throughput_results"].append({
                "model": model, "test": "throughput",
                "num_texts": BATCH_THROUGHPUT_SIZE,
                "total_time_sec": 0, "texts_per_second": 0,
                "oom": True, "error": "Skipped due to previous OOM"
            })

        time.sleep(1)

        # Build model summary
        summary = {
            "model": model,
            "is_new": model in NEW_EMBEDDING_MODELS,
            "loaded_ok": model_ok,
        }
        if model_ok:
            batch_r = next((b for b in results["batch_results"] if b["model"] == model), None)
            if batch_r:
                summary["dimensions"] = batch_r["vector_dimensions"]
            sim_rs = [s for s in results["similarity_results"] if s["model"] == model and not s.get("error")]
            if len(sim_rs) == 2:
                summary["similar_pair_score"] = sim_rs[0]["similarity"]
                summary["dissimilar_pair_score"] = sim_rs[1]["similarity"]
                summary["discrimination_gap"] = round(
                    sim_rs[0]["similarity"] - sim_rs[1]["similarity"], 4
                )
                summary["both_pass"] = all(s["verdict"] == "PASS" for s in sim_rs)
            thru_r = next((t for t in results["throughput_results"] if t["model"] == model), None)
            if thru_r and not thru_r.get("oom"):
                summary["throughput_texts_per_sec"] = thru_r["texts_per_second"]
        else:
            summary["dimensions"] = "N/A (OOM)"
            summary["discrimination_gap"] = "N/A"
            summary["both_pass"] = False

        results["model_summary"].append(summary)

    # Save raw results
    with open(
        str(TESTS_DIR / "round2_test2_embedding_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Round 2 Test 2 complete. Results saved.")
    return results


if __name__ == "__main__":
    run_round2_test2()
