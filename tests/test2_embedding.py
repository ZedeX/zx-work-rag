"""
Test 2: Embedding Model Test
Tests both embedding models for vector quality, dimensions, similarity, and throughput.
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

EMBEDDING_MODELS = [
    "text-embedding-qwen3-embedding-0.6b",
    "text-embedding-nomic-embed-text-v1.5",
]

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(TESTS_DIR / "test2_embedding.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cosine_similarity(v1: list, v2: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_batch_embedding(client: OpenAI, model: str) -> dict:
    """Test batch embedding with 5 Chinese texts."""
    result = {
        "model": model,
        "test": "batch_embedding",
        "vector_dimensions": 0,
        "num_vectors": 0,
        "error": None,
    }
    try:
        response = client.embeddings.create(model=model, input=CHINESE_TEXTS)
        vectors = [item.embedding for item in response.data]
        result["num_vectors"] = len(vectors)
        result["vector_dimensions"] = len(vectors[0]) if vectors else 0
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        logger.info(
            f"[{model}] Batch: {result['num_vectors']} vectors, "
            f"dim={result['vector_dimensions']}"
        )
    except Exception as e:
        result["error"] = str(e)
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

            logger.info(
                f"[{model}][{pair['description']}] "
                f"similarity={sim:.4f}, verdict={r['verdict']}"
            )
        except Exception as e:
            r["error"] = str(e)
            logger.error(f"[{model}] Similarity test error: {e}")
        results.append(r)
    return results


def test_throughput(client: OpenAI, model: str, num_texts: int = 20) -> dict:
    """Measure embedding throughput."""
    result = {
        "model": model,
        "test": "throughput",
        "num_texts": num_texts,
        "total_time_sec": 0,
        "texts_per_second": 0,
        "error": None,
    }
    try:
        # Use repeated Chinese texts for throughput test
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
        logger.error(f"[{model}] Throughput test error: {e}")
    return result


def run_test2() -> dict:
    """Run all embedding model tests."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Test 2: Embedding Model",
        "batch_results": [],
        "similarity_results": [],
        "throughput_results": [],
    }

    for model in EMBEDDING_MODELS:
        logger.info(f"=== Testing embedding model: {model} ===")

        # Batch test
        r = test_batch_embedding(client, model)
        results["batch_results"].append(r)

        time.sleep(0.5)

        # Similarity test
        sim_results = test_semantic_similarity(client, model)
        results["similarity_results"].extend(sim_results)

        time.sleep(0.5)

        # Throughput test
        t = test_throughput(client, model)
        results["throughput_results"].append(t)

    # Save raw results
    with open(
        str(TESTS_DIR / "test2_embedding_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Test 2 complete. Results saved to test2_embedding_results.json")
    return results


if __name__ == "__main__":
    run_test2()
