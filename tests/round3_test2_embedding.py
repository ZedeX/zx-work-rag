"""
Round 3 Test 2: Embedding Model Comprehensive Benchmark
Tests ALL 6 embedding models with:
A) Semantic Similarity Test (5 pairs)
B) Retrieval Accuracy Test (10 docs, 5 queries)
C) Throughput Test (100 texts)
D) Dimension check
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

ALL_EMBEDDING_MODELS = [
    "text-embedding-qwen3-embedding-0.6b",
    "text-embedding-qwen3-embedding-4b",
    "text-embedding-qwen3-embedding-8b",
    "text-embedding-embeddinggemma-300m",
    "text-embedding-embeddinggemma-300m-qat",
    "text-embedding-nomic-embed-text-v1.5",
]

# A) Semantic Similarity Test - 5 pairs
SIMILARITY_PAIRS = [
    {
        "text1": "数字化转型方案",
        "text2": "企业数字化改造计划",
        "expected": "similar",
        "description": "语义相似对-数字化转型",
    },
    {
        "text1": "数字化转型方案",
        "text2": "今天天气很好",
        "expected": "dissimilar",
        "description": "语义不相似对-天气",
    },
    {
        "text1": "腾讯云产品介绍",
        "text2": "腾讯云计算服务说明",
        "expected": "similar",
        "description": "语义相似对-腾讯云",
    },
    {
        "text1": "融资租赁合同",
        "text2": "汽车租赁协议",
        "expected": "somewhat_similar",
        "description": "语义部分相似对-租赁",
        "threshold_range": (0.4, 0.7),
    },
    {
        "text1": "机器学习算法",
        "text2": "今天中午吃什么",
        "expected": "dissimilar",
        "description": "语义不相似对-无关话题",
    },
]

# B) Retrieval Accuracy Test - 10 docs, 5 queries
DOCUMENTS = [
    {"id": 1, "text": "腾讯云中登产品是面向金融机构的中间件平台，提供账户管理、资金清算、对账等核心功能。"},
    {"id": 2, "text": "融资租赁行业在2023年整体规模达到8万亿，其中汽车融资租赁占比约25%。"},
    {"id": 3, "text": "数字化转型框架包括战略规划、组织变革、技术架构、数据治理四大支柱。"},
    {"id": 4, "text": "Python是一种广泛使用的高级编程语言，支持多种编程范式，包括面向对象和函数式编程。"},
    {"id": 5, "text": "2024年中国GDP增长目标设定为5%左右，强调高质量发展和新质生产力。"},
    {"id": 6, "text": "Kubernetes是一个开源的容器编排系统，用于自动化计算机应用程序的部署、扩展和管理。"},
    {"id": 7, "text": "项目风险管理包括风险识别、风险评估、风险应对和风险监控四个主要阶段。"},
    {"id": 8, "text": "深度学习是机器学习的一个分支，基于人工神经网络，特别适合处理图像和语音识别任务。"},
    {"id": 9, "text": "企业数据中台建设需要统一数据标准、构建数据资产目录、建立数据服务能力。"},
    {"id": 10, "text": "敏捷开发方法论强调迭代开发、持续交付和快速响应变化，Scrum是其中最流行的框架。"},
]

QUERIES = [
    {"id": "Q1", "text": "中登产品有什么功能？", "expected_doc_id": 1},
    {"id": "Q2", "text": "融资租赁行业规模多大？", "expected_doc_id": 2},
    {"id": "Q3", "text": "如何建设数据中台？", "expected_doc_id": 9},
    {"id": "Q4", "text": "容器编排工具有哪些？", "expected_doc_id": 6},
    {"id": "Q5", "text": "敏捷开发的核心是什么？", "expected_doc_id": 10},
]

# C) Throughput Test
THROUGHPUT_NUM_TEXTS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            str(TESTS_DIR / "round3_test2_embedding.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def is_oom_error(error_msg: str) -> bool:
    oom_keywords = ["out of memory", "OOM", "CUDA", "VRAM", "memory", "alloc", "capacity"]
    return any(kw.lower() in error_msg.lower() for kw in oom_keywords)


def test_dimensions(client: OpenAI, model: str) -> dict:
    """D) Check vector dimensions."""
    result = {"model": model, "test": "dimensions", "vector_dimensions": 0, "oom": False, "error": None}
    try:
        response = client.embeddings.create(model=model, input=["test"])
        result["vector_dimensions"] = len(response.data[0].embedding)
        logger.info(f"[{model}] Dimensions: {result['vector_dimensions']}")
    except Exception as e:
        result["error"] = str(e)
        result["oom"] = is_oom_error(str(e))
        logger.error(f"[{model}] Dimension check error: {e}")
    return result


def test_semantic_similarity(client: OpenAI, model: str) -> list:
    """A) Semantic Similarity Test with 5 pairs."""
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
            elif pair["expected"] == "dissimilar":
                r["verdict"] = "PASS" if sim < 0.5 else "FAIL"
            elif pair["expected"] == "somewhat_similar":
                low, high = pair.get("threshold_range", (0.4, 0.7))
                r["verdict"] = "PASS" if low <= sim <= high else "MARGINAL"

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


def test_retrieval_accuracy(client: OpenAI, model: str) -> dict:
    """B) Retrieval Accuracy Test with 10 docs and 5 queries."""
    result = {
        "model": model,
        "test": "retrieval_accuracy",
        "queries": [],
        "overall_hit_rate": 0,
        "overall_mrr": 0,
        "oom": False,
        "error": None,
    }

    try:
        # Embed all documents
        doc_texts = [d["text"] for d in DOCUMENTS]
        doc_response = client.embeddings.create(model=model, input=doc_texts)
        doc_embeddings = [item.embedding for item in doc_response.data]
        logger.info(f"[{model}] Embedded {len(doc_texts)} documents")

        hits = 0
        mrr_sum = 0.0

        for query_info in QUERIES:
            q_result = {
                "query_id": query_info["id"],
                "query_text": query_info["text"],
                "expected_doc_id": query_info["expected_doc_id"],
                "rankings": [],
                "hit": False,
                "reciprocal_rank": 0,
            }

            # Embed query
            q_response = client.embeddings.create(model=model, input=[query_info["text"]])
            q_embedding = q_response.data[0].embedding

            # Compute similarities
            sims = []
            for i, doc_emb in enumerate(doc_embeddings):
                sim = cosine_similarity(q_embedding, doc_emb)
                sims.append({
                    "doc_id": DOCUMENTS[i]["id"],
                    "text_snippet": DOCUMENTS[i]["text"][:30] + "...",
                    "similarity": round(sim, 4),
                })
            sims.sort(key=lambda x: x["similarity"], reverse=True)

            # Check if expected doc is in top position
            q_result["rankings"] = sims[:5]  # top 5
            top_doc_id = sims[0]["doc_id"]
            q_result["hit"] = (top_doc_id == query_info["expected_doc_id"])

            # Compute reciprocal rank
            for rank, s in enumerate(sims, 1):
                if s["doc_id"] == query_info["expected_doc_id"]:
                    q_result["reciprocal_rank"] = round(1.0 / rank, 4)
                    mrr_sum += 1.0 / rank
                    break

            if q_result["hit"]:
                hits += 1

            result["queries"].append(q_result)
            logger.info(
                f"[{model}][{query_info['id']}] "
                f"hit={q_result['hit']}, top_doc={top_doc_id}, "
                f"expected={query_info['expected_doc_id']}, "
                f"RR={q_result['reciprocal_rank']}"
            )

        result["overall_hit_rate"] = round(hits / len(QUERIES), 4)
        result["overall_mrr"] = round(mrr_sum / len(QUERIES), 4)
        logger.info(
            f"[{model}] Retrieval: hit_rate={result['overall_hit_rate']}, MRR={result['overall_mrr']}"
        )

    except Exception as e:
        result["error"] = str(e)
        result["oom"] = is_oom_error(str(e))
        logger.error(f"[{model}] Retrieval accuracy test error: {e}")

    return result


def test_throughput(client: OpenAI, model: str, num_texts: int = THROUGHPUT_NUM_TEXTS) -> dict:
    """C) Throughput Test with 100 texts."""
    result = {
        "model": model,
        "test": "throughput",
        "num_texts": num_texts,
        "total_time_sec": 0,
        "texts_per_second": 0,
        "tokens_per_second": 0,
        "oom": False,
        "error": None,
    }
    try:
        # Generate diverse Chinese texts for throughput test
        base_texts = [
            "数字化转型是企业发展的必由之路",
            "融资租赁行业在2023年规模达到8万亿",
            "腾讯云中登产品提供账户管理和资金清算功能",
            "人工智能技术正在改变金融行业的运作方式",
            "数据治理是数字化转型的核心组成部分",
            "Python编程语言在数据科学领域应用广泛",
            "Kubernetes容器编排简化了应用部署流程",
            "敏捷开发方法论提高了软件交付效率",
            "深度学习技术在图像识别领域取得突破",
            "企业数据中台建设需要统一数据标准",
        ]
        texts = (base_texts * (num_texts // len(base_texts) + 1))[:num_texts]

        start = time.time()
        response = client.embeddings.create(model=model, input=texts)
        elapsed = time.time() - start

        result["total_time_sec"] = round(elapsed, 2)
        result["texts_per_second"] = round(num_texts / elapsed, 2) if elapsed > 0 else 0
        result["total_tokens"] = response.usage.total_tokens
        if response.usage.total_tokens > 0:
            result["tokens_per_second"] = round(response.usage.total_tokens / elapsed, 2)

        logger.info(
            f"[{model}] Throughput: {num_texts} texts in {elapsed:.2f}s = "
            f"{result['texts_per_second']} texts/s, {result['tokens_per_second']} tokens/s"
        )
    except Exception as e:
        result["error"] = str(e)
        result["oom"] = is_oom_error(str(e))
        logger.error(f"[{model}] Throughput test error: {e}")
    return result


def run_round3_test2() -> dict:
    """Run Round 3 Test 2: Embedding Model Comprehensive Benchmark."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Round 3 Test 2: Embedding Model Comprehensive Benchmark",
        "models_tested": ALL_EMBEDDING_MODELS,
        "dimension_results": [],
        "similarity_results": [],
        "retrieval_results": [],
        "throughput_results": [],
        "model_summary": [],
    }

    for model in ALL_EMBEDDING_MODELS:
        logger.info(f"=== Testing embedding model: {model} ===")
        model_ok = True

        # D) Dimension check
        dim_r = test_dimensions(client, model)
        results["dimension_results"].append(dim_r)
        if dim_r.get("oom"):
            model_ok = False
            logger.warning(f"[{model}] OOM on dimension check, skipping remaining tests")
        time.sleep(1)

        # A) Semantic similarity
        if model_ok:
            sim_results = test_semantic_similarity(client, model)
            results["similarity_results"].extend(sim_results)
            if any(sr.get("oom") for sr in sim_results):
                model_ok = False
        time.sleep(1)

        # B) Retrieval accuracy
        if model_ok:
            ret_r = test_retrieval_accuracy(client, model)
            results["retrieval_results"].append(ret_r)
            if ret_r.get("oom"):
                model_ok = False
        time.sleep(1)

        # C) Throughput
        if model_ok:
            thru_r = test_throughput(client, model)
            results["throughput_results"].append(thru_r)
            if thru_r.get("oom"):
                model_ok = False
        else:
            results["throughput_results"].append({
                "model": model, "test": "throughput",
                "num_texts": THROUGHPUT_NUM_TEXTS,
                "total_time_sec": 0, "texts_per_second": 0,
                "oom": True, "error": "Skipped due to previous OOM"
            })
        time.sleep(1)

        # Build model summary
        summary = {
            "model": model,
            "loaded_ok": model_ok,
        }
        if model_ok:
            summary["dimensions"] = dim_r["vector_dimensions"]

            # Similarity scores
            sim_rs = [s for s in results["similarity_results"] if s["model"] == model and not s.get("error")]
            if sim_rs:
                summary["similarity_pairs"] = {}
                for s in sim_rs:
                    summary["similarity_pairs"][s["description"]] = {
                        "similarity": s["similarity"],
                        "expected": s["expected"],
                        "verdict": s["verdict"],
                    }
                # Compute discrimination: avg(similar) - avg(dissimilar)
                similar_scores = [s["similarity"] for s in sim_rs if s["expected"] == "similar"]
                dissimilar_scores = [s["similarity"] for s in sim_rs if s["expected"] == "dissimilar"]
                if similar_scores and dissimilar_scores:
                    avg_sim = sum(similar_scores) / len(similar_scores)
                    avg_dissim = sum(dissimilar_scores) / len(dissimilar_scores)
                    summary["avg_similar_score"] = round(avg_sim, 4)
                    summary["avg_dissimilar_score"] = round(avg_dissim, 4)
                    summary["discrimination_gap"] = round(avg_sim - avg_dissim, 4)
                pass_count = sum(1 for s in sim_rs if s["verdict"] == "PASS")
                summary["similarity_pass_rate"] = round(pass_count / len(sim_rs), 4)

            # Retrieval accuracy
            ret_r_model = next((r for r in results["retrieval_results"] if r["model"] == model), None)
            if ret_r_model and not ret_r_model.get("error"):
                summary["retrieval_hit_rate"] = ret_r_model["overall_hit_rate"]
                summary["retrieval_mrr"] = ret_r_model["overall_mrr"]

            # Throughput
            thru_r_model = next((t for t in results["throughput_results"] if t["model"] == model), None)
            if thru_r_model and not thru_r_model.get("oom"):
                summary["throughput_texts_per_sec"] = thru_r_model["texts_per_second"]
                summary["throughput_tokens_per_sec"] = thru_r_model.get("tokens_per_second", 0)
        else:
            summary["dimensions"] = "N/A (OOM)"
            summary["discrimination_gap"] = "N/A"
            summary["retrieval_hit_rate"] = "N/A"
            summary["throughput_texts_per_sec"] = "N/A"

        results["model_summary"].append(summary)

    # Save raw results
    with open(
        str(TESTS_DIR / "round3_test2_embedding_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Round 3 Test 2 complete. Results saved.")
    return results


if __name__ == "__main__":
    run_round3_test2()
