"""
Round 3 Test 3: Full RAG Pipeline
Uses top 2 embedding models + top 2 chat models (4 combinations).
Tests with 10 documents and 5 queries, evaluating answer quality.
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

# These will be determined from Test 1 & Test 2 results, but we pre-configure based on Round 2 findings
TOP_EMBEDDING_MODELS = [
    "text-embedding-qwen3-embedding-0.6b",   # Best discrimination
    "text-embedding-embeddinggemma-300m",     # Best throughput
]

TOP_CHAT_MODELS = [
    "google/gemma-4-e2b",   # Best speed + quality
    "qwen/qwen3.5-9b",     # Best Chinese understanding (thinking disabled)
]

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
    {"id": "Q1", "text": "中登产品有什么功能？", "expected_doc_id": 1, "expected_answer_terms": ["账户管理", "资金清算", "对账"]},
    {"id": "Q2", "text": "融资租赁行业规模多大？", "expected_doc_id": 2, "expected_answer_terms": ["8万亿", "25%"]},
    {"id": "Q3", "text": "如何建设数据中台？", "expected_doc_id": 9, "expected_answer_terms": ["统一数据标准", "数据资产目录", "数据服务能力"]},
    {"id": "Q4", "text": "容器编排工具有哪些？", "expected_doc_id": 6, "expected_answer_terms": ["Kubernetes", "容器编排", "部署"]},
    {"id": "Q5", "text": "敏捷开发的核心是什么？", "expected_doc_id": 10, "expected_answer_terms": ["迭代", "持续交付", "快速响应"]},
]

TOP_K = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            str(TESTS_DIR / "round3_test3_rag.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_rag_answer(answer: str, expected_terms: list) -> dict:
    """Score a RAG answer based on expected terms and quality."""
    found_terms = [t for t in expected_terms if t in answer]
    coverage = len(found_terms) / len(expected_terms) if expected_terms else 0

    score = 3  # baseline
    if coverage >= 0.8:
        score += 2
    elif coverage >= 0.5:
        score += 1
    if len(answer.strip()) < 10:
        score = 1
    elif len(answer.strip()) > 30:
        score += 0  # no extra for length
    # Check for hallucination (adding info not in docs)
    hallucination_terms = ["不确定", "可能", "猜测"]
    has_hedging = any(t in answer for t in hallucination_terms)

    return {
        "score": min(5, max(1, score)),
        "found_terms": found_terms,
        "missing_terms": [t for t in expected_terms if t not in answer],
        "coverage": round(coverage, 4),
        "has_hedging": has_hedging,
    }


def run_rag_pipeline(client: OpenAI, embed_model: str, chat_model: str) -> dict:
    """Run full RAG pipeline with specified models for all queries."""
    pipeline_result = {
        "embedding_model": embed_model,
        "chat_model": chat_model,
        "embed_time_sec": 0,
        "queries": [],
        "overall_answer_quality": 0,
        "overall_retrieval_accuracy": 0,
        "error": None,
    }

    # Step 1: Embed all documents
    logger.info(f"  Embedding {len(DOCUMENTS)} documents with {embed_model}...")
    try:
        doc_texts = [d["text"] for d in DOCUMENTS]
        start = time.time()
        doc_response = client.embeddings.create(model=embed_model, input=doc_texts)
        doc_elapsed = time.time() - start
        doc_embeddings = [item.embedding for item in doc_response.data]
        pipeline_result["embed_time_sec"] = round(doc_elapsed, 2)
        pipeline_result["embed_dimensions"] = len(doc_embeddings[0]) if doc_embeddings else 0
        logger.info(f"    Embedded {len(doc_texts)} docs in {doc_elapsed:.2f}s")
    except Exception as e:
        pipeline_result["error"] = f"Document embedding failed: {e}"
        logger.error(f"    Error embedding documents: {e}")
        return pipeline_result

    # Step 2: Process each query
    total_quality = 0
    total_retrieval_hits = 0

    for query_info in QUERIES:
        q_result = {
            "query_id": query_info["id"],
            "query_text": query_info["text"],
            "expected_doc_id": query_info["expected_doc_id"],
            "retrieval": {},
            "generation": {},
        }

        # 2a: Embed query
        try:
            start = time.time()
            q_response = client.embeddings.create(model=embed_model, input=[query_info["text"]])
            q_elapsed = time.time() - start
            q_embedding = q_response.data[0].embedding
            q_result["retrieval"]["query_embed_time_sec"] = round(q_elapsed, 2)
        except Exception as e:
            q_result["retrieval"]["error"] = str(e)
            pipeline_result["queries"].append(q_result)
            continue

        # 2b: Compute similarities and rank
        sims = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = cosine_similarity(q_embedding, doc_emb)
            sims.append({
                "doc_id": DOCUMENTS[i]["id"],
                "similarity": round(sim, 4),
            })
        sims.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = sims[:TOP_K]
        q_result["retrieval"]["top_k"] = top_k
        q_result["retrieval"]["best_match_doc_id"] = top_k[0]["doc_id"]
        q_result["retrieval"]["best_match_similarity"] = top_k[0]["similarity"]
        q_result["retrieval"]["correct_doc_rank"] = next(
            (i + 1 for i, s in enumerate(sims) if s["doc_id"] == query_info["expected_doc_id"]), -1
        )
        q_result["retrieval"]["hit"] = (top_k[0]["doc_id"] == query_info["expected_doc_id"])

        if q_result["retrieval"]["hit"]:
            total_retrieval_hits += 1

        logger.info(
            f"    [{query_info['id']}] Retrieval: best={top_k[0]['doc_id']} "
            f"(sim={top_k[0]['similarity']:.4f}), expected={query_info['expected_doc_id']}, "
            f"hit={q_result['retrieval']['hit']}"
        )

        # 2c: Build RAG context from top-K docs
        context_docs = []
        for rank_item in top_k:
            doc = next(d for d in DOCUMENTS if d["id"] == rank_item["doc_id"])
            context_docs.append(f"[文档{doc['id']}] {doc['text']}")
        context_text = "\n\n".join(context_docs)

        rag_prompt = (
            f"请根据以下参考文档回答问题。只使用文档中提供的信息，如果文档中没有相关信息，请说明。\n\n"
            f"参考文档：\n{context_text}\n\n"
            f"问题：{query_info['text']}\n\n"
            f"回答："
        )

        # 2d: Generate answer
        try:
            start = time.time()
            chat_response = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            chat_elapsed = time.time() - start

            answer = chat_response.choices[0].message.content or ""
            q_result["generation"] = {
                "answer": answer,
                "has_content": len(answer.strip()) > 0,
                "time_sec": round(chat_elapsed, 2),
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "tokens_per_second": round(
                    chat_response.usage.completion_tokens / chat_elapsed, 2
                ) if chat_elapsed > 0 and chat_response.usage.completion_tokens > 0 else 0,
            }

            # Score answer
            if answer.strip():
                scoring = score_rag_answer(answer, query_info["expected_answer_terms"])
                q_result["generation"]["quality_score"] = scoring["score"]
                q_result["generation"]["found_terms"] = scoring["found_terms"]
                q_result["generation"]["missing_terms"] = scoring["missing_terms"]
                q_result["generation"]["term_coverage"] = scoring["coverage"]
                total_quality += scoring["score"]
            else:
                q_result["generation"]["quality_score"] = 1
                total_quality += 1

            logger.info(
                f"    [{query_info['id']}] Generation: quality={q_result['generation']['quality_score']}, "
                f"time={chat_elapsed:.2f}s, tps={q_result['generation']['tokens_per_second']}"
            )
            logger.info(f"    [{query_info['id']}] Answer: {answer[:200]}...")

        except Exception as e:
            q_result["generation"] = {"error": str(e), "has_content": False, "quality_score": 0}
            logger.error(f"    [{query_info['id']}] Generation error: {e}")

        pipeline_result["queries"].append(q_result)
        time.sleep(1)

    pipeline_result["overall_answer_quality"] = round(total_quality / len(QUERIES), 2)
    pipeline_result["overall_retrieval_accuracy"] = round(total_retrieval_hits / len(QUERIES), 4)

    logger.info(
        f"  Pipeline summary: avg_quality={pipeline_result['overall_answer_quality']}, "
        f"retrieval_accuracy={pipeline_result['overall_retrieval_accuracy']}"
    )

    return pipeline_result


def run_round3_test3() -> dict:
    """Run Round 3 Test 3: Full RAG Pipeline."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Round 3 Test 3: Full RAG Pipeline",
        "top_embedding_models": TOP_EMBEDDING_MODELS,
        "top_chat_models": TOP_CHAT_MODELS,
        "num_documents": len(DOCUMENTS),
        "num_queries": len(QUERIES),
        "top_k": TOP_K,
        "pipelines": [],
        "comparison": {},
    }

    for embed_model in TOP_EMBEDDING_MODELS:
        for chat_model in TOP_CHAT_MODELS:
            logger.info(f"=== RAG Pipeline: {embed_model} + {chat_model} ===")
            pipeline_result = run_rag_pipeline(client, embed_model, chat_model)
            results["pipelines"].append(pipeline_result)
            time.sleep(2)

    # Build comparison table
    for p in results["pipelines"]:
        key = f"{p['embedding_model'].split('-')[-1]}+{p['chat_model'].split('/')[-1]}"
        results["comparison"][key] = {
            "embedding_model": p["embedding_model"],
            "chat_model": p["chat_model"],
            "embed_time_sec": p["embed_time_sec"],
            "overall_answer_quality": p["overall_answer_quality"],
            "overall_retrieval_accuracy": p["overall_retrieval_accuracy"],
            "per_query": {},
        }
        for q in p["queries"]:
            results["comparison"][key]["per_query"][q["query_id"]] = {
                "retrieval_hit": q["retrieval"].get("hit", False),
                "generation_quality": q["generation"].get("quality_score", 0),
                "generation_time_sec": q["generation"].get("time_sec", 0),
                "generation_tps": q["generation"].get("tokens_per_second", 0),
                "term_coverage": q["generation"].get("term_coverage", 0),
            }

    # Save raw results
    with open(
        str(TESTS_DIR / "round3_test3_rag_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Round 3 Test 3 complete. Results saved.")
    return results


if __name__ == "__main__":
    run_round3_test3()
