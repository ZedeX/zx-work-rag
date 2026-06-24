"""
Round 2 Test 3: RAG Pipeline with Best Models
Uses the best embedding model found + qwen3.5-9b (thinking disabled) + gemma-4-e2b for comparison.
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

DOCUMENTS = [
    {
        "id": "Doc1",
        "text": "腾讯云中登产品是面向金融机构的中间件平台，提供账户管理、资金清算、对账等核心功能。",
    },
    {
        "id": "Doc2",
        "text": "融资租赁行业在2023年整体规模达到8万亿，其中汽车融资租赁占比约25%。",
    },
    {
        "id": "Doc3",
        "text": "数字化转型框架包括战略规划、组织变革、技术架构、数据治理四大支柱。",
    },
]

QUERY = "中登产品的核心功能有哪些？"

# Will test with multiple embedding models to find best
EMBEDDING_MODELS_TO_TEST = [
    "text-embedding-qwen3-embedding-0.6b",  # baseline
    "text-embedding-qwen3-embedding-4b",    # new
    "text-embedding-embeddinggemma-300m",   # new
]

CHAT_MODELS = [
    "qwen/qwen3.5-9b",
    "google/gemma-4-e2b",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            str(TESTS_DIR / "round2_test3_rag.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_rag_pipeline(client: OpenAI, embed_model: str, chat_model: str) -> dict:
    """Run a single RAG pipeline with specified models."""
    result = {
        "embedding_model": embed_model,
        "chat_model": chat_model,
        "steps": {},
        "error": None,
    }

    # Step 1: Embed documents
    logger.info(f"  Step 1: Embedding documents with {embed_model}...")
    try:
        doc_texts = [d["text"] for d in DOCUMENTS]
        start = time.time()
        doc_response = client.embeddings.create(model=embed_model, input=doc_texts)
        doc_elapsed = time.time() - start
        doc_embeddings = [item.embedding for item in doc_response.data]
        result["steps"]["embed_docs"] = {
            "time_sec": round(doc_elapsed, 2),
            "num_docs": len(doc_texts),
            "dimensions": len(doc_embeddings[0]) if doc_embeddings else 0,
        }
        logger.info(f"    Embedded {len(doc_texts)} docs in {doc_elapsed:.2f}s")
    except Exception as e:
        result["error"] = f"Embedding failed: {e}"
        logger.error(f"    Error embedding documents: {e}")
        return result

    # Step 2: Embed query
    logger.info(f"  Step 2: Embedding query with {embed_model}...")
    try:
        start = time.time()
        query_response = client.embeddings.create(model=embed_model, input=[QUERY])
        query_elapsed = time.time() - start
        query_embedding = query_response.data[0].embedding
        result["steps"]["embed_query"] = {
            "time_sec": round(query_elapsed, 2),
            "dimensions": len(query_embedding),
        }
        logger.info(f"    Embedded query in {query_elapsed:.2f}s")
    except Exception as e:
        result["error"] = f"Query embedding failed: {e}"
        logger.error(f"    Error embedding query: {e}")
        return result

    # Step 3: Compute similarity and rank
    logger.info(f"  Step 3: Computing similarity...")
    sims = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        sims.append({
            "doc_id": DOCUMENTS[i]["id"],
            "text_snippet": DOCUMENTS[i]["text"][:40] + "...",
            "similarity": round(sim, 4),
        })
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    result["steps"]["similarity_search"] = {
        "rankings": sims,
        "best_match": sims[0],
    }
    logger.info(f"    Best match: {sims[0]['doc_id']} (sim={sims[0]['similarity']:.4f})")

    # Step 4: Generate answer with RAG context
    logger.info(f"  Step 4: Generating answer with {chat_model}...")
    best_doc_id = sims[0]["doc_id"]
    best_doc_text = next(d["text"] for d in DOCUMENTS if d["id"] == best_doc_id)

    rag_prompt = (
        f"请根据以下参考文档回答问题。如果文档中没有相关信息，请说明。\n\n"
        f"参考文档：{best_doc_text}\n\n"
        f"问题：{QUERY}\n\n"
        f"回答："
    )

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
        result["steps"]["generate"] = {
            "answer": answer,
            "has_content": len(answer.strip()) > 0,
            "context_doc": best_doc_id,
            "time_sec": round(chat_elapsed, 2),
            "prompt_tokens": chat_response.usage.prompt_tokens,
            "completion_tokens": chat_response.usage.completion_tokens,
            "tokens_per_second": round(
                chat_response.usage.completion_tokens / chat_elapsed, 2
            ) if chat_elapsed > 0 else 0,
        }
        logger.info(f"    Generated answer in {chat_elapsed:.2f}s")
        logger.info(f"    Answer: {answer[:200]}...")
    except Exception as e:
        result["steps"]["generate"] = {"error": str(e), "has_content": False}
        logger.error(f"    Error generating answer: {e}")

    return result


def run_round2_test3() -> dict:
    """Run Round 2 Test 3: RAG pipeline with best models."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Round 2 Test 3: RAG Pipeline",
        "query": QUERY,
        "documents": [d["text"][:40] + "..." for d in DOCUMENTS],
        "pipelines": [],
    }

    # Test each embedding model with each chat model
    for embed_model in EMBEDDING_MODELS_TO_TEST:
        for chat_model in CHAT_MODELS:
            logger.info(f"=== RAG Pipeline: {embed_model} + {chat_model} ===")
            pipeline_result = run_rag_pipeline(client, embed_model, chat_model)
            results["pipelines"].append(pipeline_result)
            time.sleep(2)

    # Build comparison summary
    results["comparison"] = {}
    for p in results["pipelines"]:
        key = f"{p['embedding_model']}+{p['chat_model']}"
        gen_step = p["steps"].get("generate", {})
        sim_step = p["steps"].get("similarity_search", {})
        results["comparison"][key] = {
            "has_answer": gen_step.get("has_content", False),
            "answer_preview": (gen_step.get("answer", "")[:100] + "...") if gen_step.get("answer") else "",
            "best_match_doc": sim_step.get("best_match", {}).get("doc_id", "N/A"),
            "best_match_sim": sim_step.get("best_match", {}).get("similarity", 0),
            "generation_time_sec": gen_step.get("time_sec", 0),
            "generation_tps": gen_step.get("tokens_per_second", 0),
            "error": p.get("error"),
        }

    # Save raw results
    with open(
        str(TESTS_DIR / "round2_test3_rag_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Round 2 Test 3 complete. Results saved.")
    return results


if __name__ == "__main__":
    run_round2_test3()
