"""
Test 3: RAG Pipeline Simulation
Simulates a mini RAG pipeline: embed docs -> embed query -> cosine similarity -> generate answer.
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

CHAT_MODEL = "qwen/qwen3.5-9b"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(TESTS_DIR / "test3_rag.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_test3() -> dict:
    """Run the full RAG pipeline simulation."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Test 3: RAG Pipeline Simulation",
        "steps": {},
    }

    # Step 1: Embed documents
    logger.info("Step 1: Embedding documents...")
    step1 = {"step": "embed_documents", "doc_embeddings": [], "error": None}
    try:
        doc_texts = [d["text"] for d in DOCUMENTS]
        start = time.time()
        doc_response = client.embeddings.create(model=EMBEDDING_MODEL, input=doc_texts)
        doc_elapsed = time.time() - start

        for i, item in enumerate(doc_response.data):
            step1["doc_embeddings"].append(
                {
                    "doc_id": DOCUMENTS[i]["id"],
                    "dim": len(item.embedding),
                }
            )
        step1["time_sec"] = round(doc_elapsed, 2)
        step1["tokens"] = doc_response.usage.total_tokens
        logger.info(f"  Embedded {len(doc_texts)} docs in {doc_elapsed:.2f}s")
    except Exception as e:
        step1["error"] = str(e)
        logger.error(f"  Error embedding documents: {e}")
    results["steps"]["step1_embed_docs"] = step1

    # Step 2: Embed query
    logger.info("Step 2: Embedding query...")
    step2 = {"step": "embed_query", "query": QUERY, "error": None}
    try:
        start = time.time()
        query_response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=[QUERY]
        )
        query_elapsed = time.time() - start
        query_embedding = query_response.data[0].embedding
        step2["dim"] = len(query_embedding)
        step2["time_sec"] = round(query_elapsed, 2)
        logger.info(f"  Embedded query in {query_elapsed:.2f}s")
    except Exception as e:
        step2["error"] = str(e)
        query_embedding = None
        logger.error(f"  Error embedding query: {e}")
    results["steps"]["step2_embed_query"] = step2

    # Step 3: Compute cosine similarity and find most relevant document
    logger.info("Step 3: Computing cosine similarity...")
    step3 = {"step": "similarity_search", "similarities": [], "best_match": None, "error": None}
    if query_embedding and "error" not in step1 or step1.get("error") is None:
        try:
            doc_embeddings = [item.embedding for item in doc_response.data]
            sims = []
            for i, doc_emb in enumerate(doc_embeddings):
                sim = cosine_similarity(query_embedding, doc_emb)
                sims.append(
                    {
                        "doc_id": DOCUMENTS[i]["id"],
                        "text_snippet": DOCUMENTS[i]["text"][:30] + "...",
                        "similarity": round(sim, 4),
                    }
                )
            sims.sort(key=lambda x: x["similarity"], reverse=True)
            step3["similarities"] = sims
            step3["best_match"] = sims[0]
            logger.info(
                f"  Best match: {sims[0]['doc_id']} (sim={sims[0]['similarity']:.4f})"
            )
        except Exception as e:
            step3["error"] = str(e)
            logger.error(f"  Error computing similarity: {e}")
    results["steps"]["step3_similarity"] = step3

    # Step 4: Generate answer using chat model with context
    logger.info("Step 4: Generating answer with RAG context...")
    step4 = {"step": "generate_answer", "error": None}
    if step3.get("best_match"):
        try:
            best_doc_id = step3["best_match"]["doc_id"]
            best_doc_text = next(
                d["text"] for d in DOCUMENTS if d["id"] == best_doc_id
            )

            rag_prompt = (
                f"请根据以下参考文档回答问题。如果文档中没有相关信息，请说明。\n\n"
                f"参考文档：{best_doc_text}\n\n"
                f"问题：{QUERY}\n\n"
                f"回答："
            )

            start = time.time()
            chat_response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            chat_elapsed = time.time() - start

            answer = chat_response.choices[0].message.content
            step4["answer"] = answer
            step4["context_doc"] = best_doc_id
            step4["time_sec"] = round(chat_elapsed, 2)
            step4["prompt_tokens"] = chat_response.usage.prompt_tokens
            step4["completion_tokens"] = chat_response.usage.completion_tokens
            step4["tokens_per_second"] = (
                round(chat_response.usage.completion_tokens / chat_elapsed, 2)
                if chat_elapsed > 0
                else 0
            )
            logger.info(f"  Generated answer in {chat_elapsed:.2f}s")
            logger.info(f"  Answer: {answer[:100]}...")
        except Exception as e:
            step4["error"] = str(e)
            logger.error(f"  Error generating answer: {e}")
    results["steps"]["step4_generate"] = step4

    # Also test without RAG context for comparison
    logger.info("Step 4b: Generating answer WITHOUT RAG context (comparison)...")
    step4b = {"step": "generate_answer_no_context", "error": None}
    try:
        start = time.time()
        chat_response_no_ctx = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": QUERY}],
            temperature=0.1,
            max_tokens=256,
        )
        chat_elapsed_no_ctx = time.time() - start
        answer_no_ctx = chat_response_no_ctx.choices[0].message.content
        step4b["answer"] = answer_no_ctx
        step4b["time_sec"] = round(chat_elapsed_no_ctx, 2)
        logger.info(f"  No-context answer: {answer_no_ctx[:100]}...")
    except Exception as e:
        step4b["error"] = str(e)
        logger.error(f"  Error generating no-context answer: {e}")
    results["steps"]["step4b_no_context"] = step4b

    # Save raw results
    with open(
        str(TESTS_DIR / "test3_rag_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Test 3 complete. Results saved to test3_rag_results.json")
    return results


if __name__ == "__main__":
    run_test3()
