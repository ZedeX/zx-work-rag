"""
Test 4: Capacity Test
Tests max context length, embedding capacity, and estimates batch processing time.
"""
import time
import json
import logging
from openai import OpenAI

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

CHAT_MODELS = [
    "google/gemma-4-e2b",
    "nvidia/nemotron-3-nano-4b",
    "qwen/qwen3.5-9b",
]

EMBEDDING_MODELS = [
    "text-embedding-qwen3-embedding-0.6b",
    "text-embedding-nomic-embed-text-v1.5",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"E:\git\zx-work-rag\tests\test4_capacity.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_model_info_from_api(client: OpenAI) -> list:
    """Get model info from LM Studio API."""
    models_info = []
    try:
        models = client.models.list().data
        for m in models:
            info = {
                "id": m.id,
                "object": m.object if hasattr(m, "object") else "unknown",
            }
            # Try to get extra metadata
            for attr in ["context_length", "max_context_length", "max_tokens",
                         "context_window", "max_input_tokens", "owned_by"]:
                if hasattr(m, attr):
                    info[attr] = getattr(m, attr)
            models_info.append(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
    return models_info


def probe_max_context(client: OpenAI, model: str) -> dict:
    """Probe the max context length by sending increasingly long prompts."""
    result = {
        "model": model,
        "max_context_found": 0,
        "method": "incremental_probe",
        "details": [],
    }

    # Test with progressively longer inputs
    test_lengths = [256, 512, 1024, 2048, 4096, 8192]
    base_char = "数字化转型是企业发展的必由之路，利用数字技术重塑业务流程。"

    for target_tokens in test_lengths:
        # Rough estimate: 1 Chinese char ≈ 1-2 tokens, use 1.5x multiplier
        text_length = int(target_tokens * 1.5)
        text = (base_char * ((text_length // len(base_char)) + 1))[:text_length]

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"请用一句话总结以下内容：\n{text}"}],
                temperature=0.1,
                max_tokens=50,
            )
            elapsed = time.time() - start
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0

            result["details"].append(
                {
                    "target_tokens": target_tokens,
                    "input_chars": len(text),
                    "actual_prompt_tokens": prompt_tokens,
                    "success": True,
                    "time_sec": round(elapsed, 2),
                }
            )
            result["max_context_found"] = prompt_tokens
            logger.info(
                f"[{model}] target={target_tokens}, actual_prompt={prompt_tokens}, "
                f"time={elapsed:.2f}s - OK"
            )
        except Exception as e:
            err_str = str(e)
            result["details"].append(
                {
                    "target_tokens": target_tokens,
                    "input_chars": len(text),
                    "success": False,
                    "error": err_str[:200],
                }
            )
            logger.info(
                f"[{model}] target={target_tokens} - FAILED: {err_str[:100]}"
            )
            break  # Stop probing on failure

        time.sleep(1)

    return result


def probe_embedding_capacity(client: OpenAI, model: str) -> dict:
    """Test embedding model capacity with increasing text lengths."""
    result = {
        "model": model,
        "max_tokens_found": 0,
        "details": [],
    }

    base_char = "数字化转型是企业发展的必由之路，利用数字技术重塑业务流程，优化组织结构。"
    test_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    for target_tokens in test_lengths:
        text_length = int(target_tokens * 1.5)
        text = (base_char * ((text_length // len(base_char)) + 1))[:text_length]

        try:
            start = time.time()
            response = client.embeddings.create(model=model, input=[text])
            elapsed = time.time() - start
            tokens = response.usage.prompt_tokens if response.usage else 0

            result["details"].append(
                {
                    "target_tokens": target_tokens,
                    "input_chars": len(text),
                    "actual_tokens": tokens,
                    "success": True,
                    "time_sec": round(elapsed, 2),
                }
            )
            result["max_tokens_found"] = tokens
            logger.info(
                f"[{model}] embed target={target_tokens}, actual={tokens}, "
                f"time={elapsed:.2f}s - OK"
            )
        except Exception as e:
            err_str = str(e)
            result["details"].append(
                {
                    "target_tokens": target_tokens,
                    "input_chars": len(text),
                    "success": False,
                    "error": err_str[:200],
                }
            )
            logger.info(
                f"[{model}] embed target={target_tokens} - FAILED: {err_str[:100]}"
            )
            break

        time.sleep(0.5)

    return result


def estimate_batch_embedding_time(client: OpenAI, model: str) -> dict:
    """Estimate time to embed 80,000 documents."""
    result = {
        "model": model,
        "total_docs": 80000,
        "avg_tokens_per_doc": 1000,
        "batch_size_tested": 10,
        "time_for_batch_sec": 0,
        "estimated_total_time_sec": 0,
        "estimated_total_time_hours": 0,
        "error": None,
    }

    # Generate sample texts (~1000 tokens each ≈ 1500 Chinese chars)
    base_text = "数字化转型是企业发展的必由之路。通过数字技术重塑业务流程，优化组织结构，提升客户体验。数据驱动决策，技术赋能创新，推动企业高质量发展。"  # ~70 chars
    sample_text = (base_text * 22)[:1500]  # ~1500 chars ≈ 1000 tokens

    batch_texts = [sample_text] * result["batch_size_tested"]

    try:
        start = time.time()
        response = client.embeddings.create(model=model, input=batch_texts)
        elapsed = time.time() - start

        tokens_in_batch = response.usage.total_tokens if response.usage else 0
        result["time_for_batch_sec"] = round(elapsed, 2)
        result["tokens_in_batch"] = tokens_in_batch

        # Extrapolate
        num_batches = result["total_docs"] // result["batch_size_tested"]
        result["estimated_total_time_sec"] = round(num_batches * elapsed, 2)
        result["estimated_total_time_hours"] = round(
            num_batches * elapsed / 3600, 2
        )

        logger.info(
            f"[{model}] Batch of {result['batch_size_tested']} in {elapsed:.2f}s, "
            f"estimated total: {result['estimated_total_time_hours']}h"
        )
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{model}] Batch estimation error: {e}")

    return result


def run_test4() -> dict:
    """Run all capacity tests."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Test 4: Capacity Test",
        "model_info": [],
        "chat_context_probes": [],
        "embedding_capacity_probes": [],
        "batch_estimates": [],
    }

    # Get model info from API
    logger.info("Getting model info from API...")
    results["model_info"] = get_model_info_from_api(client)

    # Probe chat model context lengths
    for model in CHAT_MODELS:
        logger.info(f"=== Probing context length: {model} ===")
        probe = probe_max_context(client, model)
        results["chat_context_probes"].append(probe)

    # Probe embedding model capacity
    for model in EMBEDDING_MODELS:
        logger.info(f"=== Probing embedding capacity: {model} ===")
        probe = probe_embedding_capacity(client, model)
        results["embedding_capacity_probes"].append(probe)

    # Estimate batch embedding time
    for model in EMBEDDING_MODELS:
        logger.info(f"=== Estimating batch time: {model} ===")
        est = estimate_batch_embedding_time(client, model)
        results["batch_estimates"].append(est)

    # Save raw results
    with open(
        r"E:\git\zx-work-rag\tests\test4_capacity_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Test 4 complete. Results saved to test4_capacity_results.json")
    return results


if __name__ == "__main__":
    run_test4()
