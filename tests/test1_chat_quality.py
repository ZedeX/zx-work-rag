"""
Test 1: Chat Model Quality Test
Tests each chat model with Chinese RAG-style prompts.
"""
import time
import json
import logging
from pathlib import Path
from openai import OpenAI

TESTS_DIR = Path(__file__).parent

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

CHAT_MODELS = [
    "google/gemma-4-e2b",
    "nvidia/nemotron-3-nano-4b",
    "qwen/qwen3.5-9b",
]

PROMPTS = [
    {
        "id": "general_knowledge",
        "prompt": "请根据以下文档内容回答问题：腾讯云中登产品的主要功能是什么？",
        "description": "通用知识问答",
    },
    {
        "id": "summarization",
        "prompt": "请总结以下内容的要点：数字化转型是指企业利用数字技术重塑业务流程、优化组织结构、提升客户体验的过程，其核心在于以数据驱动决策，以技术赋能创新。",
        "description": "摘要生成",
    },
    {
        "id": "factual_hallucination",
        "prompt": "2023年中国融资租赁行业规模是多少？请仅根据你确定的知识回答，如果不确定请说明。",
        "description": "事实性问答（幻觉测试）",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(TESTS_DIR / "test1_chat.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def test_chat_model(client: OpenAI, model: str, prompt_info: dict) -> dict:
    """Test a single chat model with a single prompt."""
    result = {
        "model": model,
        "prompt_id": prompt_info["id"],
        "description": prompt_info["description"],
        "response": "",
        "response_time_sec": 0,
        "tokens_per_second": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "error": None,
    }
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_info["prompt"]}],
            temperature=0.3,
            max_tokens=512,
        )
        elapsed = time.time() - start

        choice = response.choices[0]
        result["response"] = choice.message.content
        result["response_time_sec"] = round(elapsed, 2)

        if response.usage:
            result["prompt_tokens"] = response.usage.prompt_tokens
            result["completion_tokens"] = response.usage.completion_tokens
            result["total_tokens"] = response.usage.total_tokens
            if elapsed > 0 and response.usage.completion_tokens > 0:
                result["tokens_per_second"] = round(
                    response.usage.completion_tokens / elapsed, 2
                )

        logger.info(
            f"[{model}][{prompt_info['id']}] "
            f"time={elapsed:.2f}s, tokens={result['completion_tokens']}, "
            f"tps={result['tokens_per_second']}"
        )
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{model}][{prompt_info['id']}] Error: {e}")

    return result


def get_model_info(client: OpenAI, model: str) -> dict:
    """Try to get model info including context window."""
    info = {"model": model, "context_length": "unknown"}
    try:
        # LM Studio may expose model info via /v1/models
        models = client.models.list().data
        for m in models:
            if m.id == model:
                # Check for any extra fields
                if hasattr(m, "context_length"):
                    info["context_length"] = m.context_length
                if hasattr(m, "max_context_length"):
                    info["context_length"] = m.max_context_length
                break
    except Exception:
        pass
    return info


def run_test1() -> dict:
    """Run all chat model tests."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Test 1: Chat Model Quality",
        "models_info": [],
        "results": [],
    }

    # Get model info
    for model in CHAT_MODELS:
        info = get_model_info(client, model)
        results["models_info"].append(info)

    # Test each model with each prompt
    for model in CHAT_MODELS:
        logger.info(f"=== Testing model: {model} ===")
        for prompt_info in PROMPTS:
            r = test_chat_model(client, model, prompt_info)
            results["results"].append(r)
            # Small delay between requests
            time.sleep(1)

    # Save raw results
    with open(
        str(TESTS_DIR / "test1_chat_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Test 1 complete. Results saved to test1_chat_results.json")
    return results


if __name__ == "__main__":
    run_test1()
