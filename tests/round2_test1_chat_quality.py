"""
Round 2 Test 1: qwen3.5-9b Chat Quality (Thinking Mode DISABLED)
Tests qwen3.5-9b with thinking disabled, compares with gemma-4-e2b.
"""
import time
import json
import logging
from openai import OpenAI

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

CHAT_MODELS = [
    "qwen/qwen3.5-9b",
    "google/gemma-4-e2b",
]

PROMPTS = [
    {
        "id": "general_knowledge",
        "prompt": "请根据以下文档内容回答问题：腾讯云中登产品的主要功能是什么？",
        "description": "通用知识问答",
    },
    {
        "id": "summarization",
        "prompt": "请总结以下内容的要点：数字化转型是指企业利用数字技术重塑业务流程和组织模式，以实现业务增长和效率提升的过程。",
        "description": "摘要生成",
    },
    {
        "id": "factual_hallucination",
        "prompt": "2023年中国融资租赁行业规模是多少？",
        "description": "事实性问答（幻觉测试）",
    },
]

MAX_TOKENS = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            r"E:\git\zx-work-rag\tests\round2_test1_chat.log", encoding="utf-8"
        ),
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
        "has_content": False,
        "response_time_sec": 0,
        "tokens_per_second": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "quality_score": 0,
        "chinese_quality_score": 0,
        "error": None,
    }
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_info["prompt"]}],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - start

        choice = response.choices[0]
        content = choice.message.content or ""
        result["response"] = content
        result["has_content"] = len(content.strip()) > 0
        result["response_time_sec"] = round(elapsed, 2)

        # Check for reasoning/thinking content
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            result["reasoning_content_length"] = len(choice.message.reasoning_content)

        if response.usage:
            result["prompt_tokens"] = response.usage.prompt_tokens
            result["completion_tokens"] = response.usage.completion_tokens
            result["total_tokens"] = response.usage.total_tokens
            # Check for reasoning tokens in usage details
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens"):
                    result["reasoning_tokens"] = details.reasoning_tokens
            if elapsed > 0 and response.usage.completion_tokens > 0:
                result["tokens_per_second"] = round(
                    response.usage.completion_tokens / elapsed, 2
                )

        # Auto-score quality
        if result["has_content"]:
            result["quality_score"] = _score_quality(content, prompt_info["id"])
            result["chinese_quality_score"] = _score_chinese(content)
        else:
            result["quality_score"] = 1
            result["chinese_quality_score"] = 1

        logger.info(
            f"[{model}][{prompt_info['id']}] "
            f"time={elapsed:.2f}s, tokens={result['completion_tokens']}, "
            f"tps={result['tokens_per_second']}, "
            f"has_content={result['has_content']}, "
            f"quality={result['quality_score']}, chinese={result['chinese_quality_score']}"
        )
        logger.info(f"  Response preview: {content[:200]}...")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{model}][{prompt_info['id']}] Error: {e}")

    return result


def _score_quality(content: str, prompt_id: str) -> int:
    """Auto-score response quality 1-5."""
    score = 3  # baseline
    if len(content.strip()) < 10:
        return 1
    if len(content.strip()) > 30:
        score += 1
    # Check for structured output
    if any(marker in content for marker in ["1.", "2.", "-", "：", "："]):
        score += 1
    # Check for hallucination indicators on factual question
    if prompt_id == "factual_hallucination":
        if any(phrase in content for phrase in ["不确定", "无法确认", "建议查阅", "没有确切", "无法提供"]):
            score = min(score + 1, 5)  # Good: honest about uncertainty
        if any(phrase in content for phrase in ["万亿"]) and "不确定" not in content:
            score -= 1  # Possible hallucination with specific numbers
    if score > 5:
        score = 5
    if score < 1:
        score = 1
    return score


def _score_chinese(content: str) -> int:
    """Auto-score Chinese quality 1-5."""
    if not content.strip():
        return 1
    score = 4  # baseline for Chinese models
    # Check for translation artifacts
    if any(phrase in content for phrase in ["我将", "作为一个", "As an", "I am"]):
        score -= 1
    # Check for natural Chinese expression
    if any(phrase in content for phrase in ["综上所述", "总的来说", "主要包括", "核心功能"]):
        score += 1
    if score > 5:
        score = 5
    if score < 1:
        score = 1
    return score


def run_round2_test1() -> dict:
    """Run Round 2 Test 1: qwen3.5-9b chat quality with thinking disabled."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Round 2 Test 1: Chat Quality (Thinking Disabled)",
        "max_tokens": MAX_TOKENS,
        "models_tested": CHAT_MODELS,
        "results": [],
        "comparison": {},
    }

    for model in CHAT_MODELS:
        logger.info(f"=== Testing model: {model} ===")
        for prompt_info in PROMPTS:
            r = test_chat_model(client, model, prompt_info)
            results["results"].append(r)
            time.sleep(1)

    # Build comparison summary
    qwen_results = [r for r in results["results"] if "qwen3.5" in r["model"]]
    gemma_results = [r for r in results["results"] if "gemma" in r["model"]]

    results["comparison"] = {
        "qwen3.5-9b_avg_quality": round(
            sum(r["quality_score"] for r in qwen_results) / len(qwen_results), 2
        ) if qwen_results else 0,
        "qwen3.5-9b_avg_chinese": round(
            sum(r["chinese_quality_score"] for r in qwen_results) / len(qwen_results), 2
        ) if qwen_results else 0,
        "qwen3.5-9b_avg_tps": round(
            sum(r["tokens_per_second"] for r in qwen_results) / len(qwen_results), 2
        ) if qwen_results else 0,
        "qwen3.5-9b_has_content_ratio": sum(
            1 for r in qwen_results if r["has_content"]
        ) / len(qwen_results) if qwen_results else 0,
        "gemma-4-e2b_avg_quality": round(
            sum(r["quality_score"] for r in gemma_results) / len(gemma_results), 2
        ) if gemma_results else 0,
        "gemma-4-e2b_avg_chinese": round(
            sum(r["chinese_quality_score"] for r in gemma_results) / len(gemma_results), 2
        ) if gemma_results else 0,
        "gemma-4-e2b_avg_tps": round(
            sum(r["tokens_per_second"] for r in gemma_results) / len(gemma_results), 2
        ) if gemma_results else 0,
    }

    # Save raw results
    with open(
        r"E:\git\zx-work-rag\tests\round2_test1_chat_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Round 2 Test 1 complete. Results saved.")
    return results


if __name__ == "__main__":
    run_round2_test1()
