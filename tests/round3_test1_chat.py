"""
Round 3 Test 1: Deep Chat Model Comparison
Tests all 3 chat models with 5 Chinese RAG prompts.
Records full response, quality scores, Chinese quality, hallucination check, and speed metrics.
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

PROMPTS = [
    {
        "id": "general",
        "prompt": "请根据以下文档内容回答问题：腾讯云中登产品的主要功能是什么？",
        "description": "通用知识问答",
        "category": "general",
    },
    {
        "id": "summarization",
        "prompt": "请总结以下内容的要点：数字化转型是指企业利用数字技术重塑业务流程和组织模式，以实现业务增长和效率提升的过程。数字化转型需要从战略、组织、技术、数据四个维度进行系统性规划。",
        "description": "摘要生成",
        "category": "summarization",
    },
    {
        "id": "factual",
        "prompt": "2023年中国融资租赁行业规模是多少？",
        "description": "事实性问答（幻觉测试）",
        "category": "factual",
    },
    {
        "id": "reading_comprehension",
        "prompt": "根据以下信息，列出腾讯云中登产品的三个核心功能：腾讯云中登产品是面向金融机构的中间件平台，提供账户管理、资金清算、对账等核心功能，支持多币种结算和实时监控。",
        "description": "阅读理解",
        "category": "reading_comprehension",
    },
    {
        "id": "knowledge",
        "prompt": "请用中文解释什么是RAG（检索增强生成）技术？",
        "description": "知识问答",
        "category": "knowledge",
    },
]

MAX_TOKENS = 1024
TEMPERATURE = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            r"E:\git\zx-work-rag\tests\round3_test1_chat.log", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def test_chat_model(client: OpenAI, model: str, prompt_info: dict) -> dict:
    """Test a single chat model with a single prompt, recording detailed metrics."""
    result = {
        "model": model,
        "prompt_id": prompt_info["id"],
        "description": prompt_info["description"],
        "category": prompt_info["category"],
        "prompt": prompt_info["prompt"],
        "response": "",
        "has_content": False,
        "time_to_first_token_sec": 0,
        "total_time_sec": 0,
        "tokens_per_second": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "quality_score": 0,
        "chinese_quality_score": 0,
        "hallucination_flag": False,
        "hallucination_detail": "",
        "error": None,
    }
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_info["prompt"]}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - start

        choice = response.choices[0]
        content = choice.message.content or ""
        result["response"] = content
        result["has_content"] = len(content.strip()) > 0
        result["total_time_sec"] = round(elapsed, 2)

        # Check for reasoning/thinking content
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            result["reasoning_content_length"] = len(choice.message.reasoning_content)
            result["reasoning_content_preview"] = choice.message.reasoning_content[:500]

        if response.usage:
            result["prompt_tokens"] = response.usage.prompt_tokens
            result["completion_tokens"] = response.usage.completion_tokens
            result["total_tokens"] = response.usage.total_tokens
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens"):
                    result["reasoning_tokens"] = details.reasoning_tokens
            if elapsed > 0 and response.usage.completion_tokens > 0:
                result["tokens_per_second"] = round(
                    response.usage.completion_tokens / elapsed, 2
                )
            # Estimate time to first token (rough: total_time / completion_tokens * 1)
            # More accurate would need streaming, but we estimate
            if response.usage.completion_tokens > 0:
                result["time_to_first_token_sec"] = round(
                    elapsed / response.usage.completion_tokens, 3
                )

        # Auto-score quality
        if result["has_content"]:
            result["quality_score"] = _score_quality(content, prompt_info["category"])
            result["chinese_quality_score"] = _score_chinese(content)
            result["hallucination_flag"], result["hallucination_detail"] = _check_hallucination(
                content, prompt_info["category"]
            )
        else:
            result["quality_score"] = 1
            result["chinese_quality_score"] = 1
            result["hallucination_flag"] = False
            result["hallucination_detail"] = "No content generated"

        logger.info(
            f"[{model}][{prompt_info['id']}] "
            f"time={elapsed:.2f}s, tokens={result['completion_tokens']}, "
            f"tps={result['tokens_per_second']}, "
            f"quality={result['quality_score']}, chinese={result['chinese_quality_score']}, "
            f"hallucination={result['hallucination_flag']}"
        )
        logger.info(f"  Response: {content[:300]}...")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{model}][{prompt_info['id']}] Error: {e}")

    return result


def _score_quality(content: str, category: str) -> int:
    """Auto-score response quality 1-5 based on category."""
    score = 3  # baseline
    if len(content.strip()) < 10:
        return 1
    if len(content.strip()) < 30:
        return 2
    if len(content.strip()) > 50:
        score += 1
    if len(content.strip()) > 200:
        score += 1

    # Category-specific scoring
    if category == "summarization":
        # Check for structured output
        if any(marker in content for marker in ["1.", "2.", "3.", "-", "：", "第一", "第二"]):
            score += 1
        # Check for key terms from the source
        if any(term in content for term in ["战略", "组织", "技术", "数据", "数字化"]):
            score += 1
    elif category == "reading_comprehension":
        # Should list 3 core functions
        found = 0
        for term in ["账户管理", "资金清算", "对账"]:
            if term in content:
                found += 1
        if found >= 3:
            score += 2
        elif found >= 2:
            score += 1
    elif category == "factual":
        # Good: honest about uncertainty
        if any(phrase in content for phrase in ["不确定", "无法确认", "建议查阅", "没有确切", "无法提供", "没有公开"]):
            score = min(score + 2, 5)
        # Bad: hallucinates specific numbers
        if any(phrase in content for phrase in ["万亿"]) and not any(
            phrase in content for phrase in ["不确定", "无法", "没有", "建议"]
        ):
            score -= 2
    elif category == "knowledge":
        # Should explain RAG concept
        if any(term in content for term in ["检索", "增强", "生成", "Retrieval", "Augmented", "Generation"]):
            score += 1
        if any(term in content for term in ["向量", "嵌入", "embedding", "知识库", "文档"]):
            score += 1
    elif category == "general":
        # Should ask for document or give general answer
        if any(term in content for term in ["文档", "资料", "信息", "内容"]):
            score += 1

    return max(1, min(5, score))


def _score_chinese(content: str) -> int:
    """Auto-score Chinese language quality 1-5."""
    if not content.strip():
        return 1
    score = 3  # baseline
    # Natural Chinese expressions
    if any(phrase in content for phrase in ["综上所述", "总的来说", "主要包括", "核心功能", "具体来说", "简单来说"]):
        score += 1
    # Translation artifacts (bad)
    if any(phrase in content for phrase in ["我将", "作为一个", "As an", "I am", "Let me"]):
        score -= 1
    # Good Chinese structure
    if any(marker in content for marker in ["：", "；", "——", "、"]):
        score += 1
    # Mixed English (slight penalty for Chinese-focused tasks)
    eng_ratio = sum(1 for c in content if c.isascii() and c.isalpha()) / max(len(content), 1)
    if eng_ratio > 0.3:
        score -= 1
    return max(1, min(5, score))


def _check_hallucination(content: str, category: str) -> tuple:
    """Check for hallucination indicators. Returns (flag, detail)."""
    if category == "factual":
        # This question has no document context, so specific numbers are hallucinated
        if any(phrase in content for phrase in ["万亿", "亿元", "万元"]):
            if not any(phrase in content for phrase in ["不确定", "无法", "没有", "建议", "可能", "大约", "约"]):
                return True, "Hallucinated specific financial figures without uncertainty markers"
            else:
                return False, "Mentioned figures but with uncertainty markers"
        if any(phrase in content for phrase in ["不确定", "无法确认", "建议查阅", "没有确切"]):
            return False, "Honest about uncertainty - no hallucination"
        return False, "No specific figures claimed"

    if category == "reading_comprehension":
        # Should only list the 3 functions from the text
        extra_terms = ["风控", "合规", "审计", "反洗钱", "客户管理"]
        found_extras = [t for t in extra_terms if t in content]
        if found_extras:
            return True, f"Added functions not in source text: {found_extras}"
        return False, "Only used information from provided text"

    if category == "general":
        # No document provided, should not fabricate specific features
        if any(phrase in content for phrase in ["账户管理", "资金清算", "对账"]):
            if "文档" not in content and "资料" not in content:
                return True, "Fabricated specific features without document reference"
        return False, "No hallucination detected"

    return False, "N/A for this category"


def run_round3_test1() -> dict:
    """Run Round 3 Test 1: Deep Chat Model Comparison."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = {
        "test_name": "Round 3 Test 1: Deep Chat Model Comparison",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "models_tested": CHAT_MODELS,
        "prompts": [p["id"] for p in PROMPTS],
        "results": [],
        "model_summaries": {},
    }

    for model in CHAT_MODELS:
        logger.info(f"=== Testing model: {model} ===")
        model_results = []
        for prompt_info in PROMPTS:
            r = test_chat_model(client, model, prompt_info)
            results["results"].append(r)
            model_results.append(r)
            time.sleep(1)

        # Build model summary
        avg_quality = round(sum(r["quality_score"] for r in model_results) / len(model_results), 2)
        avg_chinese = round(sum(r["chinese_quality_score"] for r in model_results) / len(model_results), 2)
        avg_tps = round(sum(r["tokens_per_second"] for r in model_results) / len(model_results), 2)
        avg_time = round(sum(r["total_time_sec"] for r in model_results) / len(model_results), 2)
        hallucination_count = sum(1 for r in model_results if r["hallucination_flag"])
        content_rate = sum(1 for r in model_results if r["has_content"]) / len(model_results)

        results["model_summaries"][model] = {
            "avg_quality_score": avg_quality,
            "avg_chinese_quality": avg_chinese,
            "avg_tokens_per_second": avg_tps,
            "avg_total_time_sec": avg_time,
            "hallucination_count": hallucination_count,
            "content_output_rate": content_rate,
            "per_prompt_quality": {r["prompt_id"]: r["quality_score"] for r in model_results},
            "per_prompt_chinese": {r["prompt_id"]: r["chinese_quality_score"] for r in model_results},
        }

        logger.info(
            f"  Summary: avg_quality={avg_quality}, avg_chinese={avg_chinese}, "
            f"avg_tps={avg_tps}, hallucinations={hallucination_count}"
        )

    # Save raw results
    with open(
        r"E:\git\zx-work-rag\tests\round3_test1_chat_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Round 3 Test 1 complete. Results saved.")
    return results


if __name__ == "__main__":
    run_round3_test1()
