"""Quick diagnostic for qwen3.5-9b thinking mode."""
from openai import OpenAI
import time

c = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
start = time.time()
r = c.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[{"role": "user", "content": "你好，请用一句话介绍自己"}],
    temperature=0.1,
    max_tokens=2048,
)
elapsed = time.time() - start

content = r.choices[0].message.content
reasoning = getattr(r.choices[0].message, "reasoning_content", None)

print(f"time: {elapsed:.1f}s")
print(f"finish_reason: {r.choices[0].finish_reason}")
print(f"completion_tokens: {r.usage.completion_tokens}")

reasoning_tokens = 0
if r.usage and r.usage.completion_tokens_details:
    reasoning_tokens = r.usage.completion_tokens_details.reasoning_tokens or 0
print(f"reasoning_tokens: {reasoning_tokens}")
print(f"content_tokens: {r.usage.completion_tokens - reasoning_tokens}")

if content:
    print(f"content: {content[:300]}")
else:
    print("content: EMPTY")

if reasoning:
    print(f"reasoning_len: {len(reasoning)}")
    print(f"reasoning_tail: {reasoning[-200:]}")
