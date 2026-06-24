"""Quick test: AI Q&A mode with LM Studio chat model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.rag_query import RAGQueryService

svc = RAGQueryService(llm_mode="local")

# Test 1: Simple search
print("=== Search Test ===")
results = svc.search("数字化转型", top_k=3)
for r in results:
    print(f"  [{r['relevance']:.3f}] {r['metadata'].get('file_name','')}")

# Test 2: AI Q&A with increased max_tokens
print("\n=== AI Q&A Test ===")
# Patch max_tokens for thinking models
import openai
_orig = openai.resources.chat.completions.Completions.create
def _patched_create(self, **kwargs):
    kwargs.setdefault("max_tokens", 4000)
    return _orig(self, **kwargs)

# Just test directly
from openai import OpenAI
client = OpenAI(api_key="lm-studio", base_url="http://127.0.0.1:1234/v1")
resp = client.chat.completions.create(
    model="google/gemma-4-e2b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer in Chinese."},
        {"role": "user", "content": "Say hello in one sentence."},
    ],
    max_tokens=4000,
    temperature=0.3,
)
print(f"  Response: {resp.choices[0].message.content}")
print(f"  Reasoning tokens: {resp.usage.completion_tokens_details.reasoning_tokens}")
print(f"  Total tokens: {resp.usage.total_tokens}")

# Test 3: Full RAG Q&A
print("\n=== Full RAG Q&A Test ===")
answer = svc.ask("我做过哪些数字化转型项目？", top_k=5)
print(answer[:500])

svc.close()
