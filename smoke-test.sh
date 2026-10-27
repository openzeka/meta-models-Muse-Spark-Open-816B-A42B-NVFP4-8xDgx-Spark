#!/usr/bin/env bash
# Muse-Spark-Open-816B-A42B smoke test — run right after the server reports ready.
set -u
M=${M:-http://192.168.1.153:8000/v1/chat/completions}

echo "== 1) model list"
curl -s -m 10 "${M%/chat/completions}/models" | head -c 300; echo; echo

echo "== 2) plain answer (NOTE THE REASONING FIELD — content may arrive late or be empty)"
curl -s -m 300 "$M" -H 'Content-Type: application/json' -d '{
  "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
  "seed": 7, "temperature": 0.7, "top_p": 0.9,
  "max_tokens": 1500,
  "messages": [{"role": "user", "content": "Tell a story about a day a cat and a mouse spent in the forest."}]
}' | python3 -c '
import json, sys
d = json.load(sys.stdin)
msg = d["choices"][0]["message"]
print("finish_reason:", d["choices"][0].get("finish_reason"), "| usage:", d.get("usage"))
for k in ("content", "reasoning"):
    v = msg.get(k)
    if v:
        w = v.split()
        rep = [w[i] for i in range(len(w)-5) if len(set(w[i:i+5])) == 1]
        print(f"--- {k}: {len(w)} words | 5+ identical words in a row: {rep[:3] if rep else (chr(79)+chr(75))}")
        print(v[-200:])
'

echo
echo "== 3) tool-call (NOTE: the recipe runs the Python frontend — VLLM_USE_RUST_FRONTEND:"
echo "     '0' — so tool-calls work. If you switch to '1' for perf, the Rust frontend silently"
echo "     IGNORES auto tool choice and this check prints NONE.)"
curl -s -m 120 "$M" -H 'Content-Type: application/json' -d '{
  "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
  "max_tokens": 200,
  "messages": [{"role": "user", "content": "What is the weather in Istanbul?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "tool_choice": "auto"
}' | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
print("tool_calls:", m.get("tool_calls") or "NONE (see note above)")
'

echo
echo "== 4) spec metrics from the log (keep 'sparkrun logs' open):"
echo "   grep 'SpecDecoding metrics' -> Mean acceptance length / Avg Draft acceptance rate"
echo "   Rule: acceptance >95% sustained is most likely a LOOP; benchmark on loop-free text."