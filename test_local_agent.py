"""测试修复后的本地 ReAct Agent。"""
import requests
import time

# 创建会话（用最新模型）
r = requests.post('http://127.0.0.1:8000/local_agent/session', json={'memory_window': 5})
d = r.json()
sid = d['session_id']
print(f"会话: {sid}, 模型: {d.get('model_id', 'N/A')[:30]}")
print()

# 测试1：问"你能做什么"（不需要工具，应该直接回答）
print("=" * 50)
print("测试1：你能做什么？")
t1 = time.time()
r1 = requests.post('http://127.0.0.1:8000/local_agent/chat',
    json={'session_id': sid, 'message': '你能做什么'}, timeout=120)
d1 = r1.json()
print(f"回答: {d1.get('answer', '')[:200]}")
print(f"工具调用: {len(d1.get('tool_calls', []))} 次")
print(f"迭代: {d1.get('iterations', 0)} 次, 耗时: {time.time()-t1:.1f}s")
print()

# 测试2：搜索知识库（需要工具）
print("=" * 50)
print("测试2：搜索应急预案的知识")
t2 = time.time()
r2 = requests.post('http://127.0.0.1:8000/local_agent/chat',
    json={'session_id': sid, 'message': '搜索应急预案的知识'}, timeout=120)
d2 = r2.json()
print(f"回答: {d2.get('answer', '')[:200]}")
print(f"工具调用: {len(d2.get('tool_calls', []))} 次")
for tc in d2.get('tool_calls', []):
    print(f"  - {tc.get('action', '')} -> {tc.get('observation', '')[:100]}")
print(f"迭代: {d2.get('iterations', 0)} 次, 耗时: {time.time()-t2:.1f}s")
print()

# 测试3：列出模型（需要工具）
print("=" * 50)
print("测试3：有哪些已训练的模型？")
t3 = time.time()
r3 = requests.post('http://127.0.0.1:8000/local_agent/chat',
    json={'session_id': sid, 'message': '有哪些已训练的模型'}, timeout=120)
d3 = r3.json()
print(f"回答: {d3.get('answer', '')[:200]}")
print(f"工具调用: {len(d3.get('tool_calls', []))} 次")
for tc in d3.get('tool_calls', []):
    print(f"  - {tc.get('action', '')} -> {tc.get('observation', '')[:100]}")
print(f"迭代: {d3.get('iterations', 0)} 次, 耗时: {time.time()-t3:.1f}s")
