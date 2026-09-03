"""测试选择已训练模型创建会话。"""
import requests

# 获取模型列表
r = requests.get('http://127.0.0.1:8000/models')
models = r.json()
print(f'平台已训练模型数: {len(models)}')
for m in models[:5]:
    mid = m.get('model_id', '')
    name = m.get('name', '')
    backend = m.get('backend', '')
    print(f'  - {mid} | {name} | {backend}')

if models:
    mid = models[0]['model_id']
    # 用第一个模型创建会话
    r2 = requests.post('http://127.0.0.1:8000/chat/session', json={'memory_window': 10, 'model_id': mid})
    d = r2.json()
    print(f'\n创建会话(模型={mid})')
    print(f'  session_id: {d.get("session_id")}')
    print(f'  model_id: {d.get("model_id")}')

    # 测试对话
    r3 = requests.post('http://127.0.0.1:8000/chat', json={'session_id': d['session_id'], 'message': '你好'}, timeout=60)
    d3 = r3.json()
    print(f'\n对话测试:')
    print(f'  回答: {d3.get("answer", "")[:200]}')
    print(f'  工具调用: {d3.get("tool_calls", [])}')
