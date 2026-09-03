
"""快速测试当前Agent配置和降级机制。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from jiuan.pipeline import langchain_agent

print("="*60)
print("测试 LangChain Agent 配置")
print("="*60)
print()

# 测试1：检查LangChain是否可用
print("1. 检查LangChain是否可用...")
if langchain_agent.available():
    print("   ✓ LangChain 可用")
else:
    print("   ✗ LangChain 不可用，请安装 requirements-agent.txt")
    exit(1)

# 测试2：读取配置
print()
print("2. 读取配置...")
from jiuan.common import load_config
cfg = load_config()
chat_cfg = langchain_agent._chat_cfg(cfg)
print(f"   base_url: {chat_cfg.get('base_url')}")
print(f"   model: {chat_cfg.get('model')}")
print(f"   api_key: {'已设置' if langchain_agent._resolve_api_key(cfg) else '未设置'}")

# 测试3：创建会话
print()
print("3. 创建会话...")
session = langchain_agent.create_session(memory_window=10)
session_id = session["session_id"]
print(f"   会话ID: {session_id}")

# 测试4：测试对话
print()
print("4. 测试对话（简单问题）...")
message = "你好，请介绍一下你自己"
print(f"   用户输入: {message}")

result = langchain_agent.chat(session_id, message)
print(f"   回答: {result['answer']}")
print(f"   是否降级: {result['tool_calls'][0].get('tool', '').startswith('(降级')}")

print()
print("="*60)
print("测试完成")
print("="*60)

