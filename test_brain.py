from langchain_ollama import ChatOllama

# 1. 初始化大脑 (指向你下载好的 llama3.2)
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. 发送指令
print("--- CALLING local brain (Ollama) ---")
try:
    # 问一个和你的项目相关的问题
    response = llm.invoke("Answer me in Chinese：Why is an AI Agent suitable for handling complex financial legal compliance audits?")

    # 3. 打印结果
    print("\nanswer from AI：")
    print("-" * 30)
    print(response.content)
    print("-" * 30)
except Exception as e:
    print(f"\nsorry error, plz check your backend service, error: {e}")