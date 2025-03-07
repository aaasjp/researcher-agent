import requests
import json
import time
from datetime import datetime

def stream_chat_completion():
    url = "http://120.222.7.197:1025/v1/chat/completions"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Qwen2.5-72B-Instruct",
        "messages": [{
            "role": "user",
            "content": "请帮我创作一个猪八戒和潘金莲结婚的故事，要求简短有深意，有故事性"
        }],
        "stream": True,
        "presence_penalty": 1.03,
        "frequency_penalty": 1.0,
        "repetition_penalty": 1.0,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 1,
        "seed": None,
        "stop": ["stop1", "stop2"],
        "stop_token_ids": [2, 13],
        "include_stop_str_in_output": False,
        "skip_special_tokens": True,
        "ignore_eos": False,
        "max_tokens": 2048
    }

    try:
        # 记录开始时间
        start_time = time.time()
        print(f"开始请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        line_count = 0  # 初始化行数计数器
        content_buffer = ""  # 用于存储完整内容

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                line_count += 1  # 增加计数
                if line.startswith('data: '):
                    json_str = line[6:]

                    if json_str.strip() == '[DONE]':
                        break

                    try:
                        json_response = json.loads(json_str)

                        if 'choices' in json_response:
                            content = json_response['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                print(content, end='', flush=True)
                    except json.JSONDecodeError as e:
                        print(f"\nJSON解析错误: {e}")
                        print(f"原始数据: {json_str}")

        # 计算并打印总用时
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\n结束请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"总用时: {elapsed_time:.2f} 秒")
        print(f"总接收行数: {line_count} 行")  # 打印总行数

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")

if __name__ == "__main__":
    stream_chat_completion()