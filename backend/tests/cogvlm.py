from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import io

app = Flask(__name__)

# 模型和分词器初始化
MODEL_PATH = "THUDM/cogvlm2-llama3-chinese-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
).to(DEVICE).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# 全局变量用于存储对话历史
history = []

@app.route('/chat', methods=['POST'])
def chat():
    global history

    # 检查是否有文件上传
    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
    else:
        image = None

    # 获取文本查询
    query = request.form.get('query', '')

    # 处理文本查询
    if image is None:
        if not history:
            query = text_only_template.format(query)
        else:
            old_prompt = ''
            for _, (old_query, response) in enumerate(history):
                old_prompt += old_query + " " + response + "\n"
            query = old_prompt + "USER: {} ASSISTANT:".format(query)

    # 构建模型输入
    if image is None:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            template_version='chat'
        )
    else:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }

    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
    }

    # 生成响应
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]

    # 更新对话历史
    history.append((query, response))

    # 返回响应
    return jsonify({"response": response})

@app.route('/clear', methods=['POST'])
def clear_history():
    global history
    history = []
    return jsonify({"status": "history cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)