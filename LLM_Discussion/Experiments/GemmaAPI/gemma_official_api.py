from flask import Flask, request, jsonify
import torch
from transformers import pipeline
import gc

app = Flask(__name__)

# 只載入一次 pipeline，它會自動處理 model 和 tokenizer
torch.set_float32_matmul_precision('high')

text_pipeline = pipeline(
    task="text-generation",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

@app.route('/chat', methods=['POST'])
def generate_text():
    try:
        data = request.json
        
        if 'messages' not in data:
            return jsonify({'error': 'Missing messages parameter'}), 400
        
        messages = data['messages']
        # max_tokens = data.get('max_new_tokens', 50)
        print(f"\n📣:\n{messages}\n")
        
        if not isinstance(messages, list):
            return jsonify({'error': 'messages must be a list'}), 400
        
        # 生成文本
        with torch.no_grad():
            result = text_pipeline(messages, max_new_tokens=8192)
        print(f"\n💰:\n{result}\n")
        if result is not None:
            assistant_response = result[0]['generated_text'][1]['content']
            print(f"\n🤖:\n{assistant_response}\n")
        else:
            assistant_response = None
            print(f"\nNone\n")
        # 清理快取
        torch.cuda.empty_cache()
        gc.collect()
        
        return jsonify({
            'success': True,
            'result': result,
            'response': assistant_response,
            'input_messages_count': len(messages)
        })
        
    except Exception as e:
        torch.cuda.empty_cache()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/memory_status', methods=['GET'])
def memory_status():
    return jsonify({
        'gpu_memory_allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 2),
        'gpu_memory_reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 2),
    })

if __name__ == '__main__':
    print("Loading model...")
    print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("API ready!")
    app.run(host='127.0.0.1', port=8002, debug=False)