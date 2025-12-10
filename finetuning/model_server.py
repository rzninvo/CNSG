#!/usr/bin/env python3
"""
Local model server that exposes the LoRA-finetuned model via REST API
"""

from flask import Flask, request, jsonify
import torch
import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import time

########################################
#              CONFIG                  #
########################################

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER = "phi3-mr-lora-fixed-v3"
HOST = "127.0.0.1"  # localhost
PORT = 5000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

########################################
#          LOAD MODEL                  #
########################################

app = Flask(__name__)

# Global variables for model and tokenizer
_model = None
_tokenizer = None
_model_lock = threading.Lock()

def load_model():
    """Load the finetuned model and tokenizer"""
    global _model, _tokenizer
    
    logger.info("=" * 80)
    logger.info("Loading model and tokenizer...")
    logger.info("=" * 80)
    
    logger.info(f"Step 1/3: Loading tokenizer from {BASE_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.pad_token_id = _tokenizer.eos_token_id
    _tokenizer.padding_side = 'left'
    
    logger.info(f"Step 2/3: Loading base model from {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    logger.info(f"Step 3/3: Loading LoRA adapter from {LORA_ADAPTER}")
    _model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER,
        torch_dtype=torch.bfloat16,
    )
    _model.eval()
    
    # Verify device
    device = next(_model.parameters()).device
    logger.info(f"✅ Model loaded successfully on device: {device}")
    logger.info("=" * 80)

########################################
#          API ENDPOINTS               #
########################################

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if _model is None or _tokenizer is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'ready',
        'model': BASE_MODEL,
        'adapter': LORA_ADAPTER
    }), 200

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate text from the model
    
    Expected JSON body:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "max_new_tokens": 120,  // optional
        "temperature": 0.3,     // optional
        "do_sample": true       // optional
    }
    """
    if _model is None or _tokenizer is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        data = request.get_json()
        
        if 'messages' not in data:
            return jsonify({
                'error': 'Missing "messages" field in request body'
            }), 400
        
        messages = data['messages']
        max_new_tokens = data.get('max_new_tokens', 120)
        temperature = data.get('temperature', 0.3)
        do_sample = data.get('do_sample', True)
        
        logger.info(f"Received generation request: {len(messages)} messages, "
                   f"max_tokens={max_new_tokens}, temp={temperature}")
        
        # Generate response
        with _model_lock:
            response = generate_response(
                _model,
                messages,
                _tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
        
        return jsonify({
            'response': response,
            'num_tokens': len(_tokenizer.encode(response))
        }), 200
    
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/classify_intent', methods=['POST'])
def classify_intent():
    """
    Classify user intent for navigation
    
    Expected JSON body:
    {
        "user_input": "where is the kitchen?"
    }
    
    Returns "start" if navigation query, otherwise friendly response
    """
    if _model is None or _tokenizer is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        data = request.get_json()
        
        if 'user_input' not in data:
            return jsonify({
                'error': 'Missing "user_input" field in request body'
            }), 400
        
        user_input = data['user_input']
        
        system_prompt = """
You are an assistant for a home navigation system.
Your task is to interpret natural language queries from the user who might:
- ask to go to a room, or
- ask where to find an object.
- engage in friendly conversation.
If the user is asking for a room or an object, respond with "start".
If the user is not asking for navigation, respond in a friendly manner.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        logger.info(f"Classifying intent for: '{user_input[:50]}...'")
        
        with _model_lock:
            response = generate_response(
                _model,
                messages,
                _tokenizer,
                max_new_tokens=60,
                temperature=0.3,
                do_sample=True
            )
        
        is_navigation = "start" in response.lower()
        
        return jsonify({
            'user_input': user_input,
            'response': response,
            'is_navigation': is_navigation
        }), 200
    
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

########################################
#       INFERENCE FUNCTION             #
########################################

def generate_response(model, messages, tokenizer, max_new_tokens=120, 
                     temperature=0.3, do_sample=True):
    """Generate response from the model"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    input_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()

########################################
#          MAIN                        #
########################################

if __name__ == '__main__':
    logger.info("Starting model server...")
    
    # Load model before starting server
    load_model()
    
    logger.info(f"Starting Flask server on {HOST}:{PORT}")
    logger.info("Available endpoints:")
    logger.info(f"  - GET  http://{HOST}:{PORT}/health")
    logger.info(f"  - POST http://{HOST}:{PORT}/generate")
    logger.info(f"  - POST http://{HOST}:{PORT}/classify_intent")
    logger.info("=" * 80)
    
    # Run Flask server
    app.run(host=HOST, port=PORT, debug=False, threadsafe=True)