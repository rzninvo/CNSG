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
from typing import Dict, List, Optional
from dataclasses import dataclass

########################################
#              CONFIG                  #
########################################

@dataclass
class ServerConfig:
    """Server configuration"""
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    lora_adapter: str = "phi3-mr-lora-fixed-v3"
    host: str = "127.0.0.1"
    port: int = 5000
    log_level: str = "INFO"


########################################
#          MODEL HANDLER               #
########################################

class ModelHandler:
    """Handles model loading and inference"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.lock = threading.Lock()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this handler"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        
        return logger
    
    def load(self) -> None:
        """Load the finetuned model and tokenizer"""
        self.logger.info("=" * 80)
        self.logger.info("Loading model and tokenizer...")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Step 1/3: Loading tokenizer from {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        
        self.logger.info(f"Step 2/3: Loading base model from {self.config.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            # torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="eager",
        )

        self.logger.info(f"Base model loaded on device: {next(base_model.parameters()).device}")
        
        self.logger.info(f"Step 3/3: Loading LoRA adapter from {self.config.lora_adapter}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.config.lora_adapter,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if torch.cuda.is_available():
            target_device = base_model.device if base_model.device.type == 'cuda' else 'cuda:0'
            self.logger.info(f"Moving finetuned model to {target_device}...")
            self.model = self.model.to(target_device)
            
        self.logger.info(f"Finetuned model loaded on device: {next(self.model.parameters()).device}")
        self.model.eval()
        
        # Verify device
        device = next(self.model.parameters()).device
        self.logger.info(f"✅ Model loaded successfully on device: {device}")
        self.logger.info("=" * 80)
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.tokenizer is not None
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 120,
        temperature: float = 0.3,
        do_sample: bool = True
    ) -> str:
        """Generate response from the model"""
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        input_len = inputs['input_ids'].shape[1]
        
        with self.lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )
        
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )
        return response.strip()


########################################
#          API SERVER                  #
########################################

class ModelServer:
    """Flask API server for the model"""
    
    INTENT_SYSTEM_PROMPT = """
You are an assistant for a home navigation system.
Your task is to interpret natural language queries from the user who might:
- ask to go to a room, or
- ask where to find an object.
- engage in friendly conversation.
If the user is asking for a room or an object, respond with "start".
If the user is not asking for navigation, respond in a friendly manner.
"""
    
    def __init__(self, model_handler: ModelHandler, config: ServerConfig):
        self.model_handler = model_handler
        self.config = config
        self.app = Flask(__name__)
        self.logger = self._setup_logger()
        self._setup_routes()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the server"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        
        return logger
    
    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        self.app.add_url_rule('/health', 'health', self._health_check, methods=['GET'])
        self.app.add_url_rule('/generate', 'generate', self._generate, methods=['POST'])
        self.app.add_url_rule('/classify_intent', 'classify_intent', 
                             self._classify_intent, methods=['POST'])
    
    def _health_check(self):
        """Health check endpoint"""
        if not self.model_handler.is_ready():
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 503
        
        return jsonify({
            'status': 'ready',
            'model': self.config.base_model,
            'adapter': self.config.lora_adapter
        }), 200
    
    def _generate(self):
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
        if not self.model_handler.is_ready():
            return jsonify({'error': 'Model not loaded'}), 503
        
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
            
            self.logger.info(
                f"Generation request: {len(messages)} messages, "
                f"max_tokens={max_new_tokens}, temp={temperature}"
            )
            
            response = self.model_handler.generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            return jsonify({
                'response': response,
                'num_tokens': len(self.model_handler.tokenizer.encode(response))
            }), 200
        
        except Exception as e:
            self.logger.error(f"Generation error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    def _classify_intent(self):
        """
        Classify user intent for navigation
        
        Expected JSON body:
        {
            "user_input": "where is the kitchen?"
        }
        """
        if not self.model_handler.is_ready():
            return jsonify({'error': 'Model not loaded'}), 503
        
        try:
            data = request.get_json()
            
            if 'user_input' not in data:
                return jsonify({
                    'error': 'Missing "user_input" field in request body'
                }), 400
            
            user_input = data['user_input']
            
            messages = [
                {"role": "system", "content": self.INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
            
            self.logger.info(f"Classifying intent: '{user_input[:50]}...'")
            
            response = self.model_handler.generate(
                messages=messages,
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
            self.logger.error(f"Classification error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    def run(self) -> None:
        """Start the Flask server"""
        self.logger.info(f"Starting Flask server on {self.config.host}:{self.config.port}")
        self.logger.info("Available endpoints:")
        self.logger.info(f"  - GET  http://{self.config.host}:{self.config.port}/health")
        self.logger.info(f"  - POST http://{self.config.host}:{self.config.port}/generate")
        self.logger.info(f"  - POST http://{self.config.host}:{self.config.port}/classify_intent")
        self.logger.info("=" * 80)
        
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
        )


########################################
#          MAIN                        #
########################################

def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = ServerConfig()
    
    # Initialize model handler
    model_handler = ModelHandler(config)
    model_handler.load()
    
    # Create and run server
    server = ModelServer(model_handler, config)
    server.run()


if __name__ == '__main__':
    main()