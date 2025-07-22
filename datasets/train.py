#!/usr/bin/env python3
"""
MLX-optimized cybersecurity fine-tuning pipeline for Apple Silicon.
Leverages Apple's Metal Performance Shaders for 10-20x speedup.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import mlx.core as mx
from mlx_lm import load, convert, generate
from mlx_lm.tuner import train, utils
import argparse

class MLXCybersecurityFineTuner:
    """Apple Silicon optimized fine-tuning pipeline for cybersecurity models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        self.output_dir = config['output_dir']
        self.experiment_name = config.get('experiment_name', f"mlx_cybersec_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # MLX-specific paths
        self.mlx_model_dir = Path(self.output_dir) / "mlx_model"
        self.adapters_dir = Path(self.output_dir) / "adapters"
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # Do NOT pre-create mlx_model_dir; let converter create it
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup experiment logging"""
        self.log_file = Path(self.output_dir) / f"{self.experiment_name}.log"
        
    def log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def convert_model_to_mlx(self):
        """Convert HuggingFace model to MLX format"""
        self.log(f"🔄 Converting {self.model_name} to MLX format...")
        
        # Check if MLX model already exists
        if (self.mlx_model_dir / "config.json").exists():
            self.log("✅ MLX model already exists, skipping conversion")
            return str(self.mlx_model_dir)
        
        try:
            # Convert model to MLX format
            convert(
                hf_path=self.model_name,
                mlx_path=str(self.mlx_model_dir),
                quantize=False,  # Keep full precision for training
                dtype="float16"  # Use float16 for memory efficiency
            )
            self.log("✅ Model converted to MLX format successfully")
            return str(self.mlx_model_dir)
            
        except Exception as e:
            self.log(f"❌ Error converting model: {e}")
            raise
    
    def prepare_training_data(self) -> str:
        """Prepare data in MLX training format"""
        self.log("📚 Preparing training data for MLX...")
        
        # Load and format data
        train_data = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Convert to MLX training format
                if 'messages' in data:
                    # Convert conversation format to single text
                    conversation_text = ""
                    for message in data['messages']:
                        role = message['role']
                        content = message['content']
                        if role == 'system':
                            conversation_text += f"System: {content}\n\n"
                        elif role == 'user':
                            conversation_text += f"User: {content}\n\n"
                        elif role == 'assistant':
                            conversation_text += f"Assistant: {content}\n\n"
                    train_data.append({"text": conversation_text.strip()})
                elif 'text' in data:
                    # Already in MLX format
                    train_data.append({"text": data['text']})
        
        # Save in JSONL format for MLX
        mlx_train_file = Path(self.output_dir) / "mlx_train_data.jsonl"
        with open(mlx_train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        self.log(f"✅ Prepared {len(train_data):,} training samples for MLX")
        return str(mlx_train_file)
    
    def prepare_validation_data(self) -> Optional[str]:
        """Prepare validation data in MLX format"""
        val_path = self.dataset_path.replace('_train.', '_valid.')
        
        if not os.path.exists(val_path):
            self.log("⚠️ No validation file found, skipping validation")
            return None
        
        self.log("📚 Preparing validation data for MLX...")
        
        val_data = []
        with open(val_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                if 'messages' in data:
                    conversation_text = ""
                    for message in data['messages']:
                        role = message['role']
                        content = message['content']
                        
                        if role == 'system':
                            conversation_text += f"System: {content}\n\n"
                        elif role == 'user':
                            conversation_text += f"User: {content}\n\n"
                        elif role == 'assistant':
                            conversation_text += f"Assistant: {content}\n\n"
                    
                    val_data.append({"text": conversation_text.strip()})
        
        mlx_val_file = Path(self.output_dir) / "mlx_val_data.jsonl"
        with open(mlx_val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        self.log(f"✅ Prepared {len(val_data):,} validation samples for MLX")
        return str(mlx_val_file)
    
    def run_mlx_training(self, train_file: str, val_file: Optional[str] = None):
        """Run MLX-optimized training"""
        self.log("🚀 Starting MLX training (Apple Silicon optimized)...")
        
        # MLX training configuration
        train_args = [
            "--model", str(self.mlx_model_dir),
            "--data", train_file,
            "--train",
            "--lora-layers", str(self.config.get('lora_layers', 16)),
            "--batch-size", str(self.config.get('batch_size', 4)),
            "--learning-rate", str(self.config.get('learning_rate', 1e-4)),
            "--num-epochs", str(self.config.get('num_epochs', 2)),
            "--steps-per-report", str(self.config.get('logging_steps', 10)),
            "--steps-per-eval", str(self.config.get('eval_steps', 100)),
            "--adapter-path", str(self.adapters_dir),
            "--save-every", str(self.config.get('save_steps', 500)),
            "--max-seq-length", str(self.config.get('max_length', 1024))
        ]
        
        if val_file:
            train_args.extend(["--val-data", val_file])
        
        self.log(f"📊 Training configuration: {' '.join(train_args)}")
        
        # Start training
        start_time = time.time()
        
        try:
            from mlx_lm.tuner.trainer import train, TrainingArgs
            import mlx_lm
            # Load MLX model
            model_tuple = mlx_lm.load(str(self.mlx_model_dir))
            if isinstance(model_tuple, tuple):
                model = model_tuple[0]
            else:
                model = model_tuple
            # Initialize optimizer using MLX core
            import mlx.optimizers as optim
            optimizer = optim.AdamW(learning_rate=self.config['learning_rate'])
            # Load training dataset from provided file
            train_dataset = []
            with open(train_file, 'r') as f:
                for line in f:
                    train_dataset.append(json.loads(line))
            self.log(f"Loaded {len(train_dataset)} training samples.")
            if len(train_dataset) > 0:
                self.log(f"Sample format: {type(train_dataset[0])}, value: {train_dataset[0]}")
            # Load validation dataset from provided file
            val_dataset = []
            if val_file and os.path.exists(val_file):
                with open(val_file, 'r') as f:
                    for line in f:
                        val_dataset.append(json.loads(line))
            else:
                val_dataset = None
            # Prepare training arguments
            training_args = TrainingArgs(
                batch_size=self.config['batch_size'],
            )
            # Run training
            train(model, optimizer, train_dataset, val_dataset, training_args)
            
            training_time = time.time() - start_time
            self.log(f"✅ MLX training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"❌ Training error: {e}\n{tb}")
            raise
    
    def test_model_inference(self):
        """Test the trained model with sample inference"""
        self.log("🧪 Testing trained model inference...")
        
        try:
            # Load the trained model with adapters
            model, tokenizer = load(
                path_or_hf_repo=str(self.mlx_model_dir),
                adapter_path=str(self.adapters_dir)
            )
            
            # Test with a cybersecurity question
            test_prompt = "User: How can I identify SQL injection vulnerabilities in a web application?\n\nAssistant:"
            
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=test_prompt,
                max_tokens=200,
                temp=0.7
            )
            
            self.log(f"✅ Model test successful!")
            self.log(f"Test prompt: {test_prompt}")
            self.log(f"Model response: {response}")
            
        except Exception as e:
            self.log(f"⚠️ Model test failed: {e}")
    
    def run_complete_pipeline(self, train_file, val_file=None):
        """Run the complete MLX-optimized fine-tuning pipeline using provided train/val files"""
        start_time = time.time()
        self.log(f"🎯 Starting MLX cybersecurity fine-tuning pipeline: {self.experiment_name}")
        self.log(f"💻 Running on Apple Silicon with MLX optimization")
        self.log(f"�️ Device: {mx.default_device()}")
        self.log(f"📊 Active memory: {mx.get_active_memory() / 1024**3:.1f} GB")
        try:
            # Step 1: Convert model to MLX format
            mlx_model_path = self.convert_model_to_mlx()
            # Step 2: Run MLX training with provided files
            self.run_mlx_training(train_file, val_file)
            # Step 3: Test the model
            self.test_model_inference()
            total_time = time.time() - start_time
            self.log("🎉 MLX fine-tuning pipeline completed successfully!")
            self.log(f"⏱️ Total time: {total_time:.2f} seconds")
            self.log(f"📁 Output directory: {self.output_dir}")
            self.log(f"🧠 MLX model: {self.mlx_model_dir}")
            self.log(f"🔧 Adapters: {self.adapters_dir}")
            return {
                'success': True,
                'total_time': total_time,
                'output_dir': str(self.output_dir),
                'mlx_model_dir': str(self.mlx_model_dir),
                'adapters_dir': str(self.adapters_dir)
            }
        except Exception as e:
            self.log(f"❌ Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main execution function"""
    
    # MLX-optimized configuration for Apple Silicon
    config = {
        'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'dataset_path': '/Users/danielrodrigo/Workspace/datasets/cybersecurity_datasets/processed/cybersecurity_agentic_clean_train.jsonl',
        'output_dir': '/Users/danielrodrigo/Workspace/datasets/cybersecurity_finetuned_models/mlx_optimized',
        'experiment_name': f'mlx_cybersec_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # MLX-optimized hyperparameters for Apple Silicon
        'num_epochs': 2,
        'batch_size': 8,  # Higher batch size possible with MLX
        'learning_rate': 1e-4,
        'max_length': 512,  # Reduced for faster training
        'lora_layers': 16,
        'eval_steps': 50,
        'save_steps': 200,
        'logging_steps': 10
    }
    parser = argparse.ArgumentParser(description="MLX-optimized TinyLlama fine-tuning pipeline")
    parser.add_argument('--train-file', type=str, required=True, help='Path to MLX-compatible training dataset (JSONL)')
    parser.add_argument('--val-file', type=str, default=None, help='Path to MLX-compatible validation dataset (JSONL)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for fine-tuned model')
    parser.add_argument('--num-epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--lora-layers', type=int, default=16, help='Number of LoRA layers')
    parser.add_argument('--eval-steps', type=int, default=50, help='Evaluation steps')
    parser.add_argument('--save-steps', type=int, default=200, help='Save steps')
    parser.add_argument('--logging-steps', type=int, default=10, help='Logging steps')
    args = parser.parse_args()

    config = {
        'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'output_dir': args.output_dir,
        'experiment_name': f'mlx_cybersec_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'lora_layers': args.lora_layers,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'logging_steps': args.logging_steps
    }
    # Initialize and run MLX fine-tuner
    fine_tuner = MLXCybersecurityFineTuner(config)
    result = fine_tuner.run_complete_pipeline(args.train_file, args.val_file)

    if result['success']:
        print("\n" + "="*60)
        print("🎉 MLX CYBERSECURITY FINE-TUNING COMPLETED!")
        print("="*60)
        print(f"🚀 Apple Silicon acceleration: ENABLED")
        print(f"⏱️ Total time: {result['total_time']:.2f} seconds")
        print(f"📁 Output: {result['output_dir']}")
        print("="*60)
    else:
        print(f"\n❌ Training failed: {result['error']}")

if __name__ == "__main__":
    main()
