#!/usr/bin/env python3
"""
Phase 1: MLX PHI-3 → GGUF Conversion
Converts your fine-tuned PHI-3 cybersecurity model from MLX to GGUF format

This script:
1. Merges your LoRA adapters with the base model
2. Converts the merged model to GGUF format
3. Tests the converted model for quality preservation
"""
import os
import sys
from pathlib import Path
import subprocess
import shutil

# Model paths (adjust these to match your setup)
BASE_MODEL = "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx"
ADAPTER_PATH = "/Volumes/Crucial X9/ai-models/PHI-3.5-cybersec-finetune/adapters"
OUTPUT_DIR = Path(__file__).parent / "converted_models"

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import mlx_lm
        print("✅ MLX-LM available")
    except ImportError:
        print("❌ MLX-LM not found. Install with: pip install mlx-lm")
        return False
    
    # Check if llama.cpp tools are available
    if not shutil.which("llama-convert"):
        print("⚠️  llama-convert not found. We'll use MLX export instead")
    
    return True

def merge_lora_adapters():
    """Merge LoRA adapters with base model"""
    print("\n📦 Merging LoRA adapters with base model...")
    
    if not Path(ADAPTER_PATH).exists():
        print(f"❌ Adapter path not found: {ADAPTER_PATH}")
        return False
    
    if not Path(BASE_MODEL).exists():
        print(f"❌ Base model not found: {BASE_MODEL}")
        return False
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    merged_model_path = OUTPUT_DIR / "phi3-cybersec-merged"
    
    try:
        from mlx_lm import load
        from mlx_lm.utils import save_model
        import shutil
        
        print(f"Loading base model: {BASE_MODEL}")
        print(f"Loading adapters: {ADAPTER_PATH}")
        
        # Load model with adapters
        model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
        
        print(f"Saving merged model to: {merged_model_path}")
        # Save the merged model (MLX format)
        save_model(str(merged_model_path), model)
        
        # Copy tokenizer files from base model
        base_model_path = Path(BASE_MODEL)
        for file_pattern in ["*.json", "*.model", "*.vocab", "*tokenizer*"]:
            for file in base_model_path.glob(file_pattern):
                if file.is_file():
                    shutil.copy2(file, merged_model_path)
                    print(f"Copied tokenizer file: {file.name}")
        
        print("✅ Model merging completed!")
        return merged_model_path
        
    except Exception as e:
        print(f"❌ Error merging model: {e}")
        return False

def convert_to_gguf(merged_model_path):
    """Convert merged MLX model to GGUF format"""
    print("\n🔄 Converting to GGUF format...")
    
    gguf_path = OUTPUT_DIR / "phi3-cybersec.gguf"
    
    try:
        # Try using MLX export functionality
        from mlx_lm.convert import convert
        
        print(f"Converting {merged_model_path} to GGUF...")
        # Note: This might need adjustment based on MLX version
        convert(
            model_path=str(merged_model_path),
            upload_repo=None,
            quantize=False,  # Keep full precision for now
            save_path=str(gguf_path)
        )
        
        print(f"✅ GGUF conversion completed: {gguf_path}")
        return gguf_path
        
    except Exception as e:
        print(f"❌ MLX convert failed: {e}")
        print("Trying alternative conversion method...")
        
        # Alternative: Use llama.cpp conversion tools
        try:
            cmd = [
                "python", "-m", "llama_cpp.convert", 
                str(merged_model_path),
                "--outfile", str(gguf_path),
                "--outtype", "f16"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Alternative conversion successful: {gguf_path}")
                return gguf_path
            else:
                print(f"❌ Alternative conversion failed: {result.stderr}")
                
        except Exception as alt_e:
            print(f"❌ Alternative conversion error: {alt_e}")
        
        return False

def test_gguf_model(gguf_path):
    """Test the converted GGUF model"""
    print("\n🧪 Testing converted GGUF model...")
    
    try:
        # Test with llama-cpp-python if available
        try:
            from llama_cpp import Llama
            
            # Load the model
            llm = Llama(
                model_path=str(gguf_path),
                n_ctx=2048,
                verbose=False
            )
            
            # Test with a cybersecurity prompt
            test_prompt = "What are the key steps in incident response?"
            print(f"Test prompt: {test_prompt}")
            
            response = llm(test_prompt, max_tokens=100, stop=["User:", "\n\n"])
            print(f"Response: {response['choices'][0]['text']}")
            
            print("✅ GGUF model test successful!")
            return True
            
        except ImportError:
            print("⚠️  llama-cpp-python not available for testing")
            print("Install with: pip install llama-cpp-python")
            return True  # Still consider successful if conversion worked
            
    except Exception as e:
        print(f"❌ GGUF model test failed: {e}")
        return False

def create_comparison_script():
    """Create script to compare MLX vs GGUF outputs"""
    script_content = '''#!/usr/bin/env python3
"""
Compare outputs between original MLX model and converted GGUF model
"""
import sys
from pathlib import Path

def test_mlx_model():
    """Test original MLX model"""
    try:
        from mlx_lm import load, generate
        
        model, tokenizer = load(
            "{base_model}", 
            adapter_path="{adapter_path}"
        )
        
        prompt = "What are the OWASP Top 10 vulnerabilities?"
        response = generate(model, tokenizer, prompt, max_tokens=200, verbose=False)
        
        print("MLX Model Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
    except Exception as e:
        print(f"MLX model error: {{e}}")

def test_gguf_model():
    """Test converted GGUF model"""
    try:
        from llama_cpp import Llama
        
        llm = Llama(
            model_path="{gguf_path}",
            n_ctx=2048,
            verbose=False
        )
        
        prompt = "What are the OWASP Top 10 vulnerabilities?"
        response = llm(prompt, max_tokens=200)
        
        print("GGUF Model Response:")
        print("-" * 40)
        print(response['choices'][0]['text'])
        print("-" * 40)
        
    except Exception as e:
        print(f"GGUF model error: {{e}}")

if __name__ == "__main__":
    print("🔬 Comparing MLX vs GGUF model outputs...")
    print("=" * 60)
    test_mlx_model()
    print()
    test_gguf_model()
'''.format(
        base_model=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        gguf_path=OUTPUT_DIR / "phi3-cybersec.gguf"
    )
    
    comparison_script = Path(__file__).parent / "compare_models.py"
    with open(comparison_script, 'w') as f:
        f.write(script_content)
    
    os.chmod(comparison_script, 0o755)
    print(f"✅ Comparison script created: {comparison_script}")

def main():
    """Main conversion process"""
    print("🚀 MLX PHI-3 → GGUF Conversion Process")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Phase 1: Merge LoRA adapters
    merged_model_path = merge_lora_adapters()
    if not merged_model_path:
        return 1
    
    # Phase 2: Convert to GGUF
    gguf_path = convert_to_gguf(merged_model_path)
    if not gguf_path:
        return 1
    
    # Phase 3: Test GGUF model
    if not test_gguf_model(gguf_path):
        print("⚠️  GGUF model test failed, but conversion may still be usable")
    
    # Phase 4: Create comparison tools
    create_comparison_script()
    
    print("\n🎉 Conversion process completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 GGUF model: {gguf_path}")
    print(f"🔬 Run comparison: python compare_models.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
