#!/usr/bin/env python3
"""
Phase 3: Simplified Function Calling Implementation
Interactive approach that works better with PHI-3's natural language generation
"""
import sys
import json
import re
from pathlib import Path

# Import our tools
sys.path.append(str(Path(__file__).parent))
from enhanced_chat_with_tools import CyberSecurityTools

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_repetition_penalty, apply_min_p, apply_top_p
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Model paths
BASE_MODEL = "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx"
ADAPTER_PATH = "/Volumes/Crucial X9/ai-models/PHI-3.5-cybersec-finetune/adapters"

class InteractiveCyberSecChat:
    """Simplified interactive cybersecurity chat with tool integration"""
    
    def __init__(self):
        self.tools = CyberSecurityTools()
        self.model = None
        self.tokenizer = None
        self.conversation_history = ""
    
    def load_model(self):
        """Load the model"""
        if not MLX_AVAILABLE:
            print("❌ MLX-LM not available")
            return False
        
        try:
            print("Loading PHI-3 Cybersecurity model...")
            self.model, self.tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def enhanced_sampler(self, logits):
        """Enhanced sampler"""
        logits = apply_min_p(logits, min_p=0.05, min_tokens_to_keep=1)
        logits = apply_top_p(logits, top_p=0.9)
        temperature = 0.7
        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs))
    
    def detect_tool_request(self, user_input: str) -> tuple:
        """Detect if user is requesting a tool and which one"""
        input_lower = user_input.lower()
        
        # Web search detection
        if any(keyword in input_lower for keyword in ['search', 'cve', 'vulnerability', 'threat', 'intel']):
            if any(web_word in input_lower for web_word in ['web', 'online', 'internet', 'search']):
                # Extract search query
                query_patterns = [
                    r'search.*?(?:for|about)\s+(.+?)(?:\?|$)',
                    r'(?:find|look up)\s+(.+?)(?:\?|$)',
                    r'(?:cve|vulnerability)\s+(.+?)(?:\?|$)'
                ]
                for pattern in query_patterns:
                    match = re.search(pattern, input_lower)
                    if match:
                        return ("web_search", match.group(1).strip())
                return ("web_search", user_input)
        
        # File analysis detection
        if any(keyword in input_lower for keyword in ['hash', 'analyze file', 'check file']):
            file_pattern = r'(?:file|path)\s+["\']?([^"\'\\s]+)["\']?'
            match = re.search(file_pattern, user_input)
            if match:
                return ("file_hash", match.group(1))
        
        # IP analysis detection  
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ip_match = re.search(ip_pattern, user_input)
        if ip_match or 'ip' in input_lower:
            if ip_match:
                return ("ip_check", ip_match.group(0))
        
        # Directory listing detection
        if any(keyword in input_lower for keyword in ['list', 'directory', 'folder', 'ls', 'dir']):
            dir_pattern = r'(?:directory|folder|path)\s+["\']?([^"\'\\s]+)["\']?'
            match = re.search(dir_pattern, user_input)
            if match:
                return ("list_dir", match.group(1))
            elif any(word in input_lower for word in ['current', 'here', '.']):
                return ("list_dir", ".")
        
        # Log analysis detection
        if any(keyword in input_lower for keyword in ['log', 'scan log', 'analyze log']):
            log_pattern = r'(?:log|file)\s+["\']?([^"\'\\s]+)["\']?'
            match = re.search(log_pattern, user_input)
            if match:
                return ("scan_log", match.group(1))
        
        return (None, None)
    
    def execute_tool(self, tool_name: str, parameter: str) -> dict:
        """Execute a tool with the given parameter"""
        try:
            if tool_name == "web_search":
                return self.tools.web_search_threat_intel(parameter)
            elif tool_name == "file_hash":
                return self.tools.analyze_file_hash(parameter)
            elif tool_name == "ip_check":
                return self.tools.check_ip_reputation(parameter)
            elif tool_name == "list_dir":
                return self.tools.list_directory_securely(parameter)
            elif tool_name == "scan_log":
                return self.tools.scan_log_file(parameter)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}
    
    def generate_response(self, user_input: str, tool_result: dict = None) -> str:
        """Generate response with optional tool results"""
        system_prompt = """You are a cybersecurity expert assistant. You provide accurate, detailed cybersecurity advice and can analyze tool results when provided."""
        
        if tool_result:
            # Include tool results in the prompt
            prompt = f"""{system_prompt}

Previous conversation:
{self.conversation_history}

User: {user_input}

Tool Results:
{json.dumps(tool_result, indent=2)}

Assistant (analyze and explain the tool results in a helpful way):"""
        else:
            # Normal conversation
            if self.conversation_history:
                prompt = f"{system_prompt}\n\n{self.conversation_history}\nUser: {user_input}\n\nAssistant:"
            else:
                prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        
        try:
            repetition_penalty = make_repetition_penalty(penalty=1.05, context_size=64)
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=2048,  # Increased for comprehensive reports
                verbose=False,
                sampler=self.enhanced_sampler,
                logits_processors=[repetition_penalty]
            )
            
            # Clean response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            if "\nUser:" in response:
                response = response.split("\nUser:")[0].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}"
    
    def run_chat(self):
        """Main chat loop"""
        print("🤖 PHI-3 Cybersecurity Assistant with Intelligent Tool Integration")
        print("=" * 70)
        
        if not self.load_model():
            return 1
        
        print("\n🛠️  I can help you with:")
        print("  • Web search for threat intelligence and CVE information")
        print("  • File hash analysis and security scanning")
        print("  • IP address reputation checking")
        print("  • Secure directory listing and analysis")
        print("  • Log file scanning for security events")
        print("\nJust ask naturally - I'll detect when you need tools!")
        print("Commands: 'quit' to exit, 'clear' to reset")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n🔒 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Stay secure! 👨‍💻 Goodbye! 👋")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = ""
                    print("🔄 Conversation cleared!")
                    continue
                elif not user_input:
                    continue
                
                # Check if user needs a tool
                tool_name, parameter = self.detect_tool_request(user_input)
                
                if tool_name:
                    print(f"\n🛠️  I'll help you with that using my {tool_name} capability...")
                    tool_result = self.execute_tool(tool_name, parameter)
                    
                    print(f"\n📊 Tool Results:")
                    print(json.dumps(tool_result, indent=2))
                    
                    print(f"\n🤖 Assistant: ", end="", flush=True)
                    response = self.generate_response(user_input, tool_result)
                    print(response)
                else:
                    print(f"\n🤖 Assistant: ", end="", flush=True)
                    response = self.generate_response(user_input)
                    print(response)
                
                # Update conversation history
                self.conversation_history += f"User: {user_input}\n\nAssistant: {response}\n\n"
                
                # Manage history length
                if len(self.conversation_history) > 2000:
                    lines = self.conversation_history.split('\n')
                    self.conversation_history = '\n'.join(lines[-30:])
                
            except KeyboardInterrupt:
                print("\n⚡ Interrupted")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        return 0

def main():
    chat = InteractiveCyberSecChat()
    return chat.run_chat()

if __name__ == "__main__":
    sys.exit(main())
