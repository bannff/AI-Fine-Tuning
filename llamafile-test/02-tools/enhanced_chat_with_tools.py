#!/usr/bin/env python3
"""
Phase 2: Function Calling Implementation
Enhanced PHI-3 cybersecurity chat with built-in web search and filesystem tools

This creates a version of your chat interface with cybersecurity-focused tools:
- Web search for threat intelligence and CVE lookups
- Secure filesystem access for log analysis
- Hash analysis and reputation checking
- Security-focused utilities
"""
import os
import sys
import json
import hashlib
import ipaddress
from pathlib import Path
from typing import Dict, List, Any, Optional
import urllib.parse
import re

# Web search and HTTP tools
try:
    import requests
    from bs4 import BeautifulSoup
    import validators
    WEB_TOOLS_AVAILABLE = True
except ImportError:
    WEB_TOOLS_AVAILABLE = False
    print("⚠️  Web tools not available. Install: pip3 install requests beautifulsoup4 validators")

# MLX-LM for the model
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty, apply_min_p, apply_top_p
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("❌ MLX-LM not available. Install with: pip install mlx-lm")

# Model paths (using the adapter approach that works)
BASE_MODEL = "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx"
ADAPTER_PATH = "/Volumes/Crucial X9/ai-models/PHI-3.5-cybersec-finetune/adapters"

class CyberSecurityTools:
    """Cybersecurity-focused tool implementations"""
    
    def __init__(self):
        self.session = requests.Session() if WEB_TOOLS_AVAILABLE else None
        if self.session:
            # Set a proper user agent for web requests
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) CybersecAI/1.0'
            })
    
    def web_search_threat_intel(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for cybersecurity threat intelligence"""
        if not WEB_TOOLS_AVAILABLE:
            return {"error": "Web tools not available"}
        
        try:
            # Focus on reputable cybersecurity sources
            cybersec_sources = [
                "site:cve.mitre.org",
                "site:nvd.nist.gov", 
                "site:us-cert.cisa.gov",
                "site:krebsonsecurity.com",
                "site:threatpost.com",
                "site:bleepingcomputer.com"
            ]
            
            # Create search query focused on cybersecurity
            search_query = f"{query} ({' OR '.join(cybersec_sources[:3])})"
            
            # Simple DuckDuckGo search (no API key required)
            search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(search_query)}"
            
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            for result in soup.find_all('a', class_='result__a')[:max_results]:
                title = result.get_text().strip()
                url = result.get('href', '')
                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "source": "DuckDuckGo"
                    })
            
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def analyze_file_hash(self, file_path: str) -> Dict[str, Any]:
        """Analyze file hash and basic properties"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"error": f"Path is not a file: {file_path}"}
            
            # Calculate hashes
            with open(path, 'rb') as f:
                content = f.read()
                
            hashes = {
                'md5': hashlib.md5(content).hexdigest(),
                'sha1': hashlib.sha1(content).hexdigest(),
                'sha256': hashlib.sha256(content).hexdigest()
            }
            
            # Basic file info
            stat = path.stat()
            
            return {
                "file_path": str(path),
                "file_size": stat.st_size,
                "hashes": hashes,
                "last_modified": stat.st_mtime,
                "is_executable": os.access(path, os.X_OK)
            }
            
        except Exception as e:
            return {"error": f"Hash analysis failed: {str(e)}"}
    
    def scan_log_file(self, log_path: str, security_keywords: List[str] = None) -> Dict[str, Any]:
        """Scan log file for security-related events"""
        if security_keywords is None:
            security_keywords = [
                'failed login', 'authentication failed', 'invalid user',
                'ssh', 'brute force', 'attack', 'malware', 'virus',
                'intrusion', 'unauthorized', 'blocked', 'denied',
                'exploit', 'vulnerability', 'suspicious'
            ]
        
        try:
            path = Path(log_path)
            if not path.exists():
                return {"error": f"Log file not found: {log_path}"}
            
            findings = []
            line_count = 0
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line_count = line_num
                    line_lower = line.lower().strip()
                    
                    for keyword in security_keywords:
                        if keyword.lower() in line_lower:
                            findings.append({
                                "line_number": line_num,
                                "keyword": keyword,
                                "content": line.strip()[:200]  # Limit line length
                            })
                            break  # Only one finding per line
            
            return {
                "log_file": str(path),
                "total_lines": line_count,
                "security_events": len(findings),
                "findings": findings[:50]  # Limit results
            }
            
        except Exception as e:
            return {"error": f"Log scan failed: {str(e)}"}
    
    def check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Basic IP address analysis"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            result = {
                "ip_address": str(ip),
                "is_private": ip.is_private,
                "is_loopback": ip.is_loopback,
                "is_multicast": ip.is_multicast,
                "version": ip.version
            }
            
            # Add classification
            if ip.is_private:
                result["classification"] = "Private/Internal"
            elif ip.is_loopback:
                result["classification"] = "Loopback"
            elif ip.is_multicast:
                result["classification"] = "Multicast"
            else:
                result["classification"] = "Public"
            
            return result
            
        except ValueError:
            return {"error": f"Invalid IP address: {ip_address}"}
        except Exception as e:
            return {"error": f"IP analysis failed: {str(e)}"}
    
    def list_directory_securely(self, directory_path: str, max_items: int = 20) -> Dict[str, Any]:
        """Securely list directory contents with security focus"""
        try:
            path = Path(directory_path)
            if not path.exists():
                return {"error": f"Directory not found: {directory_path}"}
            
            if not path.is_dir():
                return {"error": f"Path is not a directory: {directory_path}"}
            
            items = []
            security_notes = []
            
            for item in sorted(path.iterdir())[:max_items]:
                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "permissions": oct(stat.st_mode)[-3:],
                        "is_hidden": item.name.startswith('.'),
                        "is_executable": os.access(item, os.X_OK)
                    }
                    
                    # Check for suspicious file extensions
                    if item.is_file():
                        suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js']
                        if any(item.name.lower().endswith(ext) for ext in suspicious_extensions):
                            security_notes.append(f"Potentially suspicious file: {item.name}")
                    
                    items.append(item_info)
                    
                except (OSError, PermissionError):
                    # Skip items we can't access
                    continue
            
            return {
                "directory": str(path),
                "item_count": len(items),
                "items": items,
                "security_notes": security_notes
            }
            
        except Exception as e:
            return {"error": f"Directory listing failed: {str(e)}"}

class FunctionCallingChat:
    """Enhanced chat interface with function calling capabilities"""
    
    def __init__(self):
        self.tools = CyberSecurityTools()
        self.model = None
        self.tokenizer = None
        self.conversation_history = ""
        
        # Available functions
        self.functions = {
            "web_search_threat_intel": {
                "function": self.tools.web_search_threat_intel,
                "description": "Search for cybersecurity threat intelligence and CVE information",
                "parameters": {
                    "query": "Search query for threat intelligence",
                    "max_results": "Maximum number of results to return (default: 5)"
                }
            },
            "analyze_file_hash": {
                "function": self.tools.analyze_file_hash,
                "description": "Calculate file hashes (MD5, SHA1, SHA256) and basic file analysis",
                "parameters": {
                    "file_path": "Path to the file to analyze"
                }
            },
            "scan_log_file": {
                "function": self.tools.scan_log_file,
                "description": "Scan log files for security-related events and keywords",
                "parameters": {
                    "log_path": "Path to the log file to scan",
                    "security_keywords": "Optional list of keywords to search for"
                }
            },
            "check_ip_reputation": {
                "function": self.tools.check_ip_reputation,
                "description": "Analyze IP address and classify as private/public/suspicious",
                "parameters": {
                    "ip_address": "IP address to analyze"
                }
            },
            "list_directory_securely": {
                "function": self.tools.list_directory_securely,
                "description": "Securely list directory contents with security-focused analysis",
                "parameters": {
                    "directory_path": "Path to directory to list",
                    "max_items": "Maximum number of items to return (default: 20)"
                }
            }
        }
    
    def load_model(self):
        """Load the cybersecurity model"""
        if not MLX_AVAILABLE:
            print("❌ MLX-LM not available")
            return False
        
        try:
            print(f"Loading base model with adapters...")
            print(f"Base: {BASE_MODEL}")
            print(f"Adapters: {ADAPTER_PATH}")
            
            self.model, self.tokenizer = load(
                BASE_MODEL, 
                adapter_path=ADAPTER_PATH
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def parse_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse function calls from model output"""
        # Look for function call pattern: FUNCTION_NAME(param1="value1", param2="value2")
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, text)
        
        for func_name, params_str in matches:
            if func_name in self.functions:
                try:
                    # Simple parameter parsing
                    params = {}
                    if params_str.strip():
                        # Parse key="value" pairs
                        param_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
                        param_matches = re.findall(param_pattern, params_str)
                        for key, value in param_matches:
                            params[key] = value
                    
                    return {
                        "function": func_name,
                        "parameters": params
                    }
                except Exception as e:
                    print(f"Error parsing function call: {e}")
        
        return None
    
    def execute_function(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        func_name = function_call["function"]
        params = function_call["parameters"]
        
        if func_name not in self.functions:
            return {"error": f"Unknown function: {func_name}"}
        
        try:
            func = self.functions[func_name]["function"]
            result = func(**params)
            return {
                "function": func_name,
                "parameters": params,
                "result": result
            }
        except Exception as e:
            return {
                "function": func_name,
                "parameters": params,
                "error": str(e)
            }
    
    def enhanced_sampler(self, logits):
        """Enhanced sampler with Min-P, Top-P fallback, and temperature"""
        # Apply Min-P filtering first (adaptive)
        logits = apply_min_p(logits, min_p=0.05, min_tokens_to_keep=1)
        
        # Apply Top-P as fallback
        logits = apply_top_p(logits, top_p=0.9)
        
        # Apply temperature scaling
        temperature = 0.7
        logits = logits / temperature
        
        # Sample from the distribution
        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs))
    
    def generate_response(self, user_input: str) -> str:
        """Generate response with potential function calling"""
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded"
        
        # Build enhanced prompt with function calling context
        function_descriptions = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.functions.items()
        ])
        
        system_prompt = f"""You are a cybersecurity expert assistant with access to the following tools:

{function_descriptions}

To use a tool, write: FUNCTION_NAME(param1="value1", param2="value2")

Always explain what you're doing and interpret the results for the user.
Provide actionable cybersecurity advice based on the tool results."""
        
        if self.conversation_history:
            prompt = f"{system_prompt}\n\n{self.conversation_history}\nUser: {user_input}\n\nAssistant:"
        else:
            prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        
        try:
            # Generate response
            repetition_penalty = make_repetition_penalty(penalty=1.05, context_size=64)
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=512,
                verbose=False,
                sampler=self.enhanced_sampler,
                logits_processors=[repetition_penalty]
            )
            
            # Extract assistant response
            if "Assistant:" in response:
                assistant_response = response.split("Assistant:")[-1].strip()
            else:
                assistant_response = response.strip()
            
            # Clean up response
            if "\nUser:" in assistant_response:
                assistant_response = assistant_response.split("\nUser:")[0].strip()
            
            return assistant_response
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}"
    
    def run_chat(self):
        """Main chat loop with function calling"""
        print("🤖 PHI-3 Cybersecurity Assistant with Built-in Tools")
        print("=" * 60)
        
        if not self.load_model():
            return 1
        
        print("\n🛠️  Available Tools:")
        for name, info in self.functions.items():
            print(f"  • {name}: {info['description']}")
        
        print(f"\n✅ Ready! Web tools: {'Available' if WEB_TOOLS_AVAILABLE else 'Not available'}")
        print("Commands: 'quit' to exit, 'clear' to reset, 'tools' to list tools")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n🔒 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Stay secure! 👨‍💻 Goodbye! 👋")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = ""
                    print("🔄 Conversation cleared - Fresh start!")
                    continue
                elif user_input.lower() == 'tools':
                    print("\n🛠️  Available Tools:")
                    for name, info in self.functions.items():
                        params = ", ".join(info['parameters'].keys())
                        print(f"  • {name}({params}): {info['description']}")
                    continue
                elif not user_input:
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                
                # Generate initial response
                response = self.generate_response(user_input)
                
                # Check for function calls in the response
                function_call = self.parse_function_call(response)
                if function_call:
                    print(f"[Calling {function_call['function']}...]")
                    
                    # Execute the function
                    result = self.execute_function(function_call)
                    
                    # Generate follow-up response with results
                    result_prompt = f"User asked: {user_input}\n\nI used {function_call['function']} and got these results:\n{json.dumps(result, indent=2)}\n\nNow I'll interpret these results:"
                    interpretation = self.generate_response(result_prompt)
                    
                    print(f"\n\n📊 Tool Results:\n{json.dumps(result.get('result', result), indent=2)}")
                    print(f"\n📝 Analysis: {interpretation}")
                else:
                    print(response)
                
                # Update conversation history
                self.conversation_history += f"User: {user_input}\n\nAssistant: {response}\n\n"
                
                # Manage history length
                if len(self.conversation_history) > 3000:
                    lines = self.conversation_history.split('\n')
                    self.conversation_history = '\n'.join(lines[-50:])
                
            except KeyboardInterrupt:
                print("\n⚡ Generation interrupted")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        return 0

def main():
    chat = FunctionCallingChat()
    return chat.run_chat()

if __name__ == "__main__":
    sys.exit(main())
