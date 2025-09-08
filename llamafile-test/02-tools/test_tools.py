#!/usr/bin/env python3
"""
Test the cybersecurity tools directly
"""
import sys
import json
from pathlib import Path

# Add the current directory to path so we can import our module
sys.path.append(str(Path(__file__).parent))

from enhanced_chat_with_tools import CyberSecurityTools

def test_tools():
    """Test all cybersecurity tools"""
    print("🧪 Testing Cybersecurity Tools")
    print("=" * 40)
    
    tools = CyberSecurityTools()
    
    # Test 1: Web search
    print("\n1. Testing web search...")
    result = tools.web_search_threat_intel("CVE-2024")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 2: File hash (create a test file first)
    print("\n2. Testing file hash analysis...")
    test_file = Path("test_file.txt")
    test_file.write_text("This is a test file for cybersecurity analysis")
    
    result = tools.analyze_file_hash(str(test_file))
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Clean up
    test_file.unlink()
    
    # Test 3: IP reputation
    print("\n3. Testing IP reputation check...")
    result = tools.check_ip_reputation("192.168.1.1")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    result = tools.check_ip_reputation("8.8.8.8")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 4: Directory listing
    print("\n4. Testing secure directory listing...")
    result = tools.list_directory_securely(".")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\n✅ Tool testing completed!")

if __name__ == "__main__":
    test_tools()
