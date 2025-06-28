"""
Simple test to verify qwen2.5:3b model integration for code generation using AsyncClient.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    print("❌ Ollama not available. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False
    AsyncClient = None


async def test_qwen_code_generation():
    """Test direct code generation with qwen2.5:3b using AsyncClient."""
    print("🧪 Testing qwen2.5:3b code generation with AsyncClient...")
    
    if not OLLAMA_AVAILABLE or AsyncClient is None:
        print("❌ AsyncClient not available")
        return
    
    # Initialize async Ollama client
    client = AsyncClient()
    
    print("📋 Testing with qwen2.5:3b model")
    
    # Test simple code generation
    prompt = """Create a simple Python function that calculates the factorial of a number. 
    The function should handle edge cases and include documentation."""
    
    print("🚀 Generating code with qwen2.5:3b...")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    try:
        message = {'role': 'user', 'content': prompt}
        response = await client.chat(model='qwen2.5:3b', messages=[message])
        
        print("✅ Response received:")
        print(response.message.content)
        print("-" * 50)
        
        # Save to file
        content = response.message.content or "No content generated"
        with open('factorial_async.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("💾 Saved to: factorial_async.py")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")


async def test_simple_math_prompt():
    """Test with an even simpler prompt using AsyncClient."""
    print("\n🧪 Testing simple math function generation with AsyncClient...")
    
    if not OLLAMA_AVAILABLE or AsyncClient is None:
        print("❌ AsyncClient not available")
        return
    
    client = AsyncClient()
    
    prompt = "Write a Python function that adds two numbers."
    
    print(f"Prompt: {prompt}")
    print("-" * 30)
    
    try:
        message = {'role': 'user', 'content': prompt}
        response = await client.chat(model='qwen2.5:3b', messages=[message])
        
        print("Response:")
        print(response.message.content)
        
        # Save to file
        content = response.message.content or "No content generated"
        with open('add_function_async.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("💾 Saved to: add_function_async.py")
        
    except Exception as e:
        print(f"Error: {e}")


async def test_system_prompt_async():
    """Test with system prompt using AsyncClient."""
    print("\n🧪 Testing with System Prompt using AsyncClient...")
    
    if not OLLAMA_AVAILABLE or AsyncClient is None:
        print("❌ AsyncClient not available")
        return
    
    client = AsyncClient()
    
    messages = [
        {
            'role': 'system',
            'content': 'You are a Python expert. Write clean, documented code with type hints.'
        },
        {
            'role': 'user',
            'content': 'Create a function to calculate the nth Fibonacci number using memoization.'
        }
    ]
    
    print("📤 Sending request with system prompt...")
    
    try:
        response = await client.chat(model='qwen2.5:3b', messages=messages)
        
        print("✅ Response received:")
        print(response.message.content)
        
        # Save to file
        content = response.message.content or "No content generated"
        with open('fibonacci_memoized_async.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("💾 Saved to: fibonacci_memoized_async.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_qwen_code_generation())
    asyncio.run(test_simple_math_prompt())
    asyncio.run(test_system_prompt_async())
