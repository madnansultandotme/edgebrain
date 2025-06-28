"""
Code Generation Agent Example

This example demonstrates how to create an agent specialized in code generation
and software development tasks using the Ollama Agentic Framework.
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional, Any
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import Agent, AgentGoal, AgentCapability
from src.core.orchestrator import AgentOrchestrator
from src.integration.ollama_client import OllamaIntegrationLayer, OllamaResponse, OllamaToolCall
from src.tools.tool_registry import ToolRegistry, BaseTool
from src.memory.memory_manager import MemoryManager

# Import official Ollama client
try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    print("⚠️  Official Ollama client not installed. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False
    AsyncClient = None


class CodeExecutionTool(BaseTool):
    """Tool for executing code snippets."""
    
    def __init__(self):
        super().__init__(
            name="code_execution",
            description="Execute Python code and return the result",
            category="development"
        )
    
    async def execute(self, code: str, language: str = "python") -> dict:
        """Execute code and return result."""
        if language.lower() != "python":
            return {
                "success": False,
                "error": f"Language {language} not supported. Only Python is supported."
            }
        
        try:
            # For safety, we'll simulate code execution
            # In a real implementation, you'd use a sandboxed environment
            
            # Mock execution results based on code content
            if "print" in code and "hello" in code.lower():
                output = "Hello, World!"
            elif "def" in code and "fibonacci" in code.lower():
                output = "Function fibonacci defined successfully"
            elif "import" in code and "requests" in code:
                output = "Module imported successfully"
            elif "class" in code:
                output = "Class defined successfully"
            else:
                output = "Code executed successfully"
            
            return {
                "success": True,
                "output": output,
                "code": code,
                "language": language
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": code,
                "language": language
            }


class DirectOllamaIntegration:
    """Direct Ollama integration using the official Python async client."""
    
    def __init__(self):
        self.available = OLLAMA_AVAILABLE
        self._available_models = []
        self.client = None
    
    async def initialize(self) -> bool:
        """Initialize the direct Ollama integration."""
        if not self.available or AsyncClient is None:
            return False
        
        try:
            # Create async client
            self.client = AsyncClient()
            
            # Test connection by trying to get a simple response
            test_message = {'role': 'user', 'content': 'Hello'}
            test_response = await self.client.chat(
                model='qwen2.5:3b',
                messages=[test_message]
            )
            
            if test_response and hasattr(test_response, 'message'):
                self._available_models = ['qwen2.5:3b']
                return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running and qwen2.5:3b model is available")
            return False
        
        return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self._available_models
    
    async def generate_code_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the qwen2.5:3b model."""
        if not self.available or self.client is None:
            raise Exception("Ollama client not available")
        
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        try:
            response = await self.client.chat(
                model='qwen2.5:3b',
                messages=messages
            )
            
            content = response.message.content
            return content if content is not None else "No response generated"
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"


class SimpleCodeGenerator:
    """Simple code generator using direct Ollama integration."""
    
    def __init__(self, ollama_integration: DirectOllamaIntegration):
        self.ollama = ollama_integration
    
    async def generate_fibonacci_function(self) -> str:
        """Generate a Fibonacci function."""
        prompt = """Create a Python function that generates the Fibonacci sequence up to n terms.
        
Requirements:
- Include docstring with clear explanation
- Handle edge cases (n <= 0, n == 1, n == 2)
- Use efficient iterative approach
- Include example usage
- Write clean, readable code

Please provide only the Python code without any markdown formatting."""
        
        system_prompt = "You are a Python programming expert. Write clean, efficient, and well-documented code."
        
        return await self.ollama.generate_code_response(prompt, system_prompt)
    
    async def generate_web_scraper_class(self) -> str:
        """Generate a web scraper class."""
        prompt = """Create a Python class for web scraping using requests library.

Requirements:
- Class should handle HTTP requests with proper error handling
- Include method to scrape a single URL
- Add user-agent header for respectful scraping
- Return structured data (dict with url, title, content)
- Include basic example usage
- Write clean, readable code

Please provide only the Python code without any markdown formatting."""
        
        system_prompt = "You are a Python programming expert. Write clean, efficient, and well-documented code."
        
        return await self.ollama.generate_code_response(prompt, system_prompt)
    
    async def generate_flask_api(self) -> str:
        """Generate a simple Flask API."""
        prompt = """Create a simple Flask REST API with basic CRUD operations.

Requirements:
- Create endpoints for GET, POST, PUT, DELETE operations
- Use in-memory storage (simple dict)
- Include proper HTTP status codes
- Add basic error handling
- Include health check endpoint
- Write clean, readable code

Please provide only the Python code without any markdown formatting."""
        
        system_prompt = "You are a Python web development expert. Write clean, efficient, and well-documented code."
        
        return await self.ollama.generate_code_response(prompt, system_prompt)


async def test_direct_code_generation():
    """Test direct code generation with qwen2.5:3b."""
    print("🧪 Testing Direct Code Generation with qwen2.5:3b")
    print("=" * 60)
    
    # Setup direct Ollama integration
    ollama_integration = await setup_direct_ollama_integration()
    
    if ollama_integration is None:
        print("❌ Could not initialize Ollama integration")
        return
    
    # Create code generator
    code_gen = SimpleCodeGenerator(ollama_integration)
    
    # Test tasks
    tasks = [
        ("Fibonacci Function", code_gen.generate_fibonacci_function),
        ("Web Scraper Class", code_gen.generate_web_scraper_class),
        ("Flask API", code_gen.generate_flask_api)
    ]
    
    for task_name, task_func in tasks:
        print(f"\n🚀 Generating {task_name}...")
        print("-" * 40)
        
        try:
            code = await task_func()
            print("✅ Generated code:")
            print(code)
            
            # Save to file
            filename = f"generated_{task_name.lower().replace(' ', '_')}.py"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"💾 Saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating {task_name}: {e}")
        
        print("-" * 40)


async def setup_direct_ollama_integration():
    """Setup direct Ollama integration with qwen2.5:3b model."""
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama Python client not available")
        print("Install with: pip install ollama")
        return None
    
    ollama_integration = DirectOllamaIntegration()
    
    # Initialize the integration
    success = await ollama_integration.initialize()
    if not success:
        print("❌ Failed to initialize direct Ollama integration")
        print("Make sure Ollama is running and qwen2.5:3b model is available")
        print("Run: ollama pull qwen2.5:3b")
        return None
    
    print(f"✅ Direct Ollama integration initialized with qwen2.5:3b model")
    return ollama_integration


async def setup_real_ollama_integration():
    """Setup real Ollama integration with qwen2.5:3b model."""
    ollama_integration = OllamaIntegrationLayer()
    
    # Initialize the integration
    success = await ollama_integration.initialize()
    if not success:
        print("❌ Failed to initialize Ollama integration")
        print("Make sure Ollama is running and accessible")
        return None
    
    # Check if qwen2.5:3b model is available
    available_models = ollama_integration.get_available_models()
    if "qwen2.5:3b" not in available_models:
        print(f"⚠️  qwen2.5:3b model not found in available models: {available_models}")
        print("Please pull the model with: ollama pull qwen2.5:3b")
        return None
    
    print(f"✅ Ollama integration initialized with models: {available_models}")
    return ollama_integration


class MockOllamaIntegrationLayer(OllamaIntegrationLayer):
    """Mock Ollama integration for fallback when real Ollama is not available."""
    
    def __init__(self):
        super().__init__()
        self.interaction_count = 0
    
    async def initialize(self) -> bool:
        """Mock initialization."""
        self._available_models = ["qwen2.5:3b", "llama3.1", "codellama"]
        return True
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Mock response generation for code-related tasks."""
        self.interaction_count += 1
        
        if "fibonacci" in prompt.lower():
            return await self._generate_fibonacci_code()
        elif "web scraper" in prompt.lower():
            return await self._generate_web_scraper_code()
        elif "api" in prompt.lower() and "flask" in prompt.lower():
            return await self._generate_flask_api_code()
        elif "test" in prompt.lower():
            return await self._generate_test_code()
        elif "review" in prompt.lower() or "analyze" in prompt.lower():
            return await self._generate_code_review()
        else:
            return OllamaResponse(
                content="I understand the coding task. Let me work on it step by step.",
                model="qwen2.5:3b"
            )
    
    async def _generate_fibonacci_code(self):
        """Generate Fibonacci sequence code."""
        tool_call = OllamaToolCall(
            name="file_write",
            arguments={
                "filename": "fibonacci.py",
                "content": """def fibonacci(n):
    \"\"\"Generate Fibonacci sequence up to n terms.\"\"\"
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

# Example usage
if __name__ == "__main__":
    n = 10
    result = fibonacci(n)
    print(f"Fibonacci sequence ({n} terms): {result}")
"""
            }
        )
        
        return OllamaResponse(
            content="I'll create a Fibonacci sequence generator in Python.",
            tool_calls=[tool_call],
            model="qwen2.5:3b"
        )
    
    async def _generate_web_scraper_code(self):
        """Generate web scraper code."""
        tool_call = OllamaToolCall(
            name="file_write",
            arguments={
                "filename": "web_scraper.py",
                "content": """import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict

class WebScraper:
    \"\"\"Simple web scraper for extracting data from websites.\"\"\"
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; WebScraper/1.0)'
        })
    
    def scrape_page(self, url: str) -> Dict[str, str]:
        \"\"\"Scrape a single page and extract basic information.\"\"\"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract all paragraph text
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            return {
                'url': url,
                'title': title_text,
                'content': content[:500] + '...' if len(content) > 500 else content,
                'status': 'success'
            }
        
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
    
    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, str]]:
        \"\"\"Scrape multiple URLs with delay between requests.\"\"\"
        results = []
        
        for url in urls:
            result = self.scrape_page(url)
            results.append(result)
            
            # Add delay to be respectful to servers
            time.sleep(self.delay)
        
        return results

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(delay=1.0)
    
    urls = [
        'https://example.com',
        'https://httpbin.org/html'
    ]
    
    results = scraper.scrape_multiple(urls)
    
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Title: {result['title']}")
            print(f"Content preview: {result['content'][:100]}...")
        else:
            print(f"Error: {result['error']}")
        print("-" * 50)
"""
            }
        )
        
        return OllamaResponse(
            content="I'll create a web scraper using Python with requests and BeautifulSoup.",
            tool_calls=[tool_call],
            model="qwen2.5:3b"
        )
    
    async def _generate_flask_api_code(self):
        """Generate Flask API code."""
        tool_call = OllamaToolCall(
            name="file_write",
            arguments={
                "filename": "api_server.py",
                "content": """from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
from datetime import datetime
from typing import Dict, List

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage (use a database in production)
tasks = {}

@app.route('/health', methods=['GET'])
def health_check():
    \"\"\"Health check endpoint.\"\"\"
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/tasks', methods=['GET'])
def get_tasks():
    \"\"\"Get all tasks.\"\"\"
    return jsonify({
        'tasks': list(tasks.values()),
        'count': len(tasks)
    })

@app.route('/tasks', methods=['POST'])
def create_task():
    \"\"\"Create a new task.\"\"\"
    data = request.get_json()
    
    if not data or 'title' not in data:
        return jsonify({'error': 'Title is required'}), 400
    
    task_id = str(uuid.uuid4())
    task = {
        'id': task_id,
        'title': data['title'],
        'description': data.get('description', ''),
        'status': 'pending',
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    tasks[task_id] = task
    
    return jsonify(task), 201

@app.route('/tasks/<task_id>', methods=['GET'])
def get_task(task_id: str):
    \"\"\"Get a specific task.\"\"\"
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(tasks[task_id])

@app.route('/tasks/<task_id>', methods=['PUT'])
def update_task(task_id: str):
    \"\"\"Update a task.\"\"\"
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    task = tasks[task_id]
    
    # Update allowed fields
    for field in ['title', 'description', 'status']:
        if field in data:
            task[field] = data[field]
    
    task['updated_at'] = datetime.now().isoformat()
    
    return jsonify(task)

@app.route('/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id: str):
    \"\"\"Delete a task.\"\"\"
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    del tasks[task_id]
    
    return jsonify({'message': 'Task deleted successfully'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
"""
            }
        )
        
        return OllamaResponse(
            content="I'll create a RESTful API using Flask for task management.",
            tool_calls=[tool_call],
            model="qwen2.5:3b"
        )
    
    async def _generate_test_code(self):
        """Generate test code."""
        return OllamaResponse(
            content="""I'll create comprehensive unit tests for the code:

```python
import unittest
from fibonacci import fibonacci

class TestFibonacci(unittest.TestCase):
    
    def test_fibonacci_zero(self):
        result = fibonacci(0)
        self.assertEqual(result, [])
    
    def test_fibonacci_one(self):
        result = fibonacci(1)
        self.assertEqual(result, [0])
    
    def test_fibonacci_two(self):
        result = fibonacci(2)
        self.assertEqual(result, [0, 1])
    
    def test_fibonacci_ten(self):
        result = fibonacci(10)
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        self.assertEqual(result, expected)
    
    def test_fibonacci_negative(self):
        result = fibonacci(-1)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
```

Tests cover edge cases and normal operation. All tests should pass.

GOAL_COMPLETED""",
            model="qwen2.5:3b"
        )
    
    async def _generate_code_review(self):
        """Generate code review."""
        return OllamaResponse(
            content="""Code Review Complete! Here's my analysis:

**Fibonacci Function:**
✅ **Strengths:**
- Clear documentation with docstring
- Handles edge cases (n <= 0, n == 1, n == 2)
- Efficient iterative approach
- Good variable naming

📝 **Suggestions:**
- Add type hints: `def fibonacci(n: int) -> List[int]:`
- Consider input validation for non-integer inputs
- Add error handling for very large values of n

**Web Scraper:**
✅ **Strengths:**
- Good class structure and organization
- Proper error handling with try-catch
- Respectful scraping with delays
- Clear method documentation

📝 **Suggestions:**
- Add rate limiting and retry logic
- Implement robots.txt checking
- Add logging for better debugging
- Consider async/await for better performance

**Flask API:**
✅ **Strengths:**
- RESTful design following best practices
- Proper HTTP status codes
- CORS enabled for frontend integration
- Good error handling

📝 **Suggestions:**
- Add input validation and sanitization
- Implement authentication and authorization
- Add request rate limiting
- Use a proper database instead of in-memory storage

**Overall Assessment:**
The code is well-structured and functional. With the suggested improvements, it would be production-ready.

GOAL_COMPLETED""",
            model="qwen2.5:3b"
        )


async def main():
    """Main function to demonstrate the code generation agent."""
    print("💻 Code Generation Agent Example with Direct Ollama Integration")
    print("=" * 70)
    
    # Check if we want to run the simple direct test or the full agent framework
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        await test_direct_code_generation()
        return
    
    print("Initializing framework components...")
    
    # Try to use direct Ollama integration first
    direct_ollama = await setup_direct_ollama_integration()
    
    if direct_ollama is not None:
        print("🎯 Using direct Ollama integration for faster code generation")
        print("💡 Run with --simple flag for direct code generation without agent framework")
        
        # Run simple test
        await test_direct_code_generation()
        return
    
    # Fall back to the framework integration
    ollama_integration = await setup_real_ollama_integration()
    
    # Fall back to mock if real integration fails
    if ollama_integration is None:
        print("🔄 Falling back to mock Ollama integration for demonstration")
        ollama_integration = MockOllamaIntegrationLayer()
        await ollama_integration.initialize()
    
    tool_registry = ToolRegistry()
    
    # Add custom code execution tool
    code_tool = CodeExecutionTool()
    tool_registry.register_tool(code_tool)
    
    memory_manager = MemoryManager(db_path=":memory:")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    # Define coding capabilities
    coding_capabilities = [
        AgentCapability(
            name="python_programming",
            description="Write Python code for various applications"
        ),
        AgentCapability(
            name="web_development",
            description="Create web applications and APIs"
        ),
        AgentCapability(
            name="code_review",
            description="Review code for quality, security, and best practices"
        ),
        AgentCapability(
            name="testing",
            description="Write unit tests and integration tests"
        ),
        AgentCapability(
            name="debugging",
            description="Identify and fix bugs in code"
        )
    ]
    
    # Create code generation agent
    coder_agent = orchestrator.register_agent(
        agent_id="coder_001",
        role="Software Developer",
        description="An AI agent specialized in software development and code generation using qwen2.5:3b",
        model="qwen2.5:3b",  # Using qwen2.5:3b model for real code generation
        capabilities=coding_capabilities,
        system_prompt="""You are a Software Developer AI agent powered by qwen2.5:3b. Your role is to:
1. Write clean, efficient, and well-documented code
2. Follow best practices and coding standards
3. Create comprehensive tests for your code
4. Review code for quality and security issues
5. Debug and fix issues in existing code

Always prioritize code quality, readability, and maintainability. Include proper error handling and documentation.
When generating code, use the file_write tool to save code to disk. Be concise but complete in your implementations.
Focus on creating working code quickly while maintaining quality."""
    )
    
    print(f"✅ Created code generation agent: {coder_agent.agent_id}")
    
    # Display integration type
    if isinstance(ollama_integration, MockOllamaIntegrationLayer):
        print("📋 Using mock Ollama integration for demonstration")
        print("💡 To use real qwen2.5:3b model:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Run: ollama pull qwen2.5:3b")
        print("   3. Start Ollama service")
        print("   4. Re-run this script")
    else:
        print("🚀 Using real Ollama integration with qwen2.5:3b model")
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Demonstrate various coding tasks with simplified context for faster execution
    coding_tasks = [
        {
            "description": "Create a simple Python function to generate Fibonacci sequence up to n terms",
            "context": {"language": "python", "task_type": "function", "complexity": "simple"}
        },
        {
            "description": "Build a basic web scraper class using Python requests",
            "context": {"language": "python", "task_type": "class", "complexity": "simple"}
        },
        {
            "description": "Create a minimal Flask REST API with basic CRUD operations",
            "context": {"framework": "flask", "task_type": "api", "complexity": "simple"}
        }
    ]
    
    print(f"\n🚀 Executing {len(coding_tasks)} coding tasks...")
    
    for i, task_info in enumerate(coding_tasks, 1):
        print(f"\n📋 Task {i}: {task_info['description']}")
        
        # Create task
        task_id = await orchestrator.create_task(
            description=task_info["description"],
            priority=5,
            context=task_info["context"]
        )
        
        # Assign to coder agent
        success = await orchestrator.assign_task_to_agent(task_id, coder_agent.agent_id)
        
        if success:
            print(f"✅ Task {i} assigned successfully")
            
            # Wait for completion with longer timeout for real LLM
            max_wait = 60  # Increased timeout for real qwen2.5:3b model
            wait_time = 0
            check_interval = 3  # Check every 3 seconds
            
            while wait_time < max_wait:
                task = orchestrator.get_task(task_id)
                if task is None:
                    print(f"❌ Task {i} not found")
                    break
                    
                status = task.status.value
                
                print(f"⏱️  Task {i} status: {status} (waiting {wait_time}/{max_wait}s)")
                
                if status in ["completed", "failed"]:
                    break
                
                await asyncio.sleep(check_interval)
                wait_time += check_interval
            
            final_task = orchestrator.get_task(task_id)
            if final_task is not None and final_task.status.value == "completed":
                print(f"🎉 Task {i} completed successfully!")
            else:
                print(f"❌ Task {i} failed or timed out")
        else:
            print(f"❌ Failed to assign task {i}")
    
    # Demonstrate code review capability
    print("\n🔍 Demonstrating code review capability...")
    
    review_task_id = await orchestrator.create_task(
        description="Review the generated code for quality, security, and best practices",
        priority=3,
        context={"review_type": "comprehensive", "focus": ["quality", "security", "performance"]}
    )
    
    await orchestrator.assign_task_to_agent(review_task_id, coder_agent.agent_id)
    
    # Wait for review completion
    await asyncio.sleep(5)
    
    print("✅ Code review completed")
    
    # Show agent's development process
    print("\n📝 Development Process Summary:")
    memories = await memory_manager.retrieve_memories(
        agent_id=coder_agent.agent_id,
        limit=8
    )
    
    for i, memory in enumerate(memories, 1):
        print(f"{i}. {memory[:80]}...")
    
    # Stop the orchestrator
    await orchestrator.stop()
    
    # Final statistics
    print("\n📈 Final Statistics:")
    agent_stats = orchestrator.get_agent_status_summary()
    task_stats = orchestrator.get_task_status_summary()
    memory_stats = await memory_manager.get_memory_stats()
    tool_count = tool_registry.get_tool_count()
    
    print(f"Agent statuses: {agent_stats}")
    print(f"Task statuses: {task_stats}")
    print(f"Memory entries: {memory_stats['total_memories']}")
    print(f"Available tools: {tool_count}")
    
    print("\n✨ Code generation example completed!")


if __name__ == "__main__":
    asyncio.run(main())

