# Installation and Setup Guide

This comprehensive guide will walk you through the installation and setup process for EdgeBrain, an open-source agentic framework for building AI agents with Ollama-based models. The framework supports various agent types including research agents, code generation agents, data analysis agents, and more.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: Version 3.8 or higher (3.11+ recommended)
- **Memory**: 8GB RAM minimum (16GB recommended for multiple agents)
- **Storage**: 5GB free space for the framework, dependencies, and models
- **Network**: Internet connection for downloading models and dependencies

### Recommended Requirements for Production Use
- **Operating System**: Linux (Ubuntu 22.04+) or macOS (12.0+)
- **Python**: Version 3.11 or 3.12
- **Memory**: 16GB RAM or higher (for concurrent agents and larger models)
- **Storage**: 20GB+ free space for multiple models and agent data
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster model inference)
- **CPU**: Multi-core processor recommended for concurrent agent operations

## Prerequisites

### 1. Python Installation

Ensure you have Python 3.11 or higher installed on your system.

**Check your Python version:**
```bash
python3 --version
```

**Install Python 3.11+ if needed:**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv python3.11-dev
```

**macOS (using Homebrew):**
```bash
brew install python@3.11
```

**Windows:**
Download and install from [python.org](https://www.python.org/downloads/)

### 2. Ollama Installation

The framework requires Ollama to be installed and running on your system for both the custom integration and async client usage.

**Install Ollama:**

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from [ollama.ai](https://ollama.ai/download)

**Verify Ollama installation:**
```bash
ollama --version
```

**Start Ollama service:**
```bash
ollama serve
```

The Ollama service should now be running on `http://localhost:11434`

### 3. Install Required Models

For optimal performance with the framework's code generation capabilities, install the recommended models:

**Essential Models for Different Agent Types:**

```bash
# For research and general-purpose agents
ollama pull llama3.1      # General reasoning and analysis

# For code generation agents  
ollama pull qwen2.5:3b    # Fast, efficient code generation
ollama pull codellama     # Specialized code generation

# For data analysis agents
ollama pull mistral       # Advanced analytical reasoning

# Lightweight model for testing
ollama pull phi3          # Fast prototyping and testing
```

**Model Recommendations by Use Case:**

| Agent Type | Primary Model | Alternative | Use Case |
|------------|---------------|-------------|----------|
| Research Agent | `llama3.1` | `mistral` | Web search, analysis, reports |
| Code Agent | `qwen2.5:3b` | `codellama` | Code generation, debugging |
| Data Agent | `mistral` | `llama3.1` | Data analysis, visualization |
| Communication Agent | `llama3.1` | `phi3` | Message routing, coordination |
| Planning Agent | `mistral` | `llama3.1` | Strategy, resource allocation |

**Verify models are installed:**
```bash
ollama list
```

You should see:
```
NAME            ID              SIZE    MODIFIED
qwen2.5:3b      abc123def456    2.3GB   2 minutes ago
llama3.1        def456ghi789    4.7GB   3 minutes ago
```

### 4. Git Installation

Git is required for cloning the repository and version control.

**Install Git:**

**Ubuntu/Debian:**
```bash
sudo apt install git
```

**macOS:**
```bash
brew install git
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/download/win)

## Framework Installation

### Method 1: Installation from PyPI (Recommended)

The EdgeBrain framework is available on PyPI for easy installation and use in your projects.

**1. Create a new project directory:**
```bash
mkdir my-edgebrain-app
cd my-edgebrain-app
```

**2. Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Upgrade pip:**
```bash
pip install --upgrade pip
```

**4. Install EdgeBrain from PyPI:**
```bash
pip install edgebrain
```

**5. Install the official Ollama async client:**
```bash
pip install ollama
```

### Method 2: Installation from Source (Development)

For developers who want to modify the framework or access the latest unreleased features:

**1. Clone the repository:**
```bash
git clone https://github.com/madnansultandotme/ollama-agentic-framework.git
cd ollama-agentic-framework
```

**2. Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Upgrade pip:**
```bash
pip install --upgrade pip
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt

# Install the official Ollama Python client for async support
pip install ollama
```

**5. Install the framework in development mode:**
```bash
pip install -e .
```

## Initial Configuration

### 1. Verify Ollama Models

Ensure the required models are properly installed and accessible:

```bash
# Test model availability
ollama list

# Test qwen2.5:3b specifically (required for code generation)
ollama run qwen2.5:3b "Hello, write a simple Python function"
```

**If you see models listed and get a response, you're ready to proceed.**

### 2. Test Async Ollama Integration

Create a test script to verify the async integration works:

```python
# test_setup.py
import asyncio
from ollama import AsyncClient

async def test_async_ollama():
    try:
        client = AsyncClient()
        
        message = {'role': 'user', 'content': 'Hello from async client!'}
        response = await client.chat(model='qwen2.5:3b', messages=[message])
        
        print("✅ Async Ollama integration working!")
        print(f"Response: {response.message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Async integration test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_async_ollama())
```

Run the test:
```bash
python test_setup.py
```

### 3. Environment Configuration

Create a `.env` file in the project root to configure environment variables:

```bash
# .env file
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.1
CODE_GENERATION_MODEL=qwen2.5:3b
MEMORY_DB_PATH=agent_memory.db
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=5
TOOL_TIMEOUT=30
ASYNC_CLIENT_TIMEOUT=60
```

### 3. Database Setup

The framework uses SQLite for memory storage. The database will be created automatically on first run, but you can initialize it manually:

```python
from src.memory.memory_manager import MemoryManager

# Initialize memory manager (creates database if it doesn't exist)
memory_manager = MemoryManager(db_path="agent_memory.db")
```

## Verification

### 1. Verify EdgeBrain Installation

Test your EdgeBrain installation with this simple verification script:

**`verify_installation.py`:**
```python
import asyncio
import ollama

async def verify_edgebrain():
    """Verify EdgeBrain and Ollama async client installation."""
    
    print("🔍 Verifying EdgeBrain installation...")
    
    # Test EdgeBrain imports
    try:
        from edgebrain.core.agent import Agent
        from edgebrain.integration.ollama_client import OllamaIntegrationLayer
        print("✅ EdgeBrain core modules imported successfully")
    except ImportError as e:
        print(f"❌ EdgeBrain import failed: {e}")
        return False
    
    # Test Ollama async client
    try:
        client = ollama.AsyncClient()
        models = await client.list()
        print(f"✅ Ollama async client working. Available models: {len(models.get('models', []))}")
        
        # Test model availability
        model_names = [model['name'] for model in models.get('models', [])]
        if 'qwen2.5:3b' in model_names:
            print("✅ qwen2.5:3b model available for code generation")
        else:
            print("⚠️  qwen2.5:3b model not found. Install with: ollama pull qwen2.5:3b")
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False
    
    print("🎉 Installation verification complete!")
    return True

if __name__ == "__main__":
    asyncio.run(verify_edgebrain())
```

Run the verification:
```bash
python verify_installation.py
```

### 2. Test Async Code Generation

Create a quick test to verify code generation works:

**`test_code_generation.py`:**
```python
import asyncio
import ollama

async def test_code_generation():
    """Test basic code generation functionality."""
    
    client = ollama.AsyncClient()
    
    prompt = "Write a simple Python function that adds two numbers"
    
    try:
        response = await client.chat(
            model="qwen2.5:3b",
            messages=[
                {"role": "system", "content": "You are a Python expert. Generate clean, working code."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response and 'message' in response:
            code = response['message']['content']
            print("Generated Code:")
            print("-" * 40)
            print(code)
            print("-" * 40)
            print("✅ Code generation test passed!")
        else:
            print("❌ No response from model")
            
    except Exception as e:
        print(f"❌ Code generation test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_code_generation())
```

### 3. Run Development Tests (Source Installation Only)

If you installed from source, you can run the test suite:

Test the new async code generation capabilities:

```bash
# Test direct async Ollama integration
python examples/test_async_ollama.py

# Test code generation agent (simple mode)
python examples/code_generation_agent.py --simple

# Test updated simple code test
python examples/simple_code_test.py
```

Expected output for successful async code generation:
```
🧪 Testing Direct Code Generation with qwen2.5:3b
============================================================
✅ Direct Ollama integration initialized with qwen2.5:3b model
🚀 Generating Fibonacci Function...
✅ Generated code:
def fibonacci(n):
    """Generate Fibonacci sequence up to n terms."""
    ...
💾 Saved to: generated_fibonacci_function.py
```

### 3. Run Example Scripts

Test the framework with the provided examples:

```bash
# Enhanced research agent with real tools
python examples/enhanced_research_agent.py

# Simple research agent with custom topic
python examples/simple_research_agent.py "artificial intelligence trends"

# Multi-agent collaboration
python examples/multi_agent_collaboration.py

# Comprehensive demo (requires all components)
python examples/comprehensive_demo.py
```

### 4. Check Component Status

Verify all components are working correctly including async integration:

```python
import asyncio
from ollama import AsyncClient
from src.integration.ollama_client import OllamaIntegrationLayer
from src.tools.tool_registry import ToolRegistry
from src.memory.memory_manager import MemoryManager

async def check_components():
    print("🔍 Checking EdgeBrain Framework Components...")
    
    # Test direct async Ollama client
    try:
        client = AsyncClient()
        message = {'role': 'user', 'content': 'Hello'}
        response = await client.chat(model='qwen2.5:3b', messages=[message])
        print("✅ Direct AsyncClient working")
        print(f"🎯 qwen2.5:3b model responsive")
    except Exception as e:
        print(f"❌ AsyncClient failed: {e}")
    
    # Test custom Ollama integration
    ollama = OllamaIntegrationLayer()
    if await ollama.initialize():
        print("✅ Custom Ollama integration working")
        models = ollama.get_available_models()
        print(f"📦 Available models: {models}")
    else:
        print("❌ Custom Ollama integration failed")
    
    # Test tool registry
    tools = ToolRegistry()
    tool_count = tools.get_tool_count()
    print(f"🔧 Available tools: {tool_count}")
    
    # List some tools
    tool_names = []
    for category in ['research', 'file', 'web', 'analysis']:
        category_tools = tools.get_tools_by_category(category)
        tool_names.extend([tool.name for tool in category_tools])
    print(f"📋 Tool examples: {', '.join(tool_names[:5])}")
    
    # Test memory manager
    memory = MemoryManager()
    print("🧠 Memory manager initialized")
    
    print("\n✅ All components verified!")

# Run the check
asyncio.run(check_components())
```

Save this as `verify_installation.py` and run:
```bash
python verify_installation.py
```

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed

**Problem**: Cannot connect to Ollama service
**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve

# Check firewall settings
sudo ufw allow 11434
```

#### 2. Python Version Issues

**Problem**: Framework requires Python 3.11+
**Solution**:
```bash
# Install Python 3.11 using pyenv
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0
```

#### 3. Memory/Performance Issues

**Problem**: High memory usage or slow performance
**Solution**:
- Use smaller models (phi3 instead of llama3.1)
- Reduce max_concurrent_agents in configuration
- Increase system RAM or use swap space
- Enable GPU acceleration if available

#### 4. Import Errors

**Problem**: Module import errors
**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 5. Database Permissions

**Problem**: SQLite database permission errors
**Solution**:
```bash
# Check file permissions
ls -la agent_memory.db

# Fix permissions
chmod 664 agent_memory.db

# Or use in-memory database for testing
export MEMORY_DB_PATH=":memory:"
```

### Performance Optimization

#### 1. Model Selection

Choose appropriate models based on your use case:

- **Development/Testing**: `phi3` (lightweight, fast)
- **General Purpose**: `llama3.1` (balanced performance)
- **Code Generation**: `codellama` (specialized for coding)
- **Advanced Reasoning**: `mistral` (high quality, slower)

#### 2. Memory Configuration

Optimize memory usage:

```python
# Use in-memory database for temporary workloads
memory_manager = MemoryManager(db_path=":memory:")

# Limit memory retention
memory_manager.set_retention_policy(max_memories=1000, max_age_days=30)
```

#### 3. Concurrent Processing

Configure concurrency based on your system:

```python
# Limit concurrent agents based on available resources
orchestrator = AgentOrchestrator(
    max_concurrent_agents=3,  # Adjust based on RAM
    task_timeout=60           # Prevent hanging tasks
)
```

## Advanced Configuration

### 1. Custom Model Configuration

Configure specific models for different agent types:

```python
# Agent-specific model configuration
research_agent = orchestrator.register_agent(
    agent_id="researcher_001",
    role="Research Specialist",
    model="llama3.1",  # Use specific model
    model_params={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048
    }
)
```

### 2. Tool Configuration

Configure custom tools and tool categories:

```python
from src.tools.tool_registry import BaseTool

class DatabaseTool(BaseTool):
    def __init__(self, connection_string):
        super().__init__(
            name="database_query",
            description="Execute database queries",
            category="data"
        )
        self.connection_string = connection_string
    
    async def execute(self, query: str) -> dict:
        # Database query implementation
        pass

# Register custom tool
tool_registry.register_tool(DatabaseTool("sqlite:///data.db"))
```

### 3. Logging Configuration

Set up comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_framework.log'),
        logging.StreamHandler()
    ]
)

# Set specific log levels
logging.getLogger('src.core.agent').setLevel(logging.DEBUG)
logging.getLogger('src.integration.ollama_client').setLevel(logging.INFO)
```

### 4. Security Configuration

Implement security best practices:

```python
# Secure tool execution
tool_registry.set_security_policy({
    "allow_file_operations": False,
    "allow_network_access": True,
    "sandbox_mode": True,
    "max_execution_time": 30
})

# Memory encryption (for sensitive data)
memory_manager = MemoryManager(
    db_path="encrypted_memory.db",
    encryption_key="your-encryption-key"
)
```

## Next Steps

After successful installation and configuration:

1. **Read the API Documentation**: Familiarize yourself with the framework's API
2. **Explore Examples**: Run and modify the provided examples
3. **Create Your First Agent**: Build a simple agent for your specific use case
4. **Join the Community**: Participate in discussions and contribute to the project

## Getting Help

If you encounter issues during installation:

1. **Check the FAQ**: Common issues and solutions
2. **Search Issues**: Look for similar problems on GitHub
3. **Create an Issue**: Report new bugs or request help
4. **Join Discussions**: Ask questions in the community forum

---

**Installation complete! You're ready to start building with the Ollama Agentic Framework.**

## Quick Start: Creating Your First Agent

After installing EdgeBrain from PyPI, you can quickly create a simple agent application:

### 1. Create a Simple Agent Application

Create a new Python file in your project directory:

**`simple_agent.py`:**
```python
import asyncio
from edgebrain.core.agent import Agent
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.integration.ollama_client import OllamaIntegrationLayer

async def main():
    """Create and run a simple research agent."""
    
    # Initialize Ollama integration
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(ollama_integration=ollama)
    
    # Register a research agent
    agent = orchestrator.register_agent(
        agent_id="research_agent",
        role="Research Assistant",
        capabilities=["research", "analysis", "summarization"]
    )
    
    # Assign a research task
    task_id = await orchestrator.assign_task(
        agent_id="research_agent",
        task_description="Research the benefits of async programming in Python",
        context={"focus": "performance and scalability"}
    )
    
    # Wait for completion and get results
    results = await orchestrator.wait_for_completion(task_id)
    print(f"Research Results:\n{results}")
    
    # Cleanup
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Create an Async Code Generation Agent

Create a practical code generation agent using the official Ollama async client:

**`code_generator.py`:**
```python
import asyncio
import ollama

class SimpleCodeGenerator:
    """Simple code generator using async Ollama client."""
    
    def __init__(self, model="qwen2.5:3b"):
        self.model = model
        self.client = ollama.AsyncClient()
    
    async def generate_code(self, prompt: str, language: str = "python") -> str:
        """Generate code based on a prompt."""
        
        system_prompt = f"""You are an expert {language} programmer. 
Generate clean, well-documented, and functional code based on the user's request.
Include appropriate comments and follow best practices."""
        
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            
            if response and 'message' in response:
                return response['message']['content']
            else:
                return "Error: No response from model"
            
        except Exception as e:
            return f"Error generating code: {str(e)}"
    
    async def save_code(self, code: str, filename: str):
        """Save generated code to a file."""
        try:
            with open(filename, 'w') as f:
                f.write(code)
            print(f"✅ Code saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save code: {e}")

async def main():
    """Demo of code generation."""
    
    generator = SimpleCodeGenerator()
    
    # Generate a simple function
    prompt = "Create a Python function that validates email addresses using regex"
    code = await generator.generate_code(prompt)
    
    print("Generated Code:")
    print("-" * 50)
    print(code)
    print("-" * 50)
    
    # Save to file
    await generator.save_code(code, "email_validator.py")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Create a Multi-Agent Application

Create a more complex application with multiple agents:

**`multi_agent_app.py`:**
```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.integration.ollama_client import OllamaIntegrationLayer

async def create_research_and_code_pipeline():
    """Create a pipeline with research and code generation agents."""
    
    # Initialize
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    orchestrator = AgentOrchestrator(ollama_integration=ollama)
    
    # Register research agent
    research_agent = orchestrator.register_agent(
        agent_id="researcher",
        role="Research Specialist",
        capabilities=["research", "analysis"]
    )
    
    # Register code agent
    code_agent = orchestrator.register_agent(
        agent_id="coder",
        role="Code Generator",
        capabilities=["coding", "implementation"]
    )
    
    # Phase 1: Research
    research_task = await orchestrator.assign_task(
        agent_id="researcher",
        task_description="Research best practices for async Python programming",
        context={"depth": "intermediate", "focus": "practical examples"}
    )
    
    research_results = await orchestrator.wait_for_completion(research_task)
    print("Research completed!")
    
    # Phase 2: Code Generation based on research
    code_task = await orchestrator.assign_task(
        agent_id="coder",
        task_description=f"Generate Python code examples based on this research: {research_results}",
        context={"language": "python", "include_comments": True}
    )
    
    code_results = await orchestrator.wait_for_completion(code_task)
    print("Code generation completed!")
    print(f"Generated Code:\n{code_results}")
    
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(create_research_and_code_pipeline())
```

### 4. Run Your Applications

Execute your agent applications:

```bash
# Run simple agent
python simple_agent.py

# Run code generation agent
python code_agent.py

# Run multi-agent pipeline
python multi_agent_app.py
```

### 5. Project Structure

Your project directory should look like this:

```
my-edgebrain-app/
├── venv/                    # Virtual environment
├── simple_agent.py         # Basic agent example
├── code_agent.py          # Code generation example
├── multi_agent_app.py     # Multi-agent pipeline
├── factorial_function.py  # Generated code output
└── requirements.txt       # Dependencies (optional)
```

**Optional `requirements.txt`:**
```txt
edgebrain>=0.1.1
ollama
asyncio
```

