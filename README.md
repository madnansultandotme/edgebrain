# EdgeBrain: Ollama Agentic Framework

A powerful, extensible framework for building autonomous AI agents using Ollama-based language models. This framework provides a complete solution for creating, orchestrating, and managing AI agents that can work independently or collaboratively to solve complex tasks.

## 🌟 Features

### Core Capabilities
- **Multi-Agent Orchestration**: Coordinate multiple agents working together on complex tasks
- **Flexible Agent Architecture**: Create specialized agents with custom roles, capabilities, and behaviors
- **Async Ollama Integration**: Native support for the official Ollama Python client with async/await patterns
- **Code Generation Engine**: Specialized agents for software development using qwen2.5:3b and other models
- **Tool Integration**: Extensible tool system for web search, file operations, calculations, and custom tools
- **Memory Management**: Persistent memory system with semantic search capabilities
- **Workflow Engine**: Define and execute complex multi-step workflows with dependencies
- **Inter-Agent Communication**: Built-in messaging system for agent collaboration

### Advanced Features
- **Asynchronous Processing**: Full async/await support for high-performance operations using AsyncClient
- **Real-time Code Generation**: Direct integration with qwen2.5:3b for instant code creation
- **Vector Memory**: Semantic memory storage with embedding-based retrieval
- **Task Scheduling**: Priority-based task queue with automatic assignment
- **Error Handling**: Robust error handling and recovery mechanisms with graceful fallbacks
- **Extensible Architecture**: Plugin-based system for easy customization and extension
- **Comprehensive Testing**: Full test suite with mock integrations for development

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Ollama installed and running
- Required models: `ollama pull qwen2.5:3b` (for code generation)
- SQLite (included with Python)

### Installation

### Quick Install from PyPI

```bash
# Install EdgeBrain
pip install edgebrain

# Install official Ollama async client
pip install ollama

# Pull recommended models
ollama pull qwen2.5:3b    # Fast code generation
ollama pull llama3.1      # General purpose
```

### Development Installation

1. **Clone the repository:**
```bash
git clone https://github.com/madnansultandotme/ollama-agentic-framework.git
cd ollama-agentic-framework
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install ollama  # Official Ollama Python client
```

3. **Install the framework:**
```bash
pip install -e .
```

### Basic Usage

Here's a simple example using EdgeBrain from PyPI:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.integration.ollama_client import OllamaIntegrationLayer

async def main():
    # Initialize Ollama integration
    ollama_integration = OllamaIntegrationLayer()
    await ollama_integration.initialize()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration
    )
    
    # Register an agent
    agent = orchestrator.register_agent(
        agent_id="assistant",
        role="Research Assistant",
        capabilities=["research", "analysis"]
    )
    
    # Assign a task
    task_id = await orchestrator.assign_task(
        agent_id="assistant",
        task_description="Research the benefits of async programming",
        context={"focus": "Python development"}
    )
    
    # Get results
    results = await orchestrator.wait_for_completion(task_id)
    print(f"Results: {results}")
    
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Quick Start: Simple Code Generation

Create a basic code generation agent:

```python
import asyncio
import ollama

async def generate_code():
    client = ollama.AsyncClient()
    
    response = await client.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": "Create a function to calculate fibonacci numbers"}
        ]
    )
    
    print(response['message']['content'])

asyncio.run(generate_code())
```
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    # Create an agent
    agent = orchestrator.register_agent(
        agent_id="researcher_001",
        role="Research Specialist",
        description="Conducts research and analysis",
        model="llama3.1"
    )
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Create and assign a task
    task_id = await orchestrator.create_task(
        description="Research the latest trends in artificial intelligence"
    )
    
    await orchestrator.assign_task_to_agent(task_id, agent.agent_id)
    
    # Monitor execution
    # ... (see examples for complete implementation)
    
    await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Direct Code Generation Example

For immediate code generation using the async Ollama client:

```python
import asyncio
from ollama import AsyncClient

async def generate_code():
    client = AsyncClient()
    
    # Simple code generation
    message = {
        'role': 'user', 
        'content': 'Create a Python function to calculate factorial'
    }
    
    response = await client.chat(model='qwen2.5:3b', messages=[message])
    print(response.message.content)
    
    # With system prompt for better results
    messages = [
        {
            'role': 'system',
            'content': 'You are a Python expert. Write clean, documented code.'
        },
        {
            'role': 'user',
            'content': 'Create a Fibonacci sequence generator with error handling'
        }
    ]
    
    response = await client.chat(model='qwen2.5:3b', messages=messages)
    
    # Save generated code
    with open('generated_fibonacci.py', 'w') as f:
        f.write(response.message.content)

asyncio.run(generate_code())
```

## 📚 Documentation

### Core Components

#### Agent Orchestrator
The central control unit that manages agents, tasks, and workflows. It handles:
- Agent lifecycle management
- Task distribution and execution
- Inter-agent communication
- Workflow orchestration

#### Agents
Autonomous entities with specific roles and capabilities. Each agent has:
- Unique identity and role
- Custom capabilities and tools
- Memory and learning systems
- Goal-oriented behavior

#### Tool Registry
Extensible system for managing tools that agents can use:
- Built-in tools (web search, file operations, calculations)
- Custom tool development
- Tool discovery and validation
- Secure tool execution

#### Memory Manager
Persistent storage system for agent knowledge:
- Short-term context memory
- Long-term knowledge storage
- Semantic search capabilities
- Memory importance scoring

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                       │
├─────────────────────────────────────────────────────────────┤
│  Task Management  │  Agent Lifecycle  │  Communication     │
│  Workflow Engine  │  Resource Mgmt    │  Event Handling    │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Agents      │  │  Tool Registry  │  │ Memory Manager  │
│                 │  │                 │  │                 │
│ • Research      │  │ • Web Search    │  │ • Vector Store  │
│ • Writing       │  │ • File Ops      │  │ • Semantic      │
│ • Analysis      │  │ • Calculator    │  │   Search        │
│ • Custom        │  │ • Custom Tools  │  │ • Persistence   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
           │                    │                    │
           └────────────────────┼────────────────────┘
                               ▼
                    ┌─────────────────┐
                    │ Ollama Client   │
                    │                 │
                    │ • Model Mgmt    │
                    │ • Generation    │
                    │ • Tool Calling  │
                    │ • Streaming     │
                    └─────────────────┘
```

## 🛠️ Examples

The framework includes several comprehensive examples:

### 1. Simple Research Agent
A basic agent that conducts research and provides summaries.

```bash
python examples/simple_research_agent.py
# Or specify custom topic:
python examples/simple_research_agent.py "machine learning trends 2025"
```

### 2. Code Generation Agent (NEW!)
An agent specialized in software development using qwen2.5:3b model.

```bash
# Direct code generation (fast)
python examples/code_generation_agent.py --simple

# Full agent framework integration
python examples/code_generation_agent.py
```

**Features:**
- Generates complete Python functions with documentation
- Creates web scrapers, APIs, algorithms
- Includes error handling and best practices
- Saves code to files automatically
- Real-time async generation

### 3. Async Ollama Testing
Test the direct async integration with various models.

```bash
python examples/test_async_ollama.py
python examples/simple_code_test.py
```

### 4. Multi-Agent Collaboration
Multiple agents working together to create a technical blog post.

```bash
python examples/multi_agent_collaboration.py
```

### 5. Enhanced Research Agent
Advanced research capabilities with real web search and file output.

```bash
python examples/enhanced_research_agent.py
```

### 6. Comprehensive Demo
A full demonstration of all framework capabilities.

```bash
python examples/comprehensive_demo.py
```

## 🔧 Configuration

### Async Ollama Configuration

The framework supports both the custom integration layer and direct async client usage:

**Direct AsyncClient (Recommended for Code Generation):**
```python
from ollama import AsyncClient

async def setup_direct_ollama():
    client = AsyncClient()
    # Test connection
    response = await client.chat(
        model='qwen2.5:3b',
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
    return client
```

**Custom Integration Layer:**
```python
ollama_integration = OllamaIntegrationLayer(
    base_url="http://localhost:11434",  # Ollama server URL
    default_model="llama3.1",           # Default model to use
    timeout=30                          # Request timeout
)
```

### Model Recommendations

- **qwen2.5:3b**: Best for code generation (fast, lightweight, high quality)
- **llama3.1**: General purpose tasks, research, analysis
- **codellama**: Alternative for code tasks (larger, more detailed)

### Memory Configuration

Configure the memory system for your needs:

```python
memory_manager = MemoryManager(
    db_path="agent_memory.db",    # Database file path
    embedding_dim=384             # Embedding vector dimension
)
```

### Tool Configuration

Add custom tools to extend agent capabilities:

```python
from src.tools.tool_registry import BaseTool

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="My custom tool",
            category="custom"
        )
    
    async def execute(self, param: str) -> dict:
        # Tool implementation
        return {"result": f"Processed: {param}"}

# Register the tool
tool_registry.register_tool(CustomTool())
```

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_ollama_integration.py -v
python -m pytest tests/test_tool_registry.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📦 Project Structure

```
edgebrain/
├── src/                          # Source code
│   ├── core/                     # Core framework components
│   │   ├── agent.py             # Agent implementation
│   │   └── orchestrator.py      # Orchestrator implementation
│   ├── integration/              # External integrations
│   │   └── ollama_client.py     # Ollama integration
│   ├── tools/                    # Tool system
│   │   └── tool_registry.py     # Tool registry and built-in tools
│   ├── memory/                   # Memory management
│   │   └── memory_manager.py    # Memory system implementation
│   └── __init__.py
├── tests/                        # Test suite
│   ├── test_ollama_integration.py
│   ├── test_tool_registry.py
│   └── __init__.py
├── examples/                     # Usage examples
│   ├── simple_research_agent.py
│   ├── multi_agent_collaboration.py
│   ├── code_generation_agent.py
│   └── comprehensive_demo.py
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Run tests to ensure everything works
5. Make your changes
6. Add tests for new functionality
7. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all public methods
- Maintain test coverage above 90%

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) for providing the foundation for local LLM inference
- The open-source AI community for inspiration and best practices
- Contributors and users who help improve this framework

## 📞 Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/madnansultandotme/ollama-agentic-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/madnansultandotme/ollama-agentic-framework/discussions)
- **Email**: info.adnansultan@gmail.com

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Core agent framework
- ✅ Ollama integration
- ✅ Basic tool system
- ✅ Memory management
- ✅ Multi-agent orchestration

### Version 1.1 (Planned)
- 🔄 Enhanced tool ecosystem
- 🔄 Web interface for agent management
- 🔄 Advanced workflow templates
- 🔄 Performance optimizations

### Version 2.0 (Future)
- 🔮 Multi-modal agent support
- 🔮 Distributed agent networks
- 🔮 Advanced learning algorithms
- 🔮 Enterprise features

---

**Built with ❤️ by the Muhammad Adnan Sultan**

