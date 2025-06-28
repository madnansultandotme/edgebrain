# Usage Guide

This guide provides practical tutorials and examples for using the EdgeBrain framework effectively, including the new async Ollama integration and code generation capabilities. Whether you're building your first agent or creating complex multi-agent systems, this guide will help you understand the framework's capabilities and best practices.

## Prerequisites

Before starting, ensure you have EdgeBrain installed from PyPI:

```bash
# Install from PyPI
pip install edgebrain
pip install ollama

# Verify Ollama is running
ollama serve

# Pull required models
ollama pull qwen2.5:3b    # For code generation
ollama pull llama3.1      # For general tasks
```

## Table of Contents

- [Getting Started](#getting-started)
- [PyPI Installation and Setup](#pypi-installation-and-setup)
- [Async Code Generation](#async-code-generation)
- [Direct Ollama Integration](#direct-ollama-integration)
- [Creating Your First Agent](#creating-your-first-agent)
- [Working with Tools](#working-with-tools)
- [Memory and Learning](#memory-and-learning)
- [Multi-Agent Collaboration](#multi-agent-collaboration)
- [Workflows and Orchestration](#workflows-and-orchestration)
- [Advanced Patterns](#advanced-patterns)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### PyPI Installation and Setup

Create a new project with EdgeBrain:

```bash
# Create project directory
mkdir my-agent-project
cd my-agent-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install EdgeBrain and dependencies
pip install edgebrain ollama

# Verify installation
python -c "import edgebrain; print(f'EdgeBrain v{edgebrain.__version__} installed successfully')"
```

### Quick Start: Research Agent

Create your first research agent:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import AgentCapability, AgentGoal
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager

async def create_research_agent():
    # Initialize framework components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    tools = ToolRegistry()
    memory = MemoryManager()
    orchestrator = AgentOrchestrator(ollama, tools, memory)
    
    # Start orchestrator
    await orchestrator.start()
    
    # Define research capabilities
    research_capabilities = [
        AgentCapability(
            name="web_research",
            description="Search and analyze web content",
            parameters={"depth": "comprehensive", "sources": "multiple"}
        ),
        AgentCapability(
            name="data_synthesis",
            description="Combine information from multiple sources",
            parameters={"format": "structured_report"}
        )
    ]
    
    # Create research agent
    research_agent = orchestrator.register_agent(
        agent_id="research_specialist",
        role="Research Specialist", 
        description="Expert in conducting comprehensive research and analysis",
        capabilities=research_capabilities,
        model="llama3.1"
    )
    
    # Create and assign research task
    task_id = await orchestrator.create_task(
        description="Research the latest trends in artificial intelligence for 2024",
        context={
            "focus_areas": ["machine learning", "natural language processing", "computer vision"],
            "output_format": "detailed_report",
            "depth": "comprehensive"
        }
    )
    
    # Assign task to agent
    await orchestrator.assign_task_to_agent(task_id, "research_specialist")
    
    # Wait for completion
    await asyncio.sleep(5)  # Allow processing time
    
    # Get results
    task = orchestrator.get_task(task_id)
    if task and task.result:
        print("Research Results:")
        print(task.result)
    
    await orchestrator.stop()

# Run the research agent
asyncio.run(create_research_agent())
```

### Quick Start: Code Generation Agent

Create a specialized code generation agent:

```python
import asyncio
import ollama
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import AgentCapability
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager

async def create_code_agent():
    # Initialize components
    ollama_layer = OllamaIntegrationLayer()
    await ollama_layer.initialize()
    
    tools = ToolRegistry()
    memory = MemoryManager()
    orchestrator = AgentOrchestrator(ollama_layer, tools, memory)
    
    await orchestrator.start()
    
    # Define code generation capabilities
    code_capabilities = [
        AgentCapability(
            name="python_generation",
            description="Generate Python code with best practices",
            parameters={"style": "pep8", "testing": "included"}
        ),
        AgentCapability(
            name="code_review",
            description="Review and improve code quality",
            parameters={"focus": ["security", "performance", "readability"]}
        )
    ]
    
    # Create code agent with fast model
    code_agent = orchestrator.register_agent(
        agent_id="code_specialist",
        role="Code Generation Specialist",
        description="Expert in generating high-quality, functional code",
        capabilities=code_capabilities,
        model="qwen2.5:3b",  # Fast model for code generation
        system_prompt="""You are a senior software engineer. Always:
        - Write clean, readable code with proper documentation
        - Include error handling and edge cases
        - Follow best practices and design patterns
        - Add comprehensive docstrings and comments"""
    )
    
    # Direct code generation using async Ollama client
    client = ollama.AsyncClient()
    
    prompt = """Create a Python class for managing a library book inventory system.
    Include methods for:
    - Adding books
    - Searching books by title/author
    - Checking out/returning books
    - Generating availability reports
    
    Include proper error handling and documentation."""
    
    response = await client.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": "You are an expert Python developer."},
            {"role": "user", "content": prompt}
        ]
    )
    
    if response and 'message' in response:
        code = response['message']['content']
        
        # Save generated code
        filename = "library_inventory.py"
        with open(filename, 'w') as f:
            f.write(code)
        
        print(f"✅ Generated library inventory system saved to {filename}")
        print("\nGenerated Code Preview:")
        print("-" * 50)
        print(code[:500] + "..." if len(code) > 500 else code)
    
    await orchestrator.stop()

# Run the code agent
asyncio.run(create_code_agent())
```

### Basic Framework Setup

For more complex applications, set up the full framework components:

```python
import asyncio
from src.core.orchestrator import AgentOrchestrator
from src.integration.ollama_client import OllamaIntegrationLayer
from src.tools.tool_registry import ToolRegistry
from src.memory.memory_manager import MemoryManager

async def setup_framework():
    # Initialize Ollama integration
    ollama_integration = OllamaIntegrationLayer(
        base_url="http://localhost:11434",
        default_model="llama3.1"
    )
    
    # Check if Ollama is available
    if not await ollama_integration.initialize():
        raise RuntimeError("Failed to connect to Ollama. Ensure Ollama is running.")
    
    # Initialize tool registry with default tools
    tool_registry = ToolRegistry()
    
    # Initialize memory manager
    memory_manager = MemoryManager(db_path="agent_memory.db")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    return orchestrator

# Usage
orchestrator = await setup_framework()
```

### Environment Configuration

Create a configuration file for your project:

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class FrameworkConfig:
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1"
    memory_db_path: str = "agent_memory.db"
    log_level: str = "INFO"
    max_concurrent_agents: int = 5
    task_timeout: int = 300  # 5 minutes
    
    @classmethod
    def from_env(cls) -> 'FrameworkConfig':
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            default_model=os.getenv("DEFAULT_MODEL", cls.default_model),
            memory_db_path=os.getenv("MEMORY_DB_PATH", cls.memory_db_path),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            max_concurrent_agents=int(os.getenv("MAX_CONCURRENT_AGENTS", cls.max_concurrent_agents)),
            task_timeout=int(os.getenv("TASK_TIMEOUT", cls.task_timeout))
        )

# Load configuration
config = FrameworkConfig.from_env()
```

## Async Code Generation

### Overview

The EdgeBrain framework now supports direct asynchronous code generation using the official Ollama Python client. This provides high-performance, real-time code generation capabilities for various programming tasks.

### Key Features

- **Async Performance**: Non-blocking code generation using AsyncClient
- **qwen2.5:3b Model**: Optimized lightweight model for fast, high-quality code
- **System Prompts**: Enhanced control over code generation style and requirements
- **File Integration**: Automatic saving and management of generated code
- **Error Handling**: Robust error handling with graceful fallbacks

### Basic Code Generation

Generate a simple Python function:

```python
import asyncio
from ollama import AsyncClient

async def generate_simple_function():
    client = AsyncClient()
    
    message = {
        'role': 'user',
        'content': 'Create a Python function to calculate the area of a circle'
    }
    
    response = await client.chat(model='qwen2.5:3b', messages=[message])
    
    # Save the code
    with open('circle_area.py', 'w') as f:
        f.write(response.message.content or "No content generated")
    
    print("Generated circle area function")
    return response.message.content

# Run it
result = asyncio.run(generate_simple_function())
print(result)
```

### Advanced Code Generation with System Prompts

Use system prompts for better, more structured code:

```python
async def generate_advanced_function():
    client = AsyncClient()
    
    messages = [
        {
            'role': 'system',
            'content': '''You are a Python expert. Generate clean, well-documented code with:
            - Type hints for all parameters and return values
            - Comprehensive docstrings with examples
            - Proper error handling
            - Following PEP 8 standards'''
        },
        {
            'role': 'user',
            'content': '''Create a class for managing a bank account with methods for:
            - deposit (with validation)
            - withdraw (with overdraft protection)
            - get_balance
            - transaction_history'''
        }
    ]
    
    response = await client.chat(model='qwen2.5:3b', messages=messages)
    
    # Save to file
    with open('bank_account.py', 'w') as f:
        f.write(response.message.content or "No content generated")
    
    return response.message.content
```

### Batch Code Generation

Generate multiple related files:

```python
async def generate_web_scraper_project():
    client = AsyncClient()
    
    # Define generation tasks
    tasks = [
        {
            'filename': 'scraper_base.py',
            'prompt': 'Create a base web scraper class with rate limiting and error handling'
        },
        {
            'filename': 'html_parser.py', 
            'prompt': 'Create utilities for parsing HTML content with BeautifulSoup'
        },
        {
            'filename': 'data_storage.py',
            'prompt': 'Create a class for storing scraped data in CSV and JSON formats'
        }
    ]
    
    # Generate all files
    for task in tasks:
        message = {'role': 'user', 'content': task['prompt']}
        response = await client.chat(model='qwen2.5:3b', messages=[message])
        
        with open(task['filename'], 'w') as f:
            f.write(response.message.content or f"# {task['filename']}\n# Generation failed")
        
        print(f"✅ Generated {task['filename']}")

asyncio.run(generate_web_scraper_project())
```

### Using the DirectOllamaIntegration Class

For more structured code generation within the framework:

```python
from examples.code_generation_agent import DirectOllamaIntegration, SimpleCodeGenerator

async def structured_code_generation():
    # Initialize integration
    ollama_integration = DirectOllamaIntegration()
    await ollama_integration.initialize()
    
    # Create code generator
    code_gen = SimpleCodeGenerator(ollama_integration)
    
    # Generate different types of code
    fibonacci_code = await code_gen.generate_fibonacci_function()
    web_scraper_code = await code_gen.generate_web_scraper_class()
    flask_api_code = await code_gen.generate_flask_api()
    
    # Save to files
    files = [
        ('fibonacci.py', fibonacci_code),
        ('web_scraper.py', web_scraper_code),
        ('flask_api.py', flask_api_code)
    ]
    
    for filename, code in files:
        with open(filename, 'w') as f:
            f.write(code)
        print(f"💾 Saved {filename}")
    
    return files

# Generate structured project
files = asyncio.run(structured_code_generation())
print(f"Generated {len(files)} code files")
```

## Direct Ollama Integration

### Overview

For applications requiring direct interaction with Ollama models, the framework provides seamless integration. You can directly call Ollama models from your Python code, allowing for flexible and dynamic agent capabilities.

### Example: Direct Model Invocation

Here's how you can directly invoke an Ollama model to generate a report:

```python
from ollama import Client

def generate_report():
    client = Client()
    
    response = client.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": "Generate a report on the impact of AI in healthcare"
            }
        ]
    )
    
    # The generated report
    report = response.message.content
    print(report)

# Call the function
generate_report()
```

### AsyncClient vs. Framework Integration

**AsyncClient (Recommended for Code Generation):**
- Direct access to Ollama models
- Minimal overhead, maximum performance
- Perfect for focused code generation tasks
- Simple async/await patterns

**Framework Integration (For Complex Workflows):**
- Full agent orchestration
- Tool integration and memory management
- Multi-agent coordination
- Complex workflow support

### AsyncClient Best Practices

```python
import asyncio
from ollama import AsyncClient
from typing import Optional, List, Dict

class CodeGenerationService:
    def __init__(self, model: str = 'qwen2.5:3b'):
        self.model = model
        self.client = AsyncClient()
    
    async def generate_with_context(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate code with full context support."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        # Add conversation context
        if context:
            messages.extend(context)
        
        # Add current prompt
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            response = await self.client.chat(model=self.model, messages=messages)
            return response.message.content or "No content generated"
        except Exception as e:
            return f"Error generating code: {e}"
    
    async def generate_and_save(
        self,
        prompt: str,
        filename: str,
        system_prompt: Optional[str] = None
    ) -> bool:
        """Generate code and save to file."""
        code = await self.generate_with_context(prompt, system_prompt)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"✅ Generated and saved {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save {filename}: {e}")
            return False

# Usage example
async def main():
    service = CodeGenerationService()
    
    await service.generate_and_save(
        prompt="Create a REST API client for GitHub API with authentication",
        filename="github_client.py",
        system_prompt="You are a Python expert. Write production-ready code with proper error handling and documentation."
    )

asyncio.run(main())
```

### Error Handling and Retry Logic

```python
import asyncio
from ollama import AsyncClient
from typing import Optional
import time

async def robust_code_generation(
    prompt: str,
    model: str = 'qwen2.5:3b',
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[str]:
    """Generate code with retry logic and error handling."""
    client = AsyncClient()
    
    for attempt in range(max_retries):
        try:
            message = {'role': 'user', 'content': prompt}
            response = await client.chat(model=model, messages=[message])
            
            content = response.message.content
            if content and len(content.strip()) > 0:
                return content
            else:
                print(f"⚠️ Empty response on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    print("❌ All attempts failed")
    return None

# Usage
async def safe_generation_example():
    prompt = "Create a Python function for binary search with detailed comments"
    code = await robust_code_generation(prompt)
    
    if code:
        with open('binary_search.py', 'w') as f:
            f.write(code)
        print("✅ Code generated successfully")
    else:
        print("❌ Code generation failed")

asyncio.run(safe_generation_example())
```

## Creating Your First Agent

### Simple Agent Example

Let's create a basic research agent:

```python
from src.core.agent import AgentCapability

async def create_research_agent(orchestrator):
    # Define agent capabilities
    capabilities = [
        AgentCapability(
            name="web_search",
            description="Search the web for information on various topics"
        ),
        AgentCapability(
            name="data_analysis",
            description="Analyze and synthesize information from multiple sources"
        ),
        AgentCapability(
            name="report_writing",
            description="Generate comprehensive reports based on research"
        )
    ]
    
    # Create the agent
    agent = orchestrator.register_agent(
        agent_id="research_agent_001",
        role="Research Specialist",
        description="An AI agent specialized in conducting thorough research and analysis",
        model="llama3.1",
        capabilities=capabilities,
        system_prompt="""You are a research specialist AI agent. Your role is to:
        1. Conduct comprehensive research on assigned topics
        2. Analyze information from multiple sources
        3. Synthesize findings into clear, actionable insights
        4. Generate well-structured reports
        
        Always be thorough, accurate, and cite your sources when possible.
        Focus on providing valuable insights and actionable recommendations."""
    )
    
    return agent

# Usage
research_agent = await create_research_agent(orchestrator)
print(f"Created agent: {research_agent.agent_id} with role: {research_agent.role}")
```

### Assigning Tasks to Agents

Once you have an agent, you can assign tasks:

```python
async def assign_research_task(orchestrator, agent):
    # Create a research task
    task_id = await orchestrator.create_task(
        description="Research the current state of artificial intelligence in healthcare",
        priority=7,
        context={
            "domain": "healthcare",
            "focus_areas": ["diagnosis", "treatment", "drug discovery"],
            "output_format": "executive summary",
            "target_audience": "healthcare executives"
        }
    )
    
    # Assign task to the agent
    success = await orchestrator.assign_task_to_agent(task_id, agent.agent_id)
    
    if success:
        print(f"Task {task_id} assigned successfully to {agent.role}")
        return task_id
    else:
        print("Failed to assign task")
        return None

# Usage
await orchestrator.start()
task_id = await assign_research_task(orchestrator, research_agent)
```

### Monitoring Task Execution

Monitor the progress of your tasks:

```python
async def monitor_task_execution(orchestrator, task_id, timeout=300):
    """Monitor task execution with timeout."""
    import time
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        task = orchestrator.get_task(task_id)
        
        if task:
            print(f"Task status: {task.status.value}")
            
            if task.status.value in ["completed", "failed", "cancelled"]:
                return task
        
        await asyncio.sleep(2)
    
    print("Task monitoring timed out")
    return None

# Usage
final_task = await monitor_task_execution(orchestrator, task_id)

if final_task and final_task.status.value == "completed":
    print("Task completed successfully!")
    print(f"Result: {final_task.result}")
else:
    print("Task did not complete successfully")
```

## Working with Tools

### Using Built-in Tools

The framework comes with several built-in tools:

```python
# Get available tools
tools = orchestrator.tool_registry.get_all_tools()
for tool in tools:
    print(f"Tool: {tool.name} - {tool.description}")

# Execute a tool directly
result = await orchestrator.tool_registry.execute_tool(
    "calculator",
    {"expression": "2 + 2 * 3"}
)
print(f"Calculator result: {result}")

# Search for tools
search_tools = orchestrator.tool_registry.search_tools("file")
print(f"File-related tools: {[tool.name for tool in search_tools]}")
```

### Creating Custom Tools

Create your own tools to extend agent capabilities:

```python
from src.tools.tool_registry import BaseTool
import aiohttp
import json

class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    def __init__(self, api_key: str):
        super().__init__(
            name="weather_lookup",
            description="Get current weather information for a location",
            category="information"
        )
        self.api_key = api_key
    
    async def execute(self, location: str, units: str = "metric") -> dict:
        """Get weather for a location."""
        try:
            # Mock weather API call (replace with real API)
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "location": location,
                            "temperature": data["main"]["temp"],
                            "description": data["weather"][0]["description"],
                            "humidity": data["main"]["humidity"],
                            "units": units
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"API request failed with status {response.status}"
                        }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Register the custom tool
weather_tool = WeatherTool(api_key="your_api_key_here")
orchestrator.tool_registry.register_tool(weather_tool)

# Test the tool
weather_result = await orchestrator.tool_registry.execute_tool(
    "weather_lookup",
    {"location": "New York", "units": "imperial"}
)
print(f"Weather result: {weather_result}")
```

### Tool Categories and Organization

Organize tools by category for better management:

```python
# Get tools by category
information_tools = orchestrator.tool_registry.get_tools_by_category("information")
utility_tools = orchestrator.tool_registry.get_tools_by_category("utility")

print(f"Information tools: {[tool.name for tool in information_tools]}")
print(f"Utility tools: {[tool.name for tool in utility_tools]}")

# Get all categories
categories = orchestrator.tool_registry.get_categories()
print(f"Available categories: {categories}")
```

## Memory and Learning

### Storing and Retrieving Memories

Agents can store and retrieve memories for learning and context:

```python
async def demonstrate_memory_usage(orchestrator, agent_id):
    memory_manager = orchestrator.memory_manager
    
    # Store different types of memories
    memories = [
        {
            "content": "Successfully completed research on AI in healthcare",
            "memory_type": "achievement",
            "importance": 0.8
        },
        {
            "content": "User prefers executive summaries over detailed reports",
            "memory_type": "preference",
            "importance": 0.9
        },
        {
            "content": "Healthcare domain requires focus on FDA regulations",
            "memory_type": "domain_knowledge",
            "importance": 0.7
        }
    ]
    
    # Store memories
    memory_ids = []
    for memory in memories:
        memory_id = await memory_manager.store_memory(
            agent_id=agent_id,
            content=memory["content"],
            memory_type=memory["memory_type"],
            importance=memory["importance"]
        )
        memory_ids.append(memory_id)
        print(f"Stored memory: {memory_id}")
    
    # Retrieve memories by type
    achievements = await memory_manager.retrieve_memories(
        agent_id=agent_id,
        memory_type="achievement"
    )
    print(f"Achievement memories: {achievements}")
    
    # Search memories semantically
    search_results = await memory_manager.search_memories(
        query="healthcare research",
        agent_id=agent_id,
        limit=5
    )
    print(f"Search results: {search_results}")
    
    # Get memory statistics
    stats = await memory_manager.get_memory_stats(agent_id)
    print(f"Memory stats: {stats}")

# Usage
await demonstrate_memory_usage(orchestrator, research_agent.agent_id)
```

### Memory-Based Learning

Use memories to improve agent performance:

```python
async def create_learning_agent(orchestrator):
    """Create an agent that learns from its experiences."""
    
    learning_agent = orchestrator.register_agent(
        agent_id="learning_agent_001",
        role="Learning Assistant",
        description="An agent that learns and adapts from previous interactions",
        system_prompt="""You are a learning assistant that improves over time.
        Before starting any task, review your previous experiences and learnings.
        After completing tasks, reflect on what worked well and what could be improved.
        Use your memories to provide better assistance in future interactions."""
    )
    
    return learning_agent

async def task_with_learning(orchestrator, agent, task_description):
    """Execute a task with learning integration."""
    
    # Retrieve relevant memories before starting
    relevant_memories = await orchestrator.memory_manager.search_memories(
        query=task_description,
        agent_id=agent.agent_id,
        limit=5
    )
    
    # Create task with memory context
    task_id = await orchestrator.create_task(
        description=task_description,
        context={
            "relevant_memories": relevant_memories,
            "learning_mode": True
        }
    )
    
    # Execute task
    await orchestrator.assign_task_to_agent(task_id, agent.agent_id)
    
    # Wait for completion and store learning
    task = await monitor_task_execution(orchestrator, task_id)
    
    if task and task.status.value == "completed":
        # Store learning from this task
        await orchestrator.memory_manager.store_memory(
            agent_id=agent.agent_id,
            content=f"Completed task: {task_description}. Outcome: {task.result}",
            memory_type="experience",
            importance=0.8
        )
    
    return task

# Usage
learning_agent = await create_learning_agent(orchestrator)
task = await task_with_learning(
    orchestrator,
    learning_agent,
    "Analyze customer feedback and provide improvement recommendations"
)
```

## Multi-Agent Collaboration

### Creating Agent Teams

Build teams of specialized agents:

```python
async def create_content_creation_team(orchestrator):
    """Create a team of agents for content creation."""
    
    # Research Agent
    researcher = orchestrator.register_agent(
        agent_id="researcher_001",
        role="Research Specialist",
        description="Conducts thorough research and gathers information",
        capabilities=[
            AgentCapability(name="web_research", description="Search for information"),
            AgentCapability(name="data_analysis", description="Analyze research data")
        ],
        model="llama3.1"
    )
    
    # Writer Agent
    writer = orchestrator.register_agent(
        agent_id="writer_001",
        role="Content Writer",
        description="Creates engaging and informative content",
        capabilities=[
            AgentCapability(name="content_creation", description="Write articles and posts"),
            AgentCapability(name="editing", description="Edit and refine content")
        ],
        model="llama3.1"
    )
    
    # Editor Agent
    editor = orchestrator.register_agent(
        agent_id="editor_001",
        role="Content Editor",
        description="Reviews and improves content quality",
        capabilities=[
            AgentCapability(name="content_review", description="Review content quality"),
            AgentCapability(name="fact_checking", description="Verify information accuracy")
        ],
        model="llama3.1"
    )
    
    return {
        "researcher": researcher,
        "writer": writer,
        "editor": editor
    }

# Create the team
team = await create_content_creation_team(orchestrator)
print(f"Created team with {len(team)} agents")
```

### Inter-Agent Communication

Enable agents to communicate and collaborate:

```python
async def demonstrate_agent_communication(orchestrator, team):
    """Demonstrate communication between agents."""
    
    researcher = team["researcher"]
    writer = team["writer"]
    editor = team["editor"]
    
    # Researcher sends findings to writer
    await orchestrator.send_message(
        sender_id=researcher.agent_id,
        recipient_id=writer.agent_id,
        content="""Research completed on AI trends in 2024:
        
        Key findings:
        - 40% increase in enterprise AI adoption
        - Focus on responsible AI development
        - Growth in multimodal AI applications
        
        Detailed data and sources are available in shared memory.""",
        message_type="research_findings"
    )
    
    # Writer acknowledges and requests clarification
    await orchestrator.send_message(
        sender_id=writer.agent_id,
        recipient_id=researcher.agent_id,
        content="Thank you for the research! Could you provide more specific statistics on enterprise adoption rates by industry?",
        message_type="clarification_request"
    )
    
    # Broadcast project update
    await orchestrator.broadcast_message(
        sender_id=researcher.agent_id,
        content="Research phase completed. Moving to content creation phase.",
        message_type="project_update"
    )
    
    print("Agent communication demonstrated")

# Usage
await demonstrate_agent_communication(orchestrator, team)
```

### Collaborative Task Execution

Coordinate multiple agents on a single task:

```python
async def collaborative_content_creation(orchestrator, team, topic):
    """Create content collaboratively using multiple agents."""
    
    # Phase 1: Research
    research_task_id = await orchestrator.create_task(
        description=f"Conduct comprehensive research on: {topic}",
        priority=8,
        context={"phase": "research", "topic": topic}
    )
    
    await orchestrator.assign_task_to_agent(research_task_id, team["researcher"].agent_id)
    research_task = await monitor_task_execution(orchestrator, research_task_id)
    
    if research_task.status.value != "completed":
        print("Research phase failed")
        return None
    
    # Phase 2: Writing
    writing_task_id = await orchestrator.create_task(
        description=f"Write an article about: {topic}",
        priority=7,
        context={
            "phase": "writing",
            "topic": topic,
            "research_results": research_task.result
        }
    )
    
    await orchestrator.assign_task_to_agent(writing_task_id, team["writer"].agent_id)
    writing_task = await monitor_task_execution(orchestrator, writing_task_id)
    
    if writing_task.status.value != "completed":
        print("Writing phase failed")
        return None
    
    # Phase 3: Editing
    editing_task_id = await orchestrator.create_task(
        description=f"Review and edit the article about: {topic}",
        priority=6,
        context={
            "phase": "editing",
            "topic": topic,
            "draft_content": writing_task.result
        }
    )
    
    await orchestrator.assign_task_to_agent(editing_task_id, team["editor"].agent_id)
    editing_task = await monitor_task_execution(orchestrator, editing_task_id)
    
    return editing_task

# Usage
final_article = await collaborative_content_creation(
    orchestrator,
    team,
    "The Future of Artificial Intelligence in Business"
)

if final_article and final_article.status.value == "completed":
    print("Article creation completed successfully!")
    print(f"Final article: {final_article.result}")
```

## Workflows and Orchestration

### Creating Workflows

Define complex multi-step workflows:

```python
from src.core.orchestrator import Workflow, WorkflowStep

async def create_product_launch_workflow():
    """Create a workflow for product launch preparation."""
    
    workflow = Workflow(
        name="Product Launch Preparation",
        description="Comprehensive workflow for preparing a product launch",
        steps=[
            WorkflowStep(
                id="market_research",
                description="Conduct market research and competitive analysis",
                agent_role="Research Specialist",
                dependencies=[],
                context={"research_scope": "comprehensive", "timeline": "2_weeks"}
            ),
            WorkflowStep(
                id="content_strategy",
                description="Develop content strategy and messaging",
                agent_role="Content Strategist",
                dependencies=["market_research"],
                context={"content_types": ["blog", "social", "email"], "tone": "professional"}
            ),
            WorkflowStep(
                id="content_creation",
                description="Create marketing content and materials",
                agent_role="Content Writer",
                dependencies=["content_strategy"],
                context={"deliverables": ["landing_page", "blog_posts", "social_content"]}
            ),
            WorkflowStep(
                id="design_review",
                description="Review and approve all content and designs",
                agent_role="Design Reviewer",
                dependencies=["content_creation"],
                context={"review_criteria": ["brand_consistency", "message_clarity", "visual_appeal"]}
            ),
            WorkflowStep(
                id="launch_coordination",
                description="Coordinate launch activities and timeline",
                agent_role="Project Manager",
                dependencies=["design_review"],
                context={"launch_channels": ["website", "social_media", "email"], "go_live_date": "2024-02-01"}
            )
        ],
        context={
            "product_name": "AI Assistant Pro",
            "target_audience": "business_professionals",
            "budget": "$50000",
            "timeline": "6_weeks"
        }
    )
    
    return workflow

async def execute_workflow_with_monitoring(orchestrator, workflow):
    """Execute a workflow with detailed monitoring."""
    
    print(f"Starting workflow: {workflow.name}")
    
    # Execute workflow
    success = await orchestrator.execute_workflow(workflow)
    
    if not success:
        print("Failed to start workflow")
        return False
    
    # Monitor workflow progress
    completed_steps = set()
    total_steps = len(workflow.steps)
    
    while len(completed_steps) < total_steps:
        # Check agent statuses
        agent_statuses = {}
        for agent in orchestrator.get_all_agents():
            status = agent.get_status()
            agent_statuses[agent.role] = status.value
        
        print(f"Workflow progress: {len(completed_steps)}/{total_steps} steps completed")
        print(f"Agent statuses: {agent_statuses}")
        
        # Simulate step completion check
        # In a real implementation, you'd track actual step completion
        await asyncio.sleep(5)
        
        # For demo purposes, assume steps complete over time
        if len(completed_steps) < total_steps:
            completed_steps.add(workflow.steps[len(completed_steps)].id)
    
    print("Workflow completed successfully!")
    return True

# Usage
workflow = await create_product_launch_workflow()
await execute_workflow_with_monitoring(orchestrator, workflow)
```

### Conditional Workflows

Create workflows with conditional logic:

```python
async def create_conditional_workflow(orchestrator, content_type):
    """Create a workflow that adapts based on content type."""
    
    base_steps = [
        WorkflowStep(
            id="content_planning",
            description="Plan content structure and approach",
            agent_role="Content Strategist",
            dependencies=[],
            context={"content_type": content_type}
        )
    ]
    
    # Add conditional steps based on content type
    if content_type == "technical_article":
        base_steps.extend([
            WorkflowStep(
                id="technical_research",
                description="Conduct technical research and validation",
                agent_role="Technical Researcher",
                dependencies=["content_planning"],
                context={"depth": "expert_level", "accuracy_required": True}
            ),
            WorkflowStep(
                id="technical_writing",
                description="Write technical content with code examples",
                agent_role="Technical Writer",
                dependencies=["technical_research"],
                context={"include_code": True, "target_audience": "developers"}
            )
        ])
    elif content_type == "marketing_copy":
        base_steps.extend([
            WorkflowStep(
                id="market_analysis",
                description="Analyze target market and messaging",
                agent_role="Marketing Analyst",
                dependencies=["content_planning"],
                context={"focus": "conversion_optimization"}
            ),
            WorkflowStep(
                id="copywriting",
                description="Create persuasive marketing copy",
                agent_role="Copywriter",
                dependencies=["market_analysis"],
                context={"tone": "persuasive", "cta_required": True}
            )
        ])
    
    # Common final step
    base_steps.append(
        WorkflowStep(
            id="final_review",
            description="Final review and approval",
            agent_role="Content Editor",
            dependencies=[step.id for step in base_steps[-1:]],  # Depends on last content step
            context={"review_type": "comprehensive"}
        )
    )
    
    workflow = Workflow(
        name=f"{content_type.title()} Creation Workflow",
        description=f"Adaptive workflow for creating {content_type}",
        steps=base_steps,
        context={"content_type": content_type, "adaptive": True}
    )
    
    return workflow

# Usage
technical_workflow = await create_conditional_workflow(orchestrator, "technical_article")
marketing_workflow = await create_conditional_workflow(orchestrator, "marketing_copy")
```

## Agent Types and Use Cases

EdgeBrain supports building various specialized agent types. Here are comprehensive examples:

### 1. Data Analysis Agent

Create an agent specialized in data processing and analysis:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import AgentCapability
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager

async def create_data_analysis_agent():
    # Initialize components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    tools = ToolRegistry()
    memory = MemoryManager()
    orchestrator = AgentOrchestrator(ollama, tools, memory)
    
    await orchestrator.start()
    
    # Define data analysis capabilities
    data_capabilities = [
        AgentCapability(
            name="statistical_analysis",
            description="Perform statistical computations and analysis",
            parameters={"methods": ["descriptive", "inferential", "regression"]}
        ),
        AgentCapability(
            name="data_visualization",
            description="Create charts and visualizations",
            parameters={"chart_types": ["bar", "line", "scatter", "heatmap"]}
        ),
        AgentCapability(
            name="pattern_recognition",
            description="Identify trends and patterns in data",
            parameters={"algorithms": ["clustering", "classification", "anomaly_detection"]}
        )
    ]
    
    # Create data analysis agent
    data_agent = orchestrator.register_agent(
        agent_id="data_analyst",
        role="Data Analysis Specialist",
        description="Expert in data processing, analysis, and visualization",
        capabilities=data_capabilities,
        model="mistral",  # Good for analytical reasoning
        system_prompt="""You are a data scientist with expertise in:
        - Statistical analysis and hypothesis testing
        - Data visualization and reporting
        - Pattern recognition and machine learning
        - Data quality assessment and cleaning
        Always provide evidence-based insights with proper statistical context."""
    )
    
    # Sample data analysis task
    sample_data = {
        "sales_data": [100, 150, 200, 175, 300, 250, 400, 350, 450, 500],
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    }
    
    # Store sample data
    await tools.execute_tool("data_storage", 
        action="store", 
        key="sales_analysis_data", 
        data=sample_data
    )
    
    # Create analysis task
    task_id = await orchestrator.create_task(
        description="Analyze the sales data trends and provide insights",
        context={
            "data_key": "sales_analysis_data",
            "analysis_type": "trend_analysis",
            "output_format": "detailed_report_with_recommendations"
        }
    )
    
    await orchestrator.assign_task_to_agent(task_id, "data_analyst")
    await asyncio.sleep(5)
    
    # Get analysis results
    task = orchestrator.get_task(task_id)
    if task and task.result:
        print("Data Analysis Results:")
        print(task.result)
    
    await orchestrator.stop()

# Run data analysis agent
asyncio.run(create_data_analysis_agent())
```

### 2. Multi-Agent Collaboration System

Create a team of agents working together:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator, Workflow, WorkflowStep
from edgebrain.core.agent import AgentCapability
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager

async def create_collaborative_team():
    """Create a team of agents for content creation workflow."""
    
    # Initialize components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    tools = ToolRegistry()
    memory = MemoryManager()
    orchestrator = AgentOrchestrator(ollama, tools, memory)
    
    await orchestrator.start()
    
    # 1. Research Agent
    research_agent = orchestrator.register_agent(
        agent_id="researcher",
        role="Research Specialist",
        description="Conducts thorough research and gathers information",
        capabilities=[
            AgentCapability(name="web_research", description="Search for information"),
            AgentCapability(name="data_analysis", description="Analyze research data")
        ],
        model="llama3.1"
    )
    
    # 2. Writer Agent  
    writer_agent = orchestrator.register_agent(
        agent_id="writer",
        role="Content Writer",
        description="Creates engaging and well-structured content",
        capabilities=[
            AgentCapability(name="content_creation", description="Write articles and reports"),
            AgentCapability(name="style_adaptation", description="Adapt writing style for audience")
        ],
        model="llama3.1"
    )
    
    # 3. Editor Agent
    editor_agent = orchestrator.register_agent(
        agent_id="editor",
        role="Content Editor",
        description="Reviews and improves content quality",
        capabilities=[
            AgentCapability(name="proofreading", description="Grammar and style correction"),
            AgentCapability(name="fact_checking", description="Verify content accuracy")
        ],
        model="llama3.1"
    )
    
    # Create workflow
    workflow = Workflow(
        name="content_creation_workflow",
        description="Collaborative content creation process",
        steps=[
            WorkflowStep(
                description="Research the topic and gather information",
                agent_role="Research Specialist",
                context={"depth": "comprehensive", "sources": "multiple"}
            ),
            WorkflowStep(
                description="Write article based on research findings",
                agent_role="Content Writer",
                dependencies=["research_step"],
                context={"style": "professional", "length": "1500_words"}
            ),
            WorkflowStep(
                description="Edit and proofread the article",
                agent_role="Content Editor", 
                dependencies=["writing_step"],
                context={"focus": ["grammar", "clarity", "flow"]}
            )
        ]
    )
    
    # Execute workflow
    topic = "The Future of Artificial Intelligence in Healthcare"
    print(f"🚀 Starting collaborative content creation on: {topic}")
    
    workflow_success = await orchestrator.execute_workflow(workflow)
    
    if workflow_success:
        print("✅ Collaborative workflow completed successfully!")
        
        # Get final content from memory
        memories = await memory.get_memories("editor", memory_type="task_result")
        if memories:
            print("\n📝 Final Article:")
            print(memories[-1].content)
    else:
        print("❌ Workflow execution failed")
    
    await orchestrator.stop()

# Run collaborative team
asyncio.run(create_collaborative_team())
```

### 3. Communication and Coordination Agent

Create an agent that manages communication between other agents:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import AgentCapability, AgentMessage
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager

async def create_communication_coordinator():
    """Create a communication coordinator agent."""
    
    # Initialize components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    tools = ToolRegistry()
    memory = MemoryManager() 
    orchestrator = AgentOrchestrator(ollama, tools, memory)
    
    await orchestrator.start()
    
    # Communication coordinator capabilities
    comm_capabilities = [
        AgentCapability(
            name="message_routing",
            description="Route messages between agents efficiently",
            parameters={"protocols": ["direct", "broadcast", "priority"]}
        ),
        AgentCapability(
            name="conflict_resolution",
            description="Resolve conflicts between agents",
            parameters={"strategies": ["mediation", "priority_based", "consensus"]}
        ),
        AgentCapability(
            name="workflow_coordination",
            description="Coordinate complex multi-agent workflows",
            parameters={"coordination_patterns": ["sequential", "parallel", "conditional"]}
        )
    ]
    
    # Create coordinator agent
    coordinator = orchestrator.register_agent(
        agent_id="coordinator",
        role="Communication Coordinator",
        description="Manages communication and coordination between agents",
        capabilities=comm_capabilities,
        model="llama3.1",
        system_prompt="""You are a communication coordinator responsible for:
        - Efficiently routing messages between agents
        - Resolving conflicts and deadlocks
        - Optimizing workflow execution
        - Maintaining system harmony and productivity
        Always prioritize clear communication and efficient task completion."""
    )
    
    # Create worker agents
    worker1 = orchestrator.register_agent(
        agent_id="worker_1",
        role="Task Worker",
        description="Handles data processing tasks",
        model="phi3"  # Lightweight for simple tasks
    )
    
    worker2 = orchestrator.register_agent(
        agent_id="worker_2", 
        role="Task Worker",
        description="Handles analysis tasks",
        model="phi3"
    )
    
    # Simulate coordinated task execution
    print("🔄 Testing agent coordination...")
    
    # Send coordination messages
    await orchestrator.send_message(
        sender_id="coordinator",
        recipient_id="worker_1",
        content="Process the incoming data batch and prepare summary",
        message_type="task_assignment"
    )
    
    await orchestrator.send_message(
        sender_id="coordinator", 
        recipient_id="worker_2",
        content="Analyze the processed data when worker_1 completes",
        message_type="conditional_task"
    )
    
    # Broadcast status update
    await orchestrator.broadcast_message(
        sender_id="coordinator",
        content="Workflow initiated: Data processing -> Analysis -> Report",
        message_type="status_update"
    )
    
    print("✅ Communication coordination setup complete")
    
    await orchestrator.stop()

# Run communication coordinator
asyncio.run(create_communication_coordinator())
```

### 4. Specialized Domain Agent (Financial Analysis)

Create a domain-specific agent with specialized knowledge:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import AgentCapability
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry, BaseTool
from edgebrain.memory.memory_manager import MemoryManager

# Custom financial tool
class FinancialCalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="financial_calculator",
            description="Perform financial calculations and analysis",
            category="finance",
            parameters={
                "type": "object",
                "properties": {
                    "calculation_type": {
                        "type": "string",
                        "description": "Type of calculation (roi, npv, irr, compound_interest)"
                    },
                    "values": {
                        "type": "object",
                        "description": "Input values for calculation"
                    }
                },
                "required": ["calculation_type", "values"]
            }
        )
    
    async def execute(self, calculation_type: str, values: dict) -> dict:
        """Perform financial calculations."""
        if calculation_type == "roi":
            initial = values.get("initial_investment", 0)
            final = values.get("final_value", 0)
            roi = ((final - initial) / initial) * 100 if initial > 0 else 0
            return {"roi_percentage": roi, "calculation": "ROI"}
        
        elif calculation_type == "compound_interest":
            principal = values.get("principal", 0)
            rate = values.get("annual_rate", 0) / 100
            time = values.get("years", 0)
            compounds = values.get("compounds_per_year", 1)
            
            amount = principal * (1 + rate/compounds) ** (compounds * time)
            return {
                "final_amount": amount,
                "interest_earned": amount - principal,
                "calculation": "Compound Interest"
            }
        
        return {"error": "Unsupported calculation type"}

async def create_financial_agent():
    """Create a specialized financial analysis agent."""
    
    # Initialize components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    tools = ToolRegistry()
    memory = MemoryManager()
    
    # Register custom financial tool
    financial_tool = FinancialCalculatorTool()
    tools.register_tool(financial_tool)
    
    orchestrator = AgentOrchestrator(ollama, tools, memory)
    await orchestrator.start()
    
    # Financial analysis capabilities
    financial_capabilities = [
        AgentCapability(
            name="investment_analysis",
            description="Analyze investment opportunities and risks",
            parameters={"metrics": ["roi", "npv", "irr", "payback_period"]}
        ),
        AgentCapability(
            name="market_analysis",
            description="Analyze market trends and conditions",
            parameters={"analysis_depth": "comprehensive", "timeframe": "multi_year"}
        ),
        AgentCapability(
            name="risk_assessment",
            description="Assess financial risks and mitigation strategies",
            parameters={"risk_types": ["market", "credit", "operational", "liquidity"]}
        ),
        AgentCapability(
            name="portfolio_optimization",
            description="Optimize investment portfolios",
            parameters={"optimization_goals": ["return", "risk", "diversification"]}
        )
    ]
    
    # Create financial agent
    financial_agent = orchestrator.register_agent(
        agent_id="financial_analyst",
        role="Financial Analysis Specialist",
        description="Expert in financial analysis, investment evaluation, and risk assessment",
        capabilities=financial_capabilities,
        model="mistral",  # Good for complex analytical reasoning
        system_prompt="""You are a senior financial analyst with expertise in:
        - Investment analysis and valuation
        - Market research and trend analysis
        - Risk assessment and mitigation
        - Portfolio management and optimization
        - Financial modeling and forecasting
        
        Always provide:
        - Evidence-based analysis with calculations
        - Risk assessments with mitigation strategies
        - Clear recommendations with rationale
        - Compliance considerations where applicable"""
    )
    
    # Sample financial analysis
    investment_data = {
        "initial_investment": 100000,
        "projected_returns": [15000, 18000, 22000, 25000, 28000],
        "discount_rate": 0.08,
        "market_conditions": "stable_growth",
        "sector": "technology",
        "risk_level": "moderate"
    }
    
    # Perform financial calculation
    roi_result = await tools.execute_tool("financial_calculator",
        calculation_type="roi",
        values={
            "initial_investment": 100000,
            "final_value": 150000
        }
    )
    
    print(f"📊 ROI Analysis: {roi_result}")
    
    # Create comprehensive analysis task
    task_id = await orchestrator.create_task(
        description="Perform comprehensive investment analysis for a technology startup",
        context={
            "investment_amount": 100000,
            "sector": "AI/Machine Learning",
            "timeframe": "5_years",
            "risk_tolerance": "moderate",
            "analysis_scope": "full_due_diligence"
        }
    )
    
    await orchestrator.assign_task_to_agent(task_id, "financial_analyst")
    await asyncio.sleep(5)
    
    # Get analysis results
    task = orchestrator.get_task(task_id)
    if task and task.result:
        print("\n💰 Financial Analysis Results:")
        print(task.result)
    
    await orchestrator.stop()

# Run financial agent
asyncio.run(create_financial_agent())
```

## Advanced Usage Patterns

### Tool Integration Patterns

#### 1. Sequential Tool Usage
```python
async def sequential_research_workflow():
    """Demonstrate sequential tool usage for research."""
    
    tools = ToolRegistry()
    
    # Step 1: Web search
    search_results = await tools.execute_tool("web_search",
        query="quantum computing applications 2024",
        max_results=5
    )
    
    # Step 2: Analyze search results
    analysis = await tools.execute_tool("text_analysis",
        text=" ".join([result['snippet'] for result in search_results['results']]),
        analysis_type="summary"
    )
    
    # Step 3: Store findings
    await tools.execute_tool("data_storage",
        action="store",
        key="quantum_research_findings",
        data={
            "search_results": search_results,
            "analysis": analysis,
            "timestamp": time.time()
        }
    )
    
    # Step 4: Generate report
    report_content = f"""# Quantum Computing Research Report
    
## Search Results Summary
{analysis.get('summary', 'No summary available')}

## Key Findings
{search_results.get('total_results', 0)} sources analyzed
    
## Detailed Results
{chr(10).join([f"- {r['title']}: {r['snippet']}" for r in search_results.get('results', [])])}
    """
    
    await tools.execute_tool("file_write",
        filename="quantum_computing_report.md",
        content=report_content
    )
    
    print("✅ Sequential research workflow completed")
```

#### 2. Parallel Tool Execution
```python
async def parallel_analysis_workflow():
    """Demonstrate parallel tool usage for efficiency."""
    
    tools = ToolRegistry()
    
    # Prepare multiple analysis tasks
    tasks = [
        tools.execute_tool("web_search", query="AI trends 2024", max_results=3),
        tools.execute_tool("web_search", query="machine learning advances", max_results=3),
        tools.execute_tool("calculator", expression="log(1000) + sqrt(144)"),
        tools.execute_tool("text_analysis", text="AI is transforming industries", analysis_type="sentiment")
    ]
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} completed: {type(result).__name__}")
    
    print("✅ Parallel analysis workflow completed")
```

