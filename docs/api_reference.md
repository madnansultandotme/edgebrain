# API Reference

This document provides a comprehensive reference for all classes, methods, and functions in the EdgeBrain framework available from PyPI.

## Installation

Before using any of these APIs, ensure you have EdgeBrain installed:

```bash
pip install edgebrain
pip install ollama  # For async client support
```

## Import Statements

```python
# Core components
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import Agent, AgentGoal, AgentCapability, AgentMessage, AgentStatus
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry, BaseTool, Tool
from edgebrain.memory.memory_manager import MemoryManager

# For direct async code generation
import ollama
```

## Table of Contents

- [Agent Types](#agent-types)
- [Available Tools](#available-tools)
- [Core Components](#core-components)
  - [AgentOrchestrator](#agentorchestrator)
  - [Agent](#agent)
  - [AgentCapability](#agentcapability)
- [Integration](#integration)
  - [OllamaIntegrationLayer](#ollamaintegrationlayer)
  - [AsyncClient Usage](#asyncclient-usage)
- [Tools](#tools)
  - [ToolRegistry](#toolregistry)
  - [BaseTool](#basetool)
  - [Built-in Tools](#built-in-tools)
- [Memory](#memory)
  - [MemoryManager](#memorymanager)
- [Data Models](#data-models)

## Agent Types

EdgeBrain supports building various types of specialized agents:

### 1. Research Agents
- **Purpose**: Information gathering, analysis, and synthesis
- **Capabilities**: Web searching, data analysis, report generation
- **Tools**: WebSearchTool, TextAnalysisTool, KnowledgeBaseTool
- **Use Cases**: Market research, academic research, competitive analysis

### 2. Code Generation Agents
- **Purpose**: Software development, code analysis, and automation
- **Capabilities**: Code generation, debugging, testing, documentation
- **Tools**: FileWriteTool, FileReadTool, ShellCommandTool
- **Use Cases**: API development, test generation, code review

### 3. Data Analysis Agents
- **Purpose**: Processing and analyzing structured/unstructured data
- **Capabilities**: Statistical analysis, pattern recognition, visualization
- **Tools**: CalculatorTool, DataStorageTool, TextAnalysisTool
- **Use Cases**: Business intelligence, data mining, reporting

### 4. Communication Agents
- **Purpose**: Managing interactions between agents and external systems
- **Capabilities**: Message routing, protocol translation, notification management
- **Tools**: WebSearchTool, KnowledgeBaseTool
- **Use Cases**: Chatbots, notification systems, workflow coordination

### 5. Planning & Orchestration Agents
- **Purpose**: Task planning, resource allocation, and workflow management
- **Capabilities**: Strategic planning, resource optimization, coordination
- **Tools**: All tools for comprehensive planning
- **Use Cases**: Project management, resource planning, process automation

### 6. Specialized Domain Agents
- **Purpose**: Domain-specific expertise (finance, healthcare, legal, etc.)
- **Capabilities**: Domain knowledge application, compliance checking
- **Tools**: Custom domain-specific tools + built-in tools
- **Use Cases**: Financial analysis, medical diagnosis support, legal research

## Available Tools

EdgeBrain includes a comprehensive set of built-in tools that agents can use:

### Research & Information Tools
- **WebSearchTool**: Perform web searches and extract information
- **KnowledgeBaseTool**: Store and retrieve knowledge from a structured database
- **TextAnalysisTool**: Analyze text for sentiment, keywords, and insights

### File & Data Management Tools  
- **FileWriteTool**: Create and write files with various formats
- **FileReadTool**: Read and parse files (text, JSON, CSV, etc.)
- **DataStorageTool**: Store and retrieve structured data

### Computation & Analysis Tools
- **CalculatorTool**: Perform mathematical calculations and evaluations
- **ShellCommandTool**: Execute system commands and scripts

### Example: Creating a Custom Tool

```python
from edgebrain.tools.tool_registry import BaseTool

class DatabaseQueryTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="database_query",
            description="Execute SQL queries against a database",
            category="data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query", "database"]
            }
        )
    
    async def execute(self, query: str, database: str) -> dict:
        # Implementation here
        return {"result": "query results"}
```

## Core Components

### AgentOrchestrator

The central control unit for managing agents and coordinating their interactions.

#### Class Definition

```python
class AgentOrchestrator:
    def __init__(
        self,
        ollama_integration: OllamaIntegrationLayer,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager
    )
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ollama_integration` | `OllamaIntegrationLayer` | Ollama integration layer instance |
| `tool_registry` | `ToolRegistry` | Tool registry for agents |
| `memory_manager` | `MemoryManager` | Memory management system |

#### Methods

##### `async start() -> None`

Start the orchestrator and begin processing tasks and messages.

**Example:**
```python
await orchestrator.start()
```

##### `async stop() -> None`

Stop the orchestrator and clean up resources.

**Example:**
```python
await orchestrator.stop()
```

##### `register_agent(...) -> Agent`

Register a new agent with the orchestrator.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | `str` | Yes | Unique identifier for the agent |
| `role` | `str` | Yes | Role of the agent |
| `description` | `str` | Yes | Description of the agent's purpose |
| `model` | `str` | No | Specific model to use for the agent |
| `capabilities` | `List[AgentCapability]` | No | List of agent capabilities |
| `system_prompt` | `str` | No | Custom system prompt for the agent |

**Returns:** `Agent` - The created agent instance

**Example:**
```python
agent = orchestrator.register_agent(
    agent_id="researcher_001",
    role="Research Specialist",
    description="Conducts thorough research on technical topics",
    model="llama3.1",
    capabilities=[
        AgentCapability(name="web_search", description="Search for information"),
        AgentCapability(name="data_analysis", description="Analyze research data")
    ]
)
```

##### `unregister_agent(agent_id: str) -> bool`

Unregister an agent from the orchestrator.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | ID of the agent to unregister |

**Returns:** `bool` - True if agent was unregistered, False if not found

##### `async create_task(...) -> str`

Create a new task for execution.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | `str` | Yes | Description of the task |
| `priority` | `int` | No | Priority of the task (1-10, default: 1) |
| `context` | `Dict[str, Any]` | No | Additional context for the task |

**Returns:** `str` - ID of the created task

**Example:**
```python
task_id = await orchestrator.create_task(
    description="Research current trends in artificial intelligence",
    priority=5,
    context={"topic": "AI trends", "depth": "comprehensive"}
)
```

##### `async assign_task_to_agent(task_id: str, agent_id: str) -> bool`

Assign a task to a specific agent.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | ID of the task |
| `agent_id` | `str` | ID of the agent |

**Returns:** `bool` - True if assignment successful, False otherwise

##### `async assign_task_to_role(task_id: str, role: str) -> bool`

Assign a task to an agent with a specific role.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | ID of the task |
| `role` | `str` | Role of the agent to assign to |

**Returns:** `bool` - True if assignment successful, False otherwise

##### `async execute_workflow(workflow: Workflow) -> bool`

Execute a workflow with multiple steps.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `workflow` | `Workflow` | Workflow to execute |

**Returns:** `bool` - True if workflow started successfully, False otherwise

##### `async send_message(...) -> None`

Send a message between agents.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `sender_id` | `str` | ID of the sender |
| `recipient_id` | `str` | ID of the recipient |
| `content` | `str` | Content of the message |
| `message_type` | `str` | Type of the message (default: "text") |

##### `get_agent(agent_id: str) -> Optional[Agent]`

Get an agent by ID.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | ID of the agent |

**Returns:** `Optional[Agent]` - Agent instance or None if not found

##### `get_agents_by_role(role: str) -> List[Agent]`

Get all agents with a specific role.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | `str` | Role to search for |

**Returns:** `List[Agent]` - List of agents with the specified role

##### `get_all_agents() -> List[Agent]`

Get all registered agents.

**Returns:** `List[Agent]` - List of all agents

##### `get_task(task_id: str) -> Optional[Task]`

Get a task by ID.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | ID of the task |

**Returns:** `Optional[Task]` - Task instance or None if not found

##### `get_all_tasks() -> List[Task]`

Get all tasks.

**Returns:** `List[Task]` - List of all tasks

##### `get_agent_status_summary() -> Dict[str, int]`

Get a summary of agent statuses.

**Returns:** `Dict[str, int]` - Dictionary mapping status names to counts

##### `get_task_status_summary() -> Dict[str, int]`

Get a summary of task statuses.

**Returns:** `Dict[str, int]` - Dictionary mapping status names to counts

### Agent

The core Agent class represents an autonomous entity capable of performing specific tasks.

#### Class Definition

```python
class Agent:
    def __init__(
        self,
        agent_id: str,
        role: str,
        description: str,
        ollama_integration: OllamaIntegrationLayer,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        model: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    )
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | `str` | Unique identifier for the agent |
| `role` | `str` | Role of the agent (e.g., "Researcher", "Coder", "Analyst") |
| `description` | `str` | Description of the agent's purpose |
| `ollama_integration` | `OllamaIntegrationLayer` | LLM integration layer |
| `tool_registry` | `ToolRegistry` | Available tools for the agent |
| `memory_manager` | `MemoryManager` | Memory management system |
| `model` | `Optional[str]` | Specific model to use (defaults to integration default) |
| `capabilities` | `Optional[List[AgentCapability]]` | List of agent capabilities |
| `system_prompt` | `Optional[str]` | Custom system prompt |
| `max_iterations` | `int` | Maximum iterations for task execution |

#### Key Methods

```python
async def set_goal(self, goal: AgentGoal) -> None
async def process_message(self, message: AgentMessage) -> None  
async def execute_goal(self) -> bool
async def think(self, context: str) -> str
async def use_tool(self, tool_name: str, **kwargs) -> Any
```

#### Agent Status Enum

```python
class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
```

#### Example: Creating Specialized Agents

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.core.agent import Agent, AgentCapability
from edgebrain.integration.ollama_client import OllamaIntegrationLayer
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager

async def create_research_agent():
    # Initialize components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    tools = ToolRegistry()
    memory = MemoryManager()
    
    orchestrator = AgentOrchestrator(ollama, tools, memory)
    
    # Define research capabilities
    research_capabilities = [
        AgentCapability(
            name="web_search",
            description="Search the web for information",
            parameters={"search_depth": "comprehensive"}
        ),
        AgentCapability(
            name="data_analysis", 
            description="Analyze and synthesize research data",
            parameters={"analysis_type": "qualitative"}
        ),
        AgentCapability(
            name="report_generation",
            description="Generate structured research reports",
            parameters={"format": "markdown"}
        )
    ]
    
    # Create research agent
    research_agent = orchestrator.register_agent(
        agent_id="research_specialist",
        role="Research Specialist",
        description="Expert in conducting comprehensive research and analysis",
        capabilities=research_capabilities,
        model="llama3.1",  # Use specific model for research tasks
        system_prompt="""You are a research specialist with expertise in:
        - Information gathering and verification
        - Data analysis and pattern recognition  
        - Report writing and documentation
        Always cite sources and provide evidence-based conclusions."""
    )
    
    return research_agent

async def create_code_agent():
    # Code generation capabilities
    code_capabilities = [
        AgentCapability(
            name="code_generation",
            description="Generate clean, functional code",
            parameters={"languages": ["python", "javascript", "sql"]}
        ),
        AgentCapability(
            name="code_review",
            description="Review and improve existing code",
            parameters={"focus": ["security", "performance", "readability"]}
        ),
        AgentCapability(
            name="testing",
            description="Generate and execute tests",
            parameters={"test_types": ["unit", "integration"]}
        )
    ]
    
    # Create code agent with qwen2.5:3b for fast code generation
    code_agent = orchestrator.register_agent(
        agent_id="code_specialist", 
        role="Code Specialist",
        description="Expert in software development and code generation",
        capabilities=code_capabilities,
        model="qwen2.5:3b",  # Fast model for code generation
        system_prompt="""You are a senior software engineer with expertise in:
        - Writing clean, efficient, and maintainable code
        - Following best practices and design patterns
        - Comprehensive testing and documentation
        Always include error handling and clear documentation."""
    )
    
    return code_agent
```

### AgentCapability

Represents a specific capability that an agent possesses.

#### Class Definition

```python
class AgentCapability(BaseModel):
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what the capability does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the capability")
```

#### Example Capabilities by Agent Type

**Research Agent Capabilities:**
- `information_gathering`: Web search and data collection
- `source_verification`: Fact-checking and source validation
- `data_synthesis`: Combining information from multiple sources
- `report_generation`: Creating structured research reports

**Code Agent Capabilities:**
- `code_generation`: Creating new code from specifications
- `code_review`: Analyzing and improving existing code
- `debugging`: Identifying and fixing code issues
- `test_generation`: Creating unit and integration tests
- `documentation`: Generating code documentation

**Data Analysis Agent Capabilities:**
- `statistical_analysis`: Performing statistical computations
- `data_visualization`: Creating charts and graphs
- `pattern_recognition`: Identifying trends and patterns
- `predictive_modeling`: Building predictive models

**Communication Agent Capabilities:**
- `message_routing`: Directing messages between agents
- `protocol_translation`: Converting between different formats
- `notification_management`: Handling alerts and notifications
- `api_integration`: Connecting with external services

## Tools

### ToolRegistry

Central registry for managing all available tools that agents can use.

#### Class Definition

```python
class ToolRegistry:
    def __init__(self):
        """Initialize the tool registry with built-in tools."""
```

#### Key Methods

```python
def register_tool(self, tool: BaseTool) -> None
def get_tool(self, name: str) -> Optional[BaseTool]
def get_all_tools(self) -> List[BaseTool]
def get_tools_by_category(self, category: str) -> List[BaseTool]
async def execute_tool(self, name: str, **kwargs) -> Any
def get_tool_count(self) -> int
```

#### Usage Example

```python
from edgebrain.tools.tool_registry import ToolRegistry

# Initialize registry
tools = ToolRegistry()

# Get available tools
all_tools = tools.get_all_tools()
print(f"Available tools: {[tool.name for tool in all_tools]}")

# Use a tool
result = await tools.execute_tool("web_search", query="AI trends 2024", max_results=3)
```

### BaseTool

Abstract base class for creating custom tools.

#### Class Definition

```python
class BaseTool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        category: str = "general",
        parameters: Dict[str, Any] = None
    )
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
```

### Built-in Tools

EdgeBrain includes a comprehensive set of built-in tools:

#### 1. WebSearchTool
- **Purpose**: Perform web searches using DuckDuckGo API
- **Category**: information
- **Parameters**:
  - `query` (str): Search query
  - `max_results` (int): Maximum results to return (default: 5)

```python
# Usage example
result = await tools.execute_tool("web_search", 
    query="machine learning trends 2024", 
    max_results=5
)
```

#### 2. CalculatorTool
- **Purpose**: Perform mathematical calculations and evaluations
- **Category**: computation
- **Parameters**:
  - `expression` (str): Mathematical expression to evaluate

```python
# Usage example
result = await tools.execute_tool("calculator", 
    expression="sqrt(144) + 2 * pi"
)
```

#### 3. FileWriteTool
- **Purpose**: Create and write files with various formats
- **Category**: file_management
- **Parameters**:
  - `filename` (str): Name of the file to create
  - `content` (str): Content to write
  - `mode` (str): Write mode (default: 'w')

```python
# Usage example
result = await tools.execute_tool("file_write",
    filename="research_report.md",
    content="# Research Results\n\nKey findings...",
    mode="w"
)
```

#### 4. FileReadTool  
- **Purpose**: Read and parse files (text, JSON, CSV, etc.)
- **Category**: file_management
- **Parameters**:
  - `filename` (str): Name of the file to read
  - `format` (str): File format (auto-detected if not specified)

```python
# Usage example
result = await tools.execute_tool("file_read",
    filename="data.json",
    format="json"
)
```

#### 5. TextAnalysisTool
- **Purpose**: Analyze text for sentiment, keywords, and insights
- **Category**: analysis
- **Parameters**:
  - `text` (str): Text to analyze
  - `analysis_type` (str): Type of analysis (sentiment, keywords, summary)

```python
# Usage example  
result = await tools.execute_tool("text_analysis",
    text="This is a great product with excellent features!",
    analysis_type="sentiment"
)
```

#### 6. DataStorageTool
- **Purpose**: Store and retrieve structured data
- **Category**: data
- **Parameters**:
  - `action` (str): Action to perform (store, retrieve, delete)
  - `key` (str): Data key
  - `data` (Any): Data to store (for store action)

```python
# Usage example
result = await tools.execute_tool("data_storage",
    action="store",
    key="research_findings",
    data={"topic": "AI trends", "confidence": 0.95}
)
```

#### 7. ShellCommandTool
- **Purpose**: Execute system commands and scripts
- **Category**: system
- **Parameters**:
  - `command` (str): Command to execute
  - `timeout` (int): Timeout in seconds (default: 30)

```python
# Usage example (use with caution)
result = await tools.execute_tool("shell_command",
    command="ls -la",
    timeout=10
)
```

#### 8. KnowledgeBaseTool
- **Purpose**: Store and retrieve knowledge from a structured database
- **Category**: knowledge
- **Parameters**:
  - `action` (str): Action (store, search, retrieve, update)
  - `query` (str): Search query (for search action)
  - `knowledge` (dict): Knowledge to store (for store action)

```python
# Usage example
result = await tools.execute_tool("knowledge_base",
    action="search",
    query="machine learning algorithms"
)
```

### Creating Custom Tools

#### Example: Custom API Integration Tool

```python
from edgebrain.tools.tool_registry import BaseTool
import aiohttp

class WeatherAPITool(BaseTool):
    def __init__(self, api_key: str):
        super().__init__(
            name="weather_api",
            description="Get current weather information for a location",
            category="api",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string", 
                        "description": "Temperature units (metric/imperial)",
                        "default": "metric"
                    }
                },
                "required": ["location"]
            }
        )
        self.api_key = api_key
    
    async def execute(self, location: str, units: str = "metric") -> dict:
        """Get weather data from API."""
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
                        "location": location,
                        "temperature": data["main"]["temp"],
                        "description": data["weather"][0]["description"],
                        "humidity": data["main"]["humidity"],
                        "success": True
                    }
                else:
                    return {
                        "location": location,
                        "error": f"API error: {response.status}",
                        "success": False
                    }

# Register the custom tool
tools = ToolRegistry()
weather_tool = WeatherAPITool(api_key="your_api_key")
tools.register_tool(weather_tool)
```

#### Example: Database Query Tool

```python
from edgebrain.tools.tool_registry import BaseTool
import sqlite3
import aiosqlite

class DatabaseQueryTool(BaseTool):
    def __init__(self, db_path: str):
        super().__init__(
            name="database_query",
            description="Execute SQL queries against a SQLite database",
            category="data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "fetch_mode": {
                        "type": "string",
                        "description": "Fetch mode: all, one, or count",
                        "default": "all"
                    }
                },
                "required": ["query"]
            }
        )
        self.db_path = db_path
    
    async def execute(self, query: str, fetch_mode: str = "all") -> dict:
        """Execute SQL query."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query)
                
                if fetch_mode == "all":
                    results = await cursor.fetchall()
                elif fetch_mode == "one":
                    results = await cursor.fetchone()
                else:  # count
                    results = len(await cursor.fetchall())
                
                return {
                    "query": query,
                    "results": results,
                    "success": True
                }
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "success": False
            }
```

### Tool Categories

Tools are organized into categories for better discovery and management:

- **information**: Web search, knowledge base, research tools
- **computation**: Calculator, data analysis, mathematical tools
- **file_management**: File read/write, document processing
- **data**: Database operations, data storage, retrieval
- **analysis**: Text analysis, sentiment analysis, pattern recognition
- **system**: Shell commands, system monitoring, automation
- **api**: External API integrations, web services
- **communication**: Messaging, notifications, protocol handling
- **security**: Encryption, authentication, access control
- **monitoring**: Logging, metrics, performance tracking

