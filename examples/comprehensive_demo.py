import asyncio
import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
logging.info(f"Added project root to path: {project_root}")

from src.edgebrain.core.agent import Agent, AgentGoal, AgentCapability
from src.edgebrain.core.orchestrator import AgentOrchestrator
from src.edgebrain.integration.ollama_client import OllamaIntegrationLayer, OllamaResponse
from src.edgebrain.tools.tool_registry import ToolRegistry, BaseTool
from src.edgebrain.memory.memory_manager import MemoryManager

async def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")

async def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-' * 30}\n{title}\n{'-' * 30}")

class AnalyticsDashboardTool(BaseTool):
    """Custom tool for generating analytics dashboards."""
    
    def __init__(self):
        super().__init__(
            name="analytics_dashboard",
            description="Generates an analytics dashboard with charts and KPIs",
            category="analytics"
        )
    
    async def execute(self, **kwargs) -> dict:
        """Execute the dashboard generation tool."""
        data_type = kwargs.get("data_type", "unknown")
        metrics = kwargs.get("metrics", [])
        # Simulate dashboard generation
        charts = [f"Chart_{i+1}" for i in range(len(metrics))]
        kpis = {metric: f"Value_{i+1}" for i, metric in enumerate(metrics)}
        return {
            "success": True,
            "message": f"Generated dashboard for {data_type}",
            "dashboard": {"charts": charts, "kpis": kpis}
        }

async def demonstrate_core_components(orchestrator: AgentOrchestrator):
    """Demonstrate core framework components."""
    await print_section_header("CORE COMPONENTS")
    
    await print_subsection("Agent Creation and Registration")
    try:
        research_agent = orchestrator.register_agent(
            agent_id="research_001",
            role="Research Specialist",
            description="Conducts research and gathers insights",
            capabilities=[
                AgentCapability(name="web_search", description="Search the web for information"),
                AgentCapability(name="data_analysis", description="Analyze research data")
            ],
            system_prompt="""You are a Research Specialist AI agent. Your role is to:
1. Conduct thorough research on given topics
2. Analyze and summarize findings
3. Provide accurate and concise insights"""
        )
        logging.info(f"Created research agent: {research_agent.agent_id}")
        print(f"✅ Created research agent: {research_agent.agent_id}")
    except Exception as e:
        logging.error(f"Failed to create research agent: {e}")
        print(f"❌ Failed to create research agent: {e}")
        return
    
    await print_subsection("Task Creation and Assignment")
    try:
        task_id = await orchestrator.create_task(
            description="Research recent advancements in AI",
            priority=5,
            context={"domain": "AI", "focus": "recent advancements"}
        )
        print(f"✅ Created task: {task_id}")
        
        success = await orchestrator.assign_task_to_agent(task_id, research_agent.agent_id)
        if success:
            print("✅ Task assigned successfully")
        else:
            print("❌ Failed to assign task")
    except Exception as e:
        logging.error(f"Task creation/assignment error: {e}")
        print(f"❌ Task creation/assignment error: {e}")

async def demonstrate_multi_agent_workflow(orchestrator: AgentOrchestrator):
    """Demonstrate multi-agent collaboration workflow."""
    await print_section_header("MULTI-AGENT WORKFLOW")
    
    await print_subsection("Creating Collaborative Agents")
    try:
        writer_agent = orchestrator.register_agent(
            agent_id="writer_001",
            role="Content Writer",
            description="Creates content based on research",
            capabilities=[AgentCapability(name="content_creation", description="Write articles and reports")],
            system_prompt="You are a Content Writer AI agent. Create clear and engaging content based on research inputs."
        )
        reviewer_agent = orchestrator.register_agent(
            agent_id="reviewer_001",
            role="Technical Reviewer",
            description="Reviews content for accuracy",
            capabilities=[AgentCapability(name="content_review", description="Review and validate content")],
            system_prompt="You are a Technical Reviewer AI agent. Ensure content accuracy and suggest improvements."
        )
        print(f"✅ Created agents: {writer_agent.agent_id}, {reviewer_agent.agent_id}")
        logging.info(f"Created agents: {writer_agent.agent_id}, {reviewer_agent.agent_id}")
    except Exception as e:
        logging.error(f"Failed to create collaborative agents: {e}")
        print(f"❌ Failed to create collaborative agents: {e}")
        return
    
    await print_subsection("Creating and Executing Sequential Tasks")
    try:
        # Create first task for writer
        write_task_id = await orchestrator.create_task(
            description="Create an article on AI ethics - focus on key principles and challenges",
            priority=3,
            context={"topic": "AI ethics", "output": "article", "role": "writer"}
        )
        print(f"✅ Created writing task: {write_task_id}")
        
        # Create second task for reviewer
        review_task_id = await orchestrator.create_task(
            description="Review the AI ethics article for accuracy and completeness",
            priority=3,
            context={"topic": "AI ethics", "output": "review", "role": "reviewer"}
        )
        print(f"✅ Created review task: {review_task_id}")
        
        # Assign tasks to respective agents
        write_success = await orchestrator.assign_task_to_agent(write_task_id, writer_agent.agent_id)
        review_success = await orchestrator.assign_task_to_agent(review_task_id, reviewer_agent.agent_id)
        
        if write_success and review_success:
            print("✅ Tasks assigned to agents")
            
            await print_subsection("Monitoring Task Execution")
            max_wait_time = 60
            wait_time = 0
            
            while wait_time < max_wait_time:
                write_task = orchestrator.get_task(write_task_id)
                review_task = orchestrator.get_task(review_task_id)
                
                if write_task and review_task:
                    print(f"⏱️ Time: {wait_time}s | Write Task: {write_task.status.value} | Review Task: {review_task.status.value}")
                    
                    writer = orchestrator.get_agent(writer_agent.agent_id)
                    reviewer = orchestrator.get_agent(reviewer_agent.agent_id)
                    
                    if writer:
                        print(f"   • Writer Agent: {writer.get_status().value}")
                    if reviewer:
                        print(f"   • Reviewer Agent: {reviewer.get_status().value}")
                    
                    if write_task.status.value in ["completed", "failed"] and review_task.status.value in ["completed", "failed"]:
                        break
                else:
                    print(f"⏱️ Time: {wait_time}s | Tasks not found")
                
                await asyncio.sleep(2)
                wait_time += 2
            
            final_write_task = orchestrator.get_task(write_task_id)
            final_review_task = orchestrator.get_task(review_task_id)
            
            if final_write_task:
                print(f"\n📊 Final write task status: {final_write_task.status.value}")
            if final_review_task:
                print(f"📊 Final review task status: {final_review_task.status.value}")
        else:
            print("❌ Failed to assign tasks to agents")
    except Exception as e:
        logging.error(f"Task execution error: {e}")
        print(f"❌ Task execution error: {e}")

async def demonstrate_tool_usage(orchestrator: AgentOrchestrator):
    """Demonstrate tool usage and extensibility."""
    await print_section_header("TOOL USAGE AND EXTENSIBILITY")
    
    await print_subsection("Tool Categories")
    try:
        categories = orchestrator.tool_registry.get_categories()
        for category in categories:
            tools = orchestrator.tool_registry.get_tools_by_category(category)
            print(f"📂 {category.title()}: {len(tools)} tools")
            for tool in tools:
                print(f"   • {tool.name}: {tool.description}")
    except Exception as e:
        logging.error(f"Failed to retrieve tool categories: {e}")
        print(f"❌ Failed to retrieve tool categories: {e}")
    
    await print_subsection("Built-in Tool Demonstration")
    try:
        # Test calculator tool
        calc_result = await orchestrator.tool_registry.execute_tool(
            "calculator",
            {"expression": "2 + 2 * 3"}
        )
        if calc_result.get("success"):
            print(f"✅ Calculator: {calc_result['expression']} = {calc_result['result']}")
        else:
            print(f"❌ Calculator failed: {calc_result.get('error')}")
        
        # Test web search tool
        search_result = await orchestrator.tool_registry.execute_tool(
            "web_search",
            {"query": "artificial intelligence", "max_results": 2}
        )
        if search_result:
            print(f"✅ Web Search: Found {len(search_result['results'])} results for '{search_result['query']}'")
            for i, result in enumerate(search_result['results'], 1):
                print(f"   {i}. {result['title']}")
        else:
            print("❌ Web search failed")
            
    except Exception as e:
        logging.error(f"Built-in tool execution error: {e}")
        print(f"❌ Error executing built-in tools: {e}")
    
    await print_subsection("Custom Tool Demonstration")
    try:
        result = await orchestrator.tool_registry.execute_tool(
            "analytics_dashboard",
            {"data_type": "AI Market", "metrics": ["growth", "adoption", "investment"]}
        )
        if result["success"]:
            dashboard = result["dashboard"]
            print(f"✅ {result['message']}")
            print(f"📊 Generated {len(dashboard['charts'])} charts")
            print(f"📈 KPIs: {dashboard['kpis']}")
        else:
            print(f"❌ Tool execution failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logging.error(f"Tool execution error: {e}")
        print(f"❌ Error executing tool: {e}")

async def demonstrate_agent_tool_interaction(orchestrator: AgentOrchestrator):
    """Demonstrate how agents can use tools."""
    await print_section_header("AGENT-TOOL INTERACTION")
    
    await print_subsection("Creating Tool-Enabled Agent")
    try:
        analyst_agent = orchestrator.register_agent(
            agent_id="analyst_001",
            role="Data Analyst",
            description="Analyzes data and creates reports using various tools",
            capabilities=[
                AgentCapability(name="calculator", description="Perform calculations"),
                AgentCapability(name="analytics_dashboard", description="Create dashboards"),
                AgentCapability(name="file_write", description="Save reports to files")
            ],
            system_prompt="""You are a Data Analyst AI agent. Use available tools to:
1. Perform calculations and data analysis
2. Create visualizations and dashboards
3. Generate and save reports"""
        )
        print(f"✅ Created analyst agent: {analyst_agent.agent_id}")
        
        # Create a task that requires tool usage
        analysis_task_id = await orchestrator.create_task(
            description="Calculate growth metrics and create a dashboard for Q4 performance",
            priority=4,
            context={
                "metrics": ["revenue", "users", "conversion"],
                "period": "Q4",
                "tools_required": ["calculator", "analytics_dashboard"]
            }
        )
        print(f"✅ Created analysis task: {analysis_task_id}")
        
        success = await orchestrator.assign_task_to_agent(analysis_task_id, analyst_agent.agent_id)
        if success:
            print("✅ Analysis task assigned successfully")
        else:
            print("❌ Failed to assign analysis task")
            
    except Exception as e:
        logging.error(f"Agent-tool interaction setup error: {e}")
        print(f"❌ Agent-tool interaction setup error: {e}")

async def main():
    """Main function to run the comprehensive demo."""
    print("🚀 Ollama Agentic Framework Comprehensive Demo")
    print("=" * 50)
    
    # Initialize components
    print("Initializing framework components...")
    try:
        ollama_integration = OllamaIntegrationLayer(default_model="llama3.1")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if await ollama_integration.initialize():
                    logging.info("Ollama integration initialized successfully")
                    break
                else:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed to initialize Ollama. Retrying...")
                    time.sleep(5)  # Wait before retrying
                    if attempt == max_retries - 1:
                        raise Exception("Failed to initialize Ollama after retries")
            except Exception as e:
                logging.error(f"Ollama initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to initialize Ollama integration after {max_retries} attempts: {e}")
                    return
                time.sleep(5)
    except Exception as e:
        print(f"❌ Critical error initializing Ollama integration: {e}")
        return
    
    tool_registry = ToolRegistry()
    try:
        tool_registry.register_tool(AnalyticsDashboardTool())
        logging.info("Registered AnalyticsDashboardTool")
    except Exception as e:
        logging.error(f"Failed to register tool: {e}")
        print(f"❌ Failed to register tool: {e}")
        return
    
    memory_manager = MemoryManager(db_path=":memory:")
    
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    # Start orchestrator
    try:
        await orchestrator.start()
        logging.info("Orchestrator started")
        print("✅ Orchestrator started successfully")
    except Exception as e:
        logging.error(f"Failed to start orchestrator: {e}")
        print(f"❌ Failed to start orchestrator: {e}")
        return
    
    # Run demonstrations
    try:
        await demonstrate_core_components(orchestrator)
        await demonstrate_multi_agent_workflow(orchestrator)
        await demonstrate_tool_usage(orchestrator)
        await demonstrate_agent_tool_interaction(orchestrator)
    except Exception as e:
        logging.error(f"Error during demonstrations: {e}")
        print(f"❌ Error during demonstrations: {e}")
    
    # Monitor memory
    try:
        await print_section_header("MEMORY MANAGEMENT")
        memories = await memory_manager.retrieve_memories(agent_id="research_001", limit=5)
        print(f"📚 Retrieved {len(memories)} memories for research_001")
        for i, memory in enumerate(memories, 1):
            memory_str = str(memory)[:100] if memory else "Empty memory"
            print(f"Memory {i}: {memory_str}...")
    except Exception as e:
        logging.error(f"Memory retrieval error: {e}")
        print(f"❌ Memory retrieval error: {e}")
    
    # Stop orchestrator
    try:
        await orchestrator.stop()
        logging.info("Orchestrator stopped")
        print("✅ Orchestrator stopped successfully")
    except Exception as e:
        logging.error(f"Failed to stop orchestrator: {e}")
        print(f"❌ Failed to stop orchestrator: {e}")
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            logging.error(f"Main execution error: {e}")
            print(f"❌ Main execution error: {e}")
            raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")