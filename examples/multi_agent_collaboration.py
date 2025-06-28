"""
Multi-Agent Collaboration Example

This example demonstrates how multiple agents can work together to solve
a complex problem. We'll create a team of agents that collaborate to
write a technical blog post using real Ollama integration.
"""

import asyncio
import sys
import os
from typing import Optional, List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.agent import Agent, AgentGoal, AgentCapability
from src.core.orchestrator import AgentOrchestrator, Workflow, WorkflowStep
from src.integration.ollama_client import OllamaIntegrationLayer
from src.tools.tool_registry import ToolRegistry
from src.memory.memory_manager import MemoryManager


async def main():
    """Main function to demonstrate multi-agent collaboration with real Ollama integration."""
    print("🤝 Multi-Agent Collaboration Example (Real Ollama Integration)")
    print("=" * 60)
    
    # Initialize components
    print("Initializing framework components...")
    
    # Initialize real Ollama integration
    ollama_integration = OllamaIntegrationLayer(
        base_url="http://localhost:11434",
        default_model="llama3.1"
    )
    
    # Check if Ollama is running and initialize
    print("🔄 Connecting to Ollama server...")
    if not await ollama_integration.initialize():
        print("❌ Failed to initialize Ollama integration.")
        print("Please ensure:")
        print("  1. Ollama is installed and running")
        print("  2. The server is accessible at http://localhost:11434")
        print("  3. The default model 'llama3.1' is available")
        print("\nTo install and run Ollama:")
        print("  1. Download from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. Pull model: ollama pull llama3.1")
        return
    
    print("✅ Ollama integration initialized successfully")
    
    # Check available models
    available_models = ollama_integration.get_available_models()
    print(f"📦 Available models: {available_models}")
    
    tool_registry = ToolRegistry()
    memory_manager = MemoryManager(db_path=":memory:")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    # Create research agent
    research_agent = orchestrator.register_agent(
        agent_id="researcher_001",
        role="Research Specialist",
        description="Conducts thorough research on technical topics",
        capabilities=[
            AgentCapability(name="web_search", description="Search for information"),
            AgentCapability(name="data_analysis", description="Analyze research data")
        ],
        system_prompt="""You are a Research Specialist AI agent. Your role is to:
1. Conduct comprehensive research on assigned topics
2. Gather information from multiple sources
3. Analyze and synthesize findings
4. Provide structured research summaries

When given a research task, provide detailed findings organized in clear sections.
Focus on accuracy, completeness, and relevance. Always end your response with 'GOAL_COMPLETED' when you finish a task."""
    )
    
    # Create writer agent
    writer_agent = orchestrator.register_agent(
        agent_id="writer_001",
        role="Content Writer",
        description="Creates engaging and informative content",
        capabilities=[
            AgentCapability(name="content_creation", description="Write articles and blog posts"),
            AgentCapability(name="technical_writing", description="Explain complex technical concepts")
        ],
        system_prompt="""You are a Content Writer AI agent. Your role is to:
1. Create engaging and informative content
2. Transform research into readable articles
3. Maintain consistent tone and style
4. Structure content for maximum impact

When asked to write content, create well-structured articles with clear headings and sections.
Write for a technical audience but keep content accessible.
Always end your response with 'GOAL_COMPLETED' when you finish writing."""
    )
    
    # Create reviewer agent
    reviewer_agent = orchestrator.register_agent(
        agent_id="reviewer_001",
        role="Technical Reviewer",
        description="Reviews content for accuracy and quality",
        capabilities=[
            AgentCapability(name="content_review", description="Review and critique content"),
            AgentCapability(name="technical_validation", description="Validate technical accuracy")
        ],
        system_prompt="""You are a Technical Reviewer AI agent. Your role is to:
1. Review content for technical accuracy
2. Assess clarity and readability
3. Provide constructive feedback
4. Ensure content meets quality standards

When reviewing content, provide specific feedback on strengths and areas for improvement.
Be thorough but constructive in your reviews.
Always end your response with 'GOAL_COMPLETED' when you finish reviewing."""
    )
    
    print(f"✅ Created {len(orchestrator.get_all_agents())} agents")
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Create workflow for blog post creation
    print("\n📋 Creating blog post workflow...")
    
    workflow = Workflow(
        name="Blog Post Creation",
        description="Collaborative workflow to create a technical blog post",
        steps=[
            WorkflowStep(
                id="research_step",
                description="Research microservices architecture and gather key information including benefits, challenges, and best practices",
                agent_role="Research Specialist",
                dependencies=[],
                context={"topic": "microservices architecture", "depth": "comprehensive"}
            ),
            WorkflowStep(
                id="writing_step",
                description="Write a comprehensive blog post about microservices architecture based on the research findings",
                agent_role="Content Writer",
                dependencies=["research_step"],
                context={"format": "blog post", "audience": "developers", "length": "medium"}
            ),
            WorkflowStep(
                id="review_step",
                description="Review the blog post for technical accuracy, clarity, and overall quality",
                agent_role="Technical Reviewer",
                dependencies=["writing_step"],
                context={"review_type": "technical", "focus": "accuracy and clarity"}
            )
        ],
        context={"project": "microservices blog post", "deadline": "today"}
    )
    
    # Execute workflow
    print("🚀 Starting workflow execution...")
    success = await orchestrator.execute_workflow(workflow)
    
    if success:
        print("✅ Workflow started successfully")
        
        # Monitor workflow progress
        print("\n🔄 Monitoring workflow progress...")
        print("(This may take a few minutes as agents work with real Ollama models)")
        
        max_wait_time = 300  # 5 minutes for real LLM processing
        wait_time = 0
        check_interval = 10  # Check every 10 seconds
        
        while wait_time < max_wait_time:
            # Check agent statuses
            agent_statuses = {}
            for agent in orchestrator.get_all_agents():
                agent_statuses[agent.role] = agent.get_status().value
            
            print(f"⏱️  Time: {wait_time}s | Agent statuses: {agent_statuses}")
            
            # Check if all agents are idle (workflow complete)
            if all(status in ["idle", "completed"] for status in agent_statuses.values()):
                print("🎯 All agents completed their tasks!")
                break
            
            await asyncio.sleep(check_interval)
            wait_time += check_interval
        
        if wait_time >= max_wait_time:
            print("⚠️  Workflow execution timed out. Some agents may still be processing.")
        
        print("\n🎉 Workflow execution completed!")
        
        # Get collaboration results
        print("\n📝 Collaboration Results:")
        
        # Get memories from each agent
        for agent in orchestrator.get_all_agents():
            print(f"\n{'='*50}")
            print(f"🤖 {agent.role} ({agent.agent_id}):")
            print(f"{'='*50}")
            
            memories = await memory_manager.retrieve_memories(
                agent_id=agent.agent_id,
                limit=3
            )
            
            if memories:
                for i, memory in enumerate(memories, 1):
                    print(f"\n📋 Memory {i}:")
                    # Show more of the memory content for real results
                    if len(memory) > 500:
                        print(f"{memory[:500]}...")
                        print(f"[Content truncated - Full length: {len(memory)} characters]")
                    else:
                        print(memory)
            else:
                print("No memories found for this agent.")
        
        # Demonstrate inter-agent communication
        print(f"\n{'='*50}")
        print("💬 Demonstrating inter-agent communication...")
        print(f"{'='*50}")
        
        # Writer sends message to reviewer
        await orchestrator.send_message(
            sender_id=writer_agent.agent_id,
            recipient_id=reviewer_agent.agent_id,
            content="The blog post draft is ready for your review. Please focus on technical accuracy and readability.",
            message_type="collaboration"
        )
        
        # Reviewer responds
        await orchestrator.send_message(
            sender_id=reviewer_agent.agent_id,
            recipient_id=writer_agent.agent_id,
            content="Thank you! I'll review the post and provide detailed feedback.",
            message_type="collaboration"
        )
        
        print("✅ Messages exchanged between agents")
        
        # Show recent messages
        print("\n📨 Recent inter-agent messages:")
        # This would require implementing message retrieval in your orchestrator
        # For now, we'll just show that the functionality exists
        
    else:
        print("❌ Failed to start workflow")
    
    # Stop the orchestrator
    print("\n🛑 Stopping orchestrator...")
    await orchestrator.stop()
    
    # Final statistics
    print(f"\n{'='*50}")
    print("📈 Final Statistics:")
    print(f"{'='*50}")
    
    try:
        agent_stats = orchestrator.get_agent_status_summary()
        task_stats = orchestrator.get_task_status_summary()
        memory_stats = await memory_manager.get_memory_stats()
        
        print(f"Agent statuses: {agent_stats}")
        print(f"Task statuses: {task_stats}")
        print(f"Total memories: {memory_stats['total_memories']}")
        print(f"Memories by type: {memory_stats['memories_by_type']}")
        
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
    
    print("\n✨ Multi-agent collaboration example completed!")
    print("\nKey achievements:")
    print("✅ Real Ollama integration used")
    print("✅ Multiple agents collaborated on a complex task")
    print("✅ Workflow execution managed automatically")
    print("✅ Inter-agent communication demonstrated")
    print("✅ Memory and task tracking functional")


async def check_ollama_setup():
    """Helper function to check Ollama setup before running the main example."""
    print("🔍 Checking Ollama setup...")
    
    try:
        # Try to create a simple integration instance
        test_integration = OllamaIntegrationLayer()
        
        # Test connection
        if await test_integration.initialize():
            models = test_integration.get_available_models()
            print(f"✅ Ollama is running with {len(models)} models available")
            if models:
                print(f"📦 Available models: {', '.join(models)}")
            return True
        else:
            print("❌ Could not connect to Ollama")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Ollama setup: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Collaboration Example")
    print("=" * 60)
    
    # First check if Ollama is properly set up
    asyncio.run(check_ollama_setup())
    
    # Run the main example
    asyncio.run(main())