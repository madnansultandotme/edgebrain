"""
Simple Research Agent Example

This example demonstrates how to create a basic research agent using the
Ollama Agentic Framework. The agent can search for information and summarize findings.
"""

import asyncio
import sys
import os
import datetime
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.agent import Agent, AgentGoal, AgentCapability
from src.core.orchestrator import AgentOrchestrator
from src.integration.ollama_client import OllamaIntegrationLayer
from src.tools.tool_registry import ToolRegistry
from src.memory.memory_manager import MemoryManager



async def main():
    """Main function to demonstrate the research agent."""
    print("🔬 Simple Research Agent Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing framework components...")
    
    # Use mock Ollama integration for demonstration
    ollama_integration = OllamaIntegrationLayer()
    await ollama_integration.initialize()
    
    tool_registry = ToolRegistry()
    memory_manager = MemoryManager(db_path=":memory:")  # Use in-memory database
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    # Define research capabilities
    research_capabilities = [
        AgentCapability(
            name="web_search",
            description="Search the web for information on various topics"
        ),
        AgentCapability(
            name="information_synthesis",
            description="Analyze and synthesize information from multiple sources"
        ),
        AgentCapability(
            name="report_generation",
            description="Generate comprehensive reports based on research findings"
        )
    ]
    
    # Create research agent with optimized settings
    research_agent = orchestrator.register_agent(
        agent_id="researcher_001",
        role="Research Specialist",
        description="An AI agent specialized in conducting research and generating insights",
        capabilities=research_capabilities,
        model="llama3.1:latest",
        system_prompt="""You are a research specialist AI agent. Your goal is to quickly research and summarize information.

IMPORTANT: Work efficiently and complete your research in 3-5 iterations maximum.

Process:
1. Use web_search tool to find information on the topic
2. Analyze the results briefly
3. Generate a concise summary
4. End with "GOAL_COMPLETED" when done

Be concise but thorough. Focus on key findings and trends."""
    )
    
    print(f"✅ Created research agent: {research_agent.agent_id}")
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Prompt for custom research topic
    parser = argparse.ArgumentParser(description="Simple Research Agent")
    parser.add_argument('--topic', type=str, help='Custom research topic')
    args = parser.parse_args()
    research_topic = args.topic or "artificial intelligence trends"
    print(f"\n🔎 Research Topic: {research_topic}")
    
    # Create a streamlined research task
    print("\n📋 Creating research task...")
    task_id = await orchestrator.create_task(
        description=f"Quickly research and summarize current trends in {research_topic}. Focus on 3-5 key points and complete efficiently.",
        priority=5,
        context={
            "topic": research_topic,
            "output_format": "concise summary",
            "max_iterations": 5,
            "efficiency_mode": True
        }
    )
    
    print(f"✅ Created task: {task_id}")
    
    # Assign task to research agent
    print("\n🎯 Assigning task to research agent...")
    success = await orchestrator.assign_task_to_agent(task_id, research_agent.agent_id)
    
    if success:
        print("✅ Task assigned successfully")
        
        # Monitor agent status with improved progress tracking
        print("\n🔄 Monitoring agent execution...")
        print("⏳ Research in progress... (optimized for faster completion)")
        
        # Wait for task completion with better progress indicators
        check_interval = 3  # Check every 3 seconds
        iteration_count = 0
        last_status = ""
        
        while True:
            task = orchestrator.get_task(task_id)
            if task is None:
                print(f"❌ Task not found")
                break
            
            agent_status = research_agent.get_status()
            iteration_count += 1
            
            # Show progress more frequently and detect status changes
            if agent_status.value != last_status:
                elapsed_time = iteration_count * check_interval
                print(f"📍 Status change at {elapsed_time}s: Task: {task.status.value} | Agent: {agent_status.value}")
                last_status = agent_status.value
            
            # Show periodic updates every 10 iterations (30 seconds)
            elif iteration_count % 10 == 0:
                elapsed_time = iteration_count * check_interval
                print(f"⏱️  Time: {elapsed_time}s | Task: {task.status.value} | Agent: {agent_status.value}")
            
            if task.status.value in ["completed", "failed"]:
                final_time = iteration_count * check_interval
                print(f"🏁 Task completed after {final_time}s | Task: {task.status.value} | Agent: {agent_status.value}")
                break
                
            await asyncio.sleep(check_interval)
        
        # Get final results
        final_task = orchestrator.get_task(task_id)
        if final_task is not None:
            print(f"\n📊 Final task status: {final_task.status.value}")
            if final_task.status.value == "completed":
                print("🎉 Research completed successfully!")
                
                # Get agent's memory to see the research process
                print("\n📝 Research Process Summary:")
                memories = await memory_manager.retrieve_memories(
                    agent_id=research_agent.agent_id,
                    limit=10
                )
                
                for i, memory in enumerate(memories, 1):
                    print(f"{i}. {memory}")
                
                # Create comprehensive research report from all memories
                print("\n📄 Generating comprehensive research report...")
                
                # Collect all research findings from memory
                research_findings = []
                tool_results = []
                
                for memory in memories:
                    if memory.startswith("Tool web_search result:"):
                        tool_results.append(memory)
                    elif memory.startswith("Agent response:") and len(memory) > 100:
                        research_findings.append(memory.replace("Agent response: ", "").strip())
                
                # Generate comprehensive report
                report_content = f"""# Research Report: {research_topic.title()}
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: Ollama Agentic Framework Research Agent

## Executive Summary
This report presents comprehensive research findings on {research_topic}. The research was conducted using web search tools and analysis capabilities to gather current information and trends.

## Research Process
- Total memory entries: {len(memories)}
- Web searches conducted: {len(tool_results)}
- Agent iterations: {len(research_findings)}

## Research Findings

"""
                
                # Add research findings
                for i, finding in enumerate(research_findings, 1):
                    if finding and len(finding) > 50:  # Only include substantial findings
                        report_content += f"### Finding {i}\n{finding}\n\n"
                
                # Add tool results summary
                if tool_results:
                    report_content += "## Search Results Summary\n\n"
                    for i, result in enumerate(tool_results, 1):
                        # Extract key info from tool results
                        if "results" in result:
                            report_content += f"Search {i}: {result[:200]}...\n\n"
                
                report_content += f"""
## Conclusion
This research on {research_topic} was conducted using the Ollama Agentic Framework, leveraging real LLM capabilities for comprehensive analysis and synthesis of information.

Generated by Simple Research Agent
Framework: Ollama Agentic Framework
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                # Write the comprehensive report to file
                await write_report_to_file(tool_registry, report_content, research_topic)
            else:
                print("❌ Research task failed or timed out")
        else:
            print("❌ Task not found in final check")
    
    else:
        print("❌ Failed to assign task to agent")
    
    # Stop the orchestrator
    await orchestrator.stop()
    
    # Get final statistics
    print("\n📈 Final Statistics:")
    agent_stats = orchestrator.get_agent_status_summary()
    task_stats = orchestrator.get_task_status_summary()
    memory_stats = await memory_manager.get_memory_stats()
    
    print(f"Agent statuses: {agent_stats}")
    print(f"Task statuses: {task_stats}")
    print(f"Memory entries: {memory_stats['total_memories']}")
    
    print("\n✨ Example completed!")


async def write_report_to_file(tool_registry, report_content, topic):
    """Write the research report to a txt file using the file_write tool."""
    safe_topic = "_".join(topic.lower().split())
    filename = f"research_report_{safe_topic}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    result = await tool_registry.execute_tool("file_write", {
        "filename": filename,
        "content": report_content,
        "create_dirs": True
    })
    if result.get("success"):
        print(f"\n💾 Research report saved to: {result['filename']}")
    else:
        print(f"\n❌ Failed to save research report: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())


