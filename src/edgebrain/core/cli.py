import argparse
import asyncio
import logging

from .orchestrator import AgentOrchestrator
from ..integration.ollama_client import OllamaIntegrationLayer
from ..tools.tool_registry import ToolRegistry
from ..memory.memory_manager import MemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def run_orchestrator(args):
    ollama_integration = OllamaIntegrationLayer(base_url=args.ollama_url, default_model=args.model)
    if not await ollama_integration.initialize():
        logging.error("Failed to connect to Ollama. Please ensure Ollama is running.")
        return

    tool_registry = ToolRegistry()
    memory_manager = MemoryManager(db_path=args.memory_db)

    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )

    await orchestrator.start()
    logging.info("Orchestrator started.")

    # Example: Register a simple agent
    agent = orchestrator.register_agent(
        agent_id="cli_agent",
        role="CLI Assistant",
        description="Assists with command-line tasks",
        model=args.model
    )
    logging.info(f"Registered agent: {agent.agent_id}")

    if args.task:
        task_id = await orchestrator.create_task(args.task)
        logging.info(f"Created task: {task_id}")
        await orchestrator.assign_task_to_agent(task_id, agent.agent_id)
        logging.info(f"Assigned task {task_id} to {agent.agent_id}")

        # Simple monitoring loop
        while True:
            task = orchestrator.get_task(task_id)
            
            # Add null check to prevent optional member access errors
            if task is None:
                logging.error(f"Task {task_id} not found")
                break
                
            if task.status.value in ["completed", "failed", "cancelled"]:
                logging.info(f"Task {task_id} finished with status: {task.status.value}")
                if task.result:
                    logging.info(f"Task result: {task.result}")
                break
            logging.info(f"Task {task_id} status: {task.status.value}...")
            await asyncio.sleep(2)

    await orchestrator.stop()
    logging.info("Orchestrator stopped.")

def main():
    parser = argparse.ArgumentParser(description="EdgeBrain: Ollama Agentic Framework CLI")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="URL of the Ollama server")
    parser.add_argument("--model", type=str, default="llama3.1",
                        help="Default Ollama model to use")
    parser.add_argument("--memory-db", type=str, default="agent_memory.db",
                        help="Path to the SQLite database for memory")
    parser.add_argument("--task", type=str, help="A task description for the CLI agent to execute")

    args = parser.parse_args()
    asyncio.run(run_orchestrator(args))

if __name__ == "__main__":
    main()