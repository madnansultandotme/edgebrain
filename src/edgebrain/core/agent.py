"""
Agent Implementation

This module defines the Agent class, which represents an autonomous entity
within the framework capable of performing specific tasks.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import logging

from ..integration.ollama_client import OllamaIntegrationLayer, OllamaResponse
from ..tools.tool_registry import ToolRegistry, Tool
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Enumeration of possible agent statuses."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class AgentMessage(BaseModel):
    """Represents a message sent to or from an agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = Field(..., description="ID of the message sender")
    recipient: str = Field(..., description="ID of the message recipient")
    content: str = Field(..., description="Content of the message")
    message_type: str = Field(default="text", description="Type of the message")
    timestamp: float = Field(default_factory=lambda: asyncio.get_event_loop().time())


class AgentGoal(BaseModel):
    """Represents a goal assigned to an agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Description of the goal")
    priority: int = Field(default=1, description="Priority of the goal (1-10)")
    deadline: Optional[float] = Field(None, description="Deadline for completing the goal")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the goal")


class AgentCapability(BaseModel):
    """Represents a capability of an agent."""
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what the capability does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the capability")


class Agent:
    """
    An autonomous agent capable of performing specific tasks.
    
    Each agent has its own set of capabilities, tools, and a defined role.
    Agents interact with the environment and other agents through the Orchestrator.
    """
    
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
    ):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent (e.g., "Researcher", "Coder", "Planner")
            description: Description of the agent's purpose
            ollama_integration: Ollama integration layer for LLM interactions
            tool_registry: Registry of available tools
            memory_manager: Memory management system
            model: Specific model to use (optional)
            capabilities: List of agent capabilities
            system_prompt: Custom system prompt for the agent
            max_iterations: Maximum number of iterations for task execution
        """
        self.agent_id = agent_id
        self.role = role
        self.description = description
        self.ollama_integration = ollama_integration
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.model = model
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.current_goal: Optional[AgentGoal] = None
        self.message_queue: List[AgentMessage] = []
        self.iteration_count = 0
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt(system_prompt)
        
        # Callbacks
        self.on_status_change: Optional[Callable[[AgentStatus], None]] = None
        self.on_message_sent: Optional[Callable[[AgentMessage], None]] = None
        
        logger.info(f"Agent {self.agent_id} ({self.role}) initialized")
    
    def _build_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """Build the system prompt for the agent."""
        base_prompt = f"""You are an AI agent with the role of {self.role}.

Description: {self.description}

Your capabilities include:
{chr(10).join([f"- {cap.name}: {cap.description}" for cap in self.capabilities])}

Available tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in self.tool_registry.get_all_tools()])}

Guidelines:
1. Always work towards your assigned goal
2. Use available tools when necessary to gather information or perform actions
3. Think step by step and explain your reasoning
4. If you need help or clarification, ask for it
5. Be concise but thorough in your responses
6. If you encounter an error, try to recover or ask for assistance

When you need to use a tool, clearly indicate which tool you want to use and provide the necessary parameters."""
        
        if custom_prompt:
            return f"{base_prompt}\n\nAdditional instructions:\n{custom_prompt}"
        
        return base_prompt
    
    async def set_goal(self, goal: AgentGoal) -> None:
        """
        Set a new goal for the agent.
        
        Args:
            goal: The goal to assign to the agent
        """
        self.current_goal = goal
        self.iteration_count = 0
        await self._change_status(AgentStatus.THINKING)
        
        # Store goal in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            content=f"New goal assigned: {goal.description}",
            memory_type="goal",
            metadata={"goal_id": goal.id, "priority": goal.priority}
        )
        
        logger.info(f"Agent {self.agent_id} assigned goal: {goal.description}")
    
    async def process_message(self, message: AgentMessage) -> None:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
        """
        self.message_queue.append(message)
        logger.info(f"Agent {self.agent_id} received message from {message.sender}")
        
        # Store message in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            content=f"Received message: {message.content}",
            memory_type="communication",
            metadata={"sender": message.sender, "message_id": message.id}
        )
    
    async def execute_goal(self) -> bool:
        """
        Execute the current goal.
        
        Returns:
            True if goal completed successfully, False otherwise
        """
        if not self.current_goal:
            logger.warning(f"Agent {self.agent_id} has no goal to execute")
            return False
        
        await self._change_status(AgentStatus.THINKING)
        
        try:
            while self.iteration_count < self.max_iterations:
                self.iteration_count += 1
                
                # Get relevant context from memory
                context = await self.memory_manager.retrieve_memories(
                    agent_id=self.agent_id,
                    query=self.current_goal.description,
                    limit=5
                )
                
                # Build prompt for current iteration
                prompt = self._build_execution_prompt(context)
                
                # Generate response
                await self._change_status(AgentStatus.THINKING)
                response = await self.ollama_integration.generate_response(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    model=self.model,
                    tools=self._get_tool_definitions()
                )
                
                # Process response
                result = await self._process_response(response)
                
                if result == "completed":
                    await self._change_status(AgentStatus.COMPLETED)
                    logger.info(f"Agent {self.agent_id} completed goal: {self.current_goal.description}")
                    return True
                elif result == "error":
                    await self._change_status(AgentStatus.ERROR)
                    logger.error(f"Agent {self.agent_id} encountered error while executing goal")
                    return False
                
                # Brief pause between iterations
                await asyncio.sleep(0.1)
            
            # Max iterations reached
            logger.warning(f"Agent {self.agent_id} reached max iterations without completing goal")
            await self._change_status(AgentStatus.ERROR)
            return False
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution error: {e}")
            await self._change_status(AgentStatus.ERROR)
            return False
    
    def _build_execution_prompt(self, context: List[str]) -> str:
        """Build the prompt for goal execution."""
        # Check if current_goal is not None before accessing its attributes
        if not self.current_goal:
            raise ValueError("Cannot build execution prompt without a current goal")
        
        prompt_parts = [
            f"Current goal: {self.current_goal.description}",
            f"Iteration: {self.iteration_count}/{self.max_iterations}",
        ]
        
        # Safely access context attribute
        if self.current_goal.context:
            prompt_parts.append(f"Goal context: {self.current_goal.context}")
        
        if context:
            prompt_parts.append("Relevant context from memory:")
            prompt_parts.extend([f"- {ctx}" for ctx in context])
        
        if self.message_queue:
            prompt_parts.append("Recent messages:")
            for msg in self.message_queue[-3:]:  # Last 3 messages
                prompt_parts.append(f"- From {msg.sender}: {msg.content}")
        
        prompt_parts.append(
            "What should you do next to work towards your goal? "
            "If you need to use a tool, specify which tool and its parameters. "
            "If you have completed the goal, respond with 'GOAL_COMPLETED'. "
            "If you need help, ask for it clearly."
        )
        
        return "\n\n".join(prompt_parts)
    
    async def _process_response(self, response: OllamaResponse) -> str:
        """
        Process the response from Ollama.
        
        Args:
            response: Response from Ollama
            
        Returns:
            Status of processing ("continue", "completed", "error")
        """
        # Store the response in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            content=f"Agent response: {response.content}",
            memory_type="reasoning",
            metadata={"iteration": self.iteration_count}
        )
        
        # Check if goal is completed
        if "GOAL_COMPLETED" in response.content.upper():
            return "completed"
        
        # Process tool calls
        if response.tool_calls:
            await self._change_status(AgentStatus.ACTING)
            
            for tool_call in response.tool_calls:
                try:
                    tool_result = await self.tool_registry.execute_tool(
                        tool_name=tool_call.name,
                        parameters=tool_call.arguments
                    )
                    
                    # Store tool execution result
                    await self.memory_manager.store_memory(
                        agent_id=self.agent_id,
                        content=f"Tool {tool_call.name} result: {tool_result}",
                        memory_type="action",
                        metadata={"tool": tool_call.name, "iteration": self.iteration_count}
                    )
                    
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    await self.memory_manager.store_memory(
                        agent_id=self.agent_id,
                        content=f"Tool {tool_call.name} error: {str(e)}",
                        memory_type="error",
                        metadata={"tool": tool_call.name, "iteration": self.iteration_count}
                    )
        
        return "continue"
    
    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for Ollama."""
        tools = []
        for tool in self.tool_registry.get_all_tools():
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            tools.append(tool_def)
        return tools
    
    async def send_message(self, recipient: str, content: str, message_type: str = "text") -> None:
        """
        Send a message to another agent or the orchestrator.
        
        Args:
            recipient: ID of the message recipient
            content: Content of the message
            message_type: Type of the message
        """
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        
        if self.on_message_sent:
            self.on_message_sent(message)
        
        # Store sent message in memory
        await self.memory_manager.store_memory(
            agent_id=self.agent_id,
            content=f"Sent message to {recipient}: {content}",
            memory_type="communication",
            metadata={"recipient": recipient, "message_id": message.id}
        )
        
        logger.info(f"Agent {self.agent_id} sent message to {recipient}")
    
    async def _change_status(self, new_status: AgentStatus) -> None:
        """Change the agent's status."""
        old_status = self.status
        self.status = new_status
        
        if self.on_status_change:
            self.on_status_change(new_status)
        
        logger.debug(f"Agent {self.agent_id} status changed: {old_status} -> {new_status}")
    
    def get_status(self) -> AgentStatus:
        """Get the current status of the agent."""
        return self.status
    
    def get_current_goal(self) -> Optional[AgentGoal]:
        """Get the current goal of the agent."""
        return self.current_goal
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a new capability to the agent."""
        self.capabilities.append(capability)
        # Rebuild system prompt with new capability
        self.system_prompt = self._build_system_prompt()
        logger.info(f"Added capability {capability.name} to agent {self.agent_id}")
    
    async def reset(self) -> None:
        """Reset the agent to initial state."""
        self.current_goal = None
        self.message_queue.clear()
        self.iteration_count = 0
        await self._change_status(AgentStatus.IDLE)
        logger.info(f"Agent {self.agent_id} reset to initial state")