"""
Agent Orchestrator Implementation

The Agent Orchestrator is the central control unit of the framework.
It manages the lifecycle of agents, coordinates their interactions,
and oversees the execution of tasks.
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .agent import Agent, AgentGoal, AgentMessage, AgentStatus
from ..integration.ollama_client import OllamaIntegrationLayer
from ..tools.tool_registry import ToolRegistry
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Represents a high-level task to be executed."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Description of the task")
    priority: int = Field(default=1, description="Priority of the task (1-10)")
    assigned_agents: List[str] = Field(default_factory=list, description="IDs of agents assigned to this task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the task")
    created_at: float = Field(default_factory=lambda: time.time())
    completed_at: Optional[float] = None
    result: Optional[Any] = None


class WorkflowStep(BaseModel):
    """Represents a step in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., description="Description of the step")
    agent_role: str = Field(..., description="Role of the agent that should execute this step")
    dependencies: List[str] = Field(default_factory=list, description="IDs of steps that must complete first")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for this step")


class Workflow(BaseModel):
    """Represents a workflow consisting of multiple steps."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Description of the workflow")
    steps: List[WorkflowStep] = Field(..., description="Steps in the workflow")
    context: Dict[str, Any] = Field(default_factory=dict, description="Global context for the workflow")


class AgentOrchestrator:
    """
    Central control unit for managing agents and coordinating their interactions.
    
    The orchestrator handles task management, agent lifecycle, communication,
    and workflow execution.
    """
    
    def __init__(
        self,
        ollama_integration: OllamaIntegrationLayer,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager
    ):
        """
        Initialize the agent orchestrator.
        
        Args:
            ollama_integration: Ollama integration layer
            tool_registry: Tool registry for agents
            memory_manager: Memory management system
        """
        self.ollama_integration = ollama_integration
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        
        # Agent management
        self._agents: Dict[str, Agent] = {}
        self._agent_roles: Dict[str, List[str]] = {}  # role -> list of agent IDs
        
        # Task management
        self._tasks: Dict[str, Task] = {}
        self._task_queue: List[str] = []
        
        # Workflow management
        self._workflows: Dict[str, Workflow] = {}
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        self._message_queue: List[AgentMessage] = []
        self._message_handlers: Dict[str, Callable[[AgentMessage], None]] = {}
        
        # Event callbacks
        self.on_task_completed: Optional[Callable[[Task], None]] = None
        self.on_agent_status_change: Optional[Callable[[str, AgentStatus], None]] = None
        
        # Control flags
        self._running = False
        self._processing_task = None
        
        logger.info("Agent orchestrator initialized")
    
    async def start(self) -> None:
        """Start the orchestrator."""
        self._running = True
        logger.info("Agent orchestrator started")
        
        # Start background tasks
        asyncio.create_task(self._process_message_queue())
        asyncio.create_task(self._process_task_queue())
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        logger.info("Agent orchestrator stopped")
    
    def register_agent(
        self,
        agent_id: str,
        role: str,
        description: str,
        model: Optional[str] = None,
        capabilities: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Agent:
        """
        Register a new agent with the orchestrator.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent
            description: Description of the agent's purpose
            model: Specific model to use for the agent
            capabilities: List of agent capabilities
            system_prompt: Custom system prompt for the agent
            
        Returns:
            The created agent instance
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent with ID '{agent_id}' already exists")
        
        # Create agent
        agent = Agent(
            agent_id=agent_id,
            role=role,
            description=description,
            ollama_integration=self.ollama_integration,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            model=model,
            capabilities=capabilities,
            system_prompt=system_prompt
        )
        
        # Set up callbacks
        agent.on_status_change = lambda status: self._on_agent_status_change(agent_id, status)
        agent.on_message_sent = self._on_agent_message_sent
        
        # Register agent
        self._agents[agent_id] = agent
        
        # Add to role mapping
        if role not in self._agent_roles:
            self._agent_roles[role] = []
        self._agent_roles[role].append(agent_id)
        
        logger.info(f"Registered agent {agent_id} with role {role}")
        return agent
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        # Remove from role mapping
        if agent.role in self._agent_roles:
            if agent_id in self._agent_roles[agent.role]:
                self._agent_roles[agent.role].remove(agent_id)
            
            # Remove empty role
            if not self._agent_roles[agent.role]:
                del self._agent_roles[agent.role]
        
        # Remove agent
        del self._agents[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_agents_by_role(self, role: str) -> List[Agent]:
        """Get all agents with a specific role."""
        agent_ids = self._agent_roles.get(role, [])
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]
    
    def get_all_agents(self) -> List[Agent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    async def create_task(
        self,
        description: str,
        priority: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task.
        
        Args:
            description: Description of the task
            priority: Priority of the task (1-10)
            context: Additional context for the task
            
        Returns:
            ID of the created task
        """
        task = Task(
            description=description,
            priority=priority,
            context=context or {}
        )
        
        self._tasks[task.id] = task
        self._task_queue.append(task.id)
        
        # Sort task queue by priority
        self._task_queue.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
        
        logger.info(f"Created task {task.id}: {description}")
        return task.id
    
    async def assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """
        Assign a task to a specific agent.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            
        Returns:
            True if assignment successful, False otherwise
        """
        if task_id not in self._tasks or agent_id not in self._agents:
            return False
        
        task = self._tasks[task_id]
        agent = self._agents[agent_id]
        
        # Create goal for agent
        goal = AgentGoal(
            description=task.description,
            priority=task.priority,
            deadline=None,
            context=task.context
        )
        
        # Assign goal to agent
        await agent.set_goal(goal)
        
        # Update task
        task.assigned_agents.append(agent_id)
        task.status = TaskStatus.IN_PROGRESS
        
        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        return True
    
    async def assign_task_to_role(self, task_id: str, role: str) -> bool:
        """
        Assign a task to an agent with a specific role.
        
        Args:
            task_id: ID of the task
            role: Role of the agent to assign to
            
        Returns:
            True if assignment successful, False otherwise
        """
        agents = self.get_agents_by_role(role)
        if not agents:
            logger.warning(f"No agents found with role {role}")
            return False
        
        # Find an idle agent
        for agent in agents:
            if agent.get_status() == AgentStatus.IDLE:
                return await self.assign_task_to_agent(task_id, agent.agent_id)
        
        # If no idle agents, assign to the first one
        return await self.assign_task_to_agent(task_id, agents[0].agent_id)
    
    async def execute_workflow(self, workflow: Workflow) -> bool:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            True if workflow started successfully, False otherwise
        """
        if workflow.id in self._active_workflows:
            logger.warning(f"Workflow {workflow.id} is already active")
            return False
        
        # Initialize workflow state
        workflow_state = {
            "workflow": workflow,
            "completed_steps": set(),
            "in_progress_steps": set(),
            "step_results": {}
        }
        
        self._active_workflows[workflow.id] = workflow_state
        
        # Start executing steps
        await self._execute_workflow_steps(workflow.id)
        
        logger.info(f"Started workflow {workflow.id}: {workflow.name}")
        return True
    
    async def _execute_workflow_steps(self, workflow_id: str) -> None:
        """Execute workflow steps based on dependencies."""
        if workflow_id not in self._active_workflows:
            return
        
        workflow_state = self._active_workflows[workflow_id]
        workflow = workflow_state["workflow"]
        
        # Find steps that can be executed (dependencies met)
        ready_steps = []
        for step in workflow.steps:
            if (step.id not in workflow_state["completed_steps"] and 
                step.id not in workflow_state["in_progress_steps"]):
                
                # Check if all dependencies are completed
                dependencies_met = all(
                    dep_id in workflow_state["completed_steps"] 
                    for dep_id in step.dependencies
                )
                
                if dependencies_met:
                    ready_steps.append(step)
        
        # Execute ready steps
        for step in ready_steps:
            await self._execute_workflow_step(workflow_id, step)
    
    async def _execute_workflow_step(self, workflow_id: str, step: WorkflowStep) -> None:
        """Execute a single workflow step."""
        workflow_state = self._active_workflows[workflow_id]
        workflow_state["in_progress_steps"].add(step.id)
        
        # Find an agent with the required role
        agents = self.get_agents_by_role(step.agent_role)
        if not agents:
            logger.error(f"No agents found with role {step.agent_role} for step {step.id}")
            return
        
        # Create a task for this step
        task_id = await self.create_task(
            description=step.description,
            context=step.context
        )
        
        # Assign to an available agent
        success = await self.assign_task_to_role(task_id, step.agent_role)
        
        if success:
            logger.info(f"Executing workflow step {step.id} with agent role {step.agent_role}")
        else:
            logger.error(f"Failed to assign workflow step {step.id}")
    
    async def send_message(self, sender_id: str, recipient_id: str, content: str, message_type: str = "text") -> None:
        """
        Send a message between agents.
        
        Args:
            sender_id: ID of the sender
            recipient_id: ID of the recipient
            content: Content of the message
            message_type: Type of the message
        """
        message = AgentMessage(
            sender=sender_id,
            recipient=recipient_id,
            content=content,
            message_type=message_type
        )
        
        self._message_queue.append(message)
        logger.debug(f"Queued message from {sender_id} to {recipient_id}")
    
    async def broadcast_message(self, sender_id: str, content: str, message_type: str = "broadcast") -> None:
        """
        Broadcast a message to all agents.
        
        Args:
            sender_id: ID of the sender
            content: Content of the message
            message_type: Type of the message
        """
        for agent_id in self._agents:
            if agent_id != sender_id:
                await self.send_message(sender_id, agent_id, content, message_type)
    
    async def _process_message_queue(self) -> None:
        """Process the message queue."""
        while self._running:
            if self._message_queue:
                message = self._message_queue.pop(0)
                await self._deliver_message(message)
            else:
                await asyncio.sleep(0.1)
    
    async def _deliver_message(self, message: AgentMessage) -> None:
        """Deliver a message to its recipient."""
        if message.recipient in self._agents:
            agent = self._agents[message.recipient]
            await agent.process_message(message)
        elif message.recipient in self._message_handlers:
            self._message_handlers[message.recipient](message)
        else:
            logger.warning(f"No recipient found for message: {message.recipient}")
    
    async def _process_task_queue(self) -> None:
        """Process the task queue."""
        while self._running:
            if self._task_queue and not self._processing_task:
                task_id = self._task_queue.pop(0)
                self._processing_task = asyncio.create_task(self._execute_task(task_id))
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    logger.info(f"Task {task_id} was cancelled")
                finally:
                    self._processing_task = None
            else:
                await asyncio.sleep(0.1)
    
    async def _execute_task(self, task_id: str) -> None:
        """Execute a task."""
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        if not task.assigned_agents:
            # Auto-assign based on task description
            # This is a simple implementation - in practice, you might use more sophisticated assignment logic
            if self._agents:
                first_agent_id = list(self._agents.keys())[0]
                await self.assign_task_to_agent(task_id, first_agent_id)
        
        # Execute with assigned agents
        for agent_id in task.assigned_agents:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                success = await agent.execute_goal()
                
                if success:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    
                    if self.on_task_completed:
                        self.on_task_completed(task)
                    
                    logger.info(f"Task {task_id} completed successfully")
                    break
                else:
                    logger.warning(f"Agent {agent_id} failed to complete task {task_id}")
        
        if task.status != TaskStatus.COMPLETED:
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task_id} failed")
    
    def _on_agent_status_change(self, agent_id: str, status: AgentStatus) -> None:
        """Handle agent status changes."""
        if self.on_agent_status_change:
            self.on_agent_status_change(agent_id, status)
        
        logger.debug(f"Agent {agent_id} status changed to {status}")
    
    def _on_agent_message_sent(self, message: AgentMessage) -> None:
        """Handle messages sent by agents."""
        self._message_queue.append(message)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def get_agent_status_summary(self) -> Dict[str, int]:
        """Get a summary of agent statuses."""
        status_counts = {}
        for agent in self._agents.values():
            status = agent.get_status().value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    def get_task_status_summary(self) -> Dict[str, int]:
        """Get a summary of task statuses."""
        status_counts = {}
        for task in self._tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts