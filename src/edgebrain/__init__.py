"""
EdgeBrain - An open-source agentic framework for building AI agents with Ollama-based models

This package provides a comprehensive framework for building and orchestrating AI agents
that can collaborate, reason, and execute tasks using local language models via Ollama.
"""

__version__ = "0.1.2"
__author__ = "Muhammad Adnan Sultan"
__email__ = "info.adnansultan@gmail.com"

# Core imports for easy access
from .core.agent import Agent
from .core.orchestrator import AgentOrchestrator
from .integration.ollama_client import OllamaIntegrationLayer
from .tools.tool_registry import ToolRegistry
from .memory.memory_manager import MemoryManager

__all__ = [
    "Agent",
    "AgentOrchestrator", 
    "OllamaIntegrationLayer",
    "ToolRegistry",
    "MemoryManager"
]
