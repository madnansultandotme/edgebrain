"""
Basic tests for EdgeBrain framework core functionality.
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from edgebrain.core.agent import Agent
from edgebrain.integration.ollama_client import OllamaClient
from edgebrain.tools.tool_registry import ToolRegistry
from edgebrain.memory.memory_manager import MemoryManager


class TestCoreImports:
    """Test that all core modules can be imported successfully."""
    
    def test_agent_import(self):
        """Test that Agent class can be imported."""
        assert Agent is not None
        
    def test_ollama_client_import(self):
        """Test that OllamaClient class can be imported."""
        assert OllamaClient is not None
        
    def test_tool_registry_import(self):
        """Test that ToolRegistry class can be imported."""
        assert ToolRegistry is not None
        
    def test_memory_manager_import(self):
        """Test that MemoryManager class can be imported."""
        assert MemoryManager is not None


class TestAgentBasics:
    """Test basic Agent functionality."""
    
    def test_agent_creation(self):
        """Test that an Agent can be created with basic configuration."""
        # Create required dependencies
        from edgebrain.integration.ollama_client import OllamaIntegrationLayer
        
        ollama = OllamaIntegrationLayer()  # Use default parameters
        tool_registry = ToolRegistry()
        memory = MemoryManager()
        
        agent = Agent(
            agent_id="test_agent",
            role="testing",
            description="Test agent for CI",
            ollama_integration=ollama,
            tool_registry=tool_registry,
            memory_manager=memory
        )
        assert agent.agent_id == "test_agent"
        assert agent.role == "testing"
        assert agent.description == "Test agent for CI"


class TestToolRegistry:
    """Test ToolRegistry functionality."""
    
    def test_tool_registry_creation(self):
        """Test that ToolRegistry can be created."""
        registry = ToolRegistry()
        assert registry is not None
        
    def test_default_tools_available(self):
        """Test that default tools are available in registry."""
        registry = ToolRegistry()
        # Should have some default tools
        assert len(registry.list_tools()) > 0


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def test_memory_manager_creation(self):
        """Test that MemoryManager can be created."""
        memory = MemoryManager()
        assert memory is not None


if __name__ == "__main__":
    # Run basic import tests
    print("Running basic EdgeBrain framework tests...")
    
    # Test imports
    try:
        from edgebrain.core.agent import Agent
        from edgebrain.integration.ollama_client import OllamaClient
        from edgebrain.tools.tool_registry import ToolRegistry
        from edgebrain.memory.memory_manager import MemoryManager
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    # Test basic functionality
    try:
        from edgebrain.integration.ollama_client import OllamaIntegrationLayer
        
        ollama = OllamaIntegrationLayer()  # Use default parameters
        tool_registry = ToolRegistry()
        memory = MemoryManager()
        
        agent = Agent(
            agent_id="test_agent",
            role="testing",
            description="Test agent for CI",
            ollama_integration=ollama,
            tool_registry=tool_registry,
            memory_manager=memory
        )
        print("✅ Basic object creation successful")
        print(f"✅ Available tools: {len(tool_registry.list_tools())}")
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        sys.exit(1)
    
    print("✅ All basic tests passed!")
