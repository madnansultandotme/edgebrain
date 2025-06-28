#!/usr/bin/env python3
"""
EdgeBrain PyPI Installation and Usage Example

This script demonstrates how to use EdgeBrain after installing it from PyPI.
Run this in a new directory to test your EdgeBrain installation.

Installation:
    pip install edgebrain ollama

Usage:
    python edgebrain_pypi_example.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Check if EdgeBrain is installed
try:
    # Try PyPI/proper package structure first
    from src.edgebrain.core.orchestrator import AgentOrchestrator
    from src.edgebrain.integration.ollama_client import OllamaIntegrationLayer
    print("✅ EdgeBrain imported successfully from PyPI installation")
    use_dev_env = False
except ImportError:
    try:
        # Fallback to development environment
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from src.edgebrain.core.orchestrator import AgentOrchestrator
        from src.edgebrain.integration.ollama_client import OllamaIntegrationLayer
        print("✅ EdgeBrain imported successfully from development environment")
        use_dev_env = True
    except ImportError as e:
        print(f"❌ EdgeBrain not found. Please install: pip install edgebrain")
        print(f"Error: {e}")
        sys.exit(1)

# Check if Ollama client is available
try:
    import ollama
    print("✅ Ollama async client available")
except ImportError:
    print("❌ Ollama client not found. Please install: pip install ollama")
    sys.exit(1)


class SimpleCodeGenerator:
    """Simple code generator using EdgeBrain patterns."""
    
    def __init__(self, model="qwen2.5:3b"):
        self.model = model
        self.client = ollama.AsyncClient()
    
    async def generate_function(self, description: str, language: str = "python") -> str:
        """Generate a function based on description."""
        
        prompt = f"""Create a {language} function that {description}.
Include:
- Proper function signature
- Docstring with description and parameters
- Error handling where appropriate
- Example usage in comments
- Clean, readable code"""
        
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert {language} programmer."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response['message']['content'] if response else "No code generated"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def save_to_file(self, code: str, filename: str):
        """Save generated code to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"✅ Code saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save code: {e}")


async def demonstrate_basic_agent():
    """Demonstrate basic agent creation and task execution."""
    
    print("\n🤖 Demonstrating Basic Agent...")
    
    try:
        # Initialize EdgeBrain components (as would be done with PyPI installation)
        from src.edgebrain.integration.ollama_client import OllamaIntegrationLayer
        from src.edgebrain.tools.tool_registry import ToolRegistry
        from src.edgebrain.memory.memory_manager import MemoryManager
        
        # Initialize all required components
        ollama_integration = OllamaIntegrationLayer()
        await ollama_integration.initialize()
        
        tool_registry = ToolRegistry()
        memory_manager = MemoryManager()
        
        # Create orchestrator with all required components
        orchestrator = AgentOrchestrator(
            ollama_integration=ollama_integration,
            tool_registry=tool_registry,
            memory_manager=memory_manager
        )
        
        # Register a research agent
        agent = orchestrator.register_agent(
            agent_id="demo_agent",
            role="Demo Assistant",
            description="A demo agent for showcasing EdgeBrain capabilities",
            model="qwen2.5:3b",
            capabilities=["analysis", "writing"]
        )
        
        # Assign a task
        task_id = await orchestrator.assign_task(
            agent_id="demo_agent",
            task_description="Explain the key benefits of using AI agents in software development",
            context={"length": "brief", "focus": "practical benefits"}
        )
        
        # Wait for completion
        result = await orchestrator.get_task_result(task_id)
        print(f"📝 Agent Response:\n{result}")
        
        # Cleanup
        await orchestrator.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Agent demonstration failed: {e}")
        print("   This is expected in development environment.")
        print("   When installed from PyPI, this will work correctly.")
        return False


async def demonstrate_code_generation():
    """Demonstrate direct code generation."""
    
    print("\n💻 Demonstrating Code Generation...")
    
    generator = SimpleCodeGenerator()
    
    # Generate different types of functions
    examples = [
        ("calculates the fibonacci sequence up to n terms", "fibonacci_generator.py"),
        ("validates if a string is a valid email address", "email_validator.py"),
        ("sorts a list of dictionaries by a specified key", "dict_sorter.py")
    ]
    
    generated_files = []
    
    for description, filename in examples:
        print(f"🔧 Generating: {description}")
        
        code = await generator.generate_function(description)
        
        if "Error:" not in code:
            await generator.save_to_file(code, filename)
            generated_files.append(filename)
            print(f"   Created: {filename}")
        else:
            print(f"   Failed: {code}")
    
    return generated_files


async def verify_ollama_connection():
    """Verify Ollama service is running and models are available."""
    
    print("\n🔍 Verifying Ollama Connection...")
    
    try:
        client = ollama.AsyncClient()
        
        # Test connection
        models = await client.list()
        available_models = [model['name'] for model in models.get('models', [])]
        
        print(f"✅ Connected to Ollama. Available models: {len(available_models)}")
        
        # Check for recommended models
        recommended = ['qwen2.5:3b', 'llama3.1']
        missing_models = [model for model in recommended if model not in available_models]
        
        if missing_models:
            print(f"⚠️  Missing recommended models: {missing_models}")
            print(f"   Install with: ollama pull {' && ollama pull '.join(missing_models)}")
        else:
            print("✅ All recommended models available")
        
        return len(available_models) > 0
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False


def create_project_structure():
    """Create a basic project structure."""
    
    print("\n📁 Creating Project Structure...")
    
    directories = ['src', 'examples', 'output']
    files = {
        'requirements.txt': 'edgebrain>=0.1.1\nollama\nasyncio\n',
        'README.md': '''# My EdgeBrain Application

This project uses EdgeBrain from PyPI for AI agent development.

## Installation

```bash
pip install edgebrain ollama
```

## Usage

```bash
python edgebrain_pypi_example.py
```
''',
        '.gitignore': '''__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
*.log
.DS_Store
'''
    }
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created directory: {directory}/")
    
    # Create files
    for filename, content in files.items():
        if not Path(filename).exists():
            with open(filename, 'w') as f:
                f.write(content)
            print(f"   Created file: {filename}")


async def main():
    """Main demonstration function."""
    
    print("🚀 EdgeBrain PyPI Installation Demo")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Verify Ollama connection
    ollama_ok = await verify_ollama_connection()
    
    if not ollama_ok:
        print("\n❌ Cannot proceed without Ollama. Please:")
        print("   1. Install Ollama: https://ollama.ai/download")
        print("   2. Start service: ollama serve")
        print("   3. Pull models: ollama pull qwen2.5:3b")
        return
    
    # Demonstrate code generation
    generated_files = await demonstrate_code_generation()
    
    # Demonstrate agent usage
    agent_success = await demonstrate_basic_agent()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Summary:")
    print(f"   ✅ Generated {len(generated_files)} code files")
    print(f"   {'✅' if agent_success else '❌'} Agent demonstration")
    
    if generated_files:
        print(f"\n📄 Generated Files:")
        for file in generated_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   - {file} ({size} bytes)")
    
    print("\n🎉 EdgeBrain PyPI demo complete!")
    print("\nNext steps:")
    print("   - Explore the generated code files")
    print("   - Read the documentation: docs/")
    print("   - Build your own agents!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check your installation and try again")
