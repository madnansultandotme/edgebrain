# EdgeBrain PyPI Usage Guide

This guide shows you how to create a new project using EdgeBrain installed from PyPI.

## Step 1: Create New Project Directory

```bash
# Create and enter new project directory
mkdir my-edgebrain-project
cd my-edgebrain-project
```

## Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

## Step 3: Install EdgeBrain

```bash
# Install EdgeBrain from PyPI
pip install edgebrain

# Install Ollama async client
pip install ollama
```

## Step 4: Verify Ollama Setup

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, pull required models
ollama pull qwen2.5:3b    # For code generation
ollama pull llama3.1      # For general tasks
```

## Step 5: Create Your First Agent

Create a file called `my_first_agent.py`:

```python
import asyncio
from edgebrain.core.orchestrator import AgentOrchestrator
from edgebrain.integration.ollama_client import OllamaIntegrationLayer

async def main():
    # Initialize EdgeBrain components
    ollama = OllamaIntegrationLayer()
    await ollama.initialize()
    
    orchestrator = AgentOrchestrator(ollama_integration=ollama)
    
    # Create an agent
    agent = orchestrator.register_agent(
        agent_id="my_assistant",
        role="Personal Assistant",
        capabilities=["research", "writing"]
    )
    
    # Give it a task
    task_id = await orchestrator.assign_task(
        agent_id="my_assistant",
        task_description="Write a brief explanation of machine learning",
        context={"audience": "beginners", "length": "2 paragraphs"}
    )
    
    # Get the result
    result = await orchestrator.wait_for_completion(task_id)
    print("Agent Response:")
    print(result)
    
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 6: Create a Code Generation Script

Create a file called `code_generator.py`:

```python
import asyncio
import ollama

async def generate_python_function():
    # Connect to Ollama
    client = ollama.AsyncClient()
    
    # Define what we want
    prompt = "Create a Python function that calculates the area of a circle"
    
    # Generate code
    response = await client.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": "You are a Python expert. Generate clean, documented code."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Save the code
    if response and 'message' in response:
        code = response['message']['content']
        
        with open("circle_area.py", "w") as f:
            f.write(code)
        
        print("✅ Generated code saved to circle_area.py")
        print("\nGenerated Code:")
        print("-" * 40)
        print(code)
    else:
        print("❌ Failed to generate code")

if __name__ == "__main__":
    asyncio.run(generate_python_function())
```

## Step 7: Run Your Applications

```bash
# Run the agent
python my_first_agent.py

# Run the code generator
python code_generator.py
```

## Project Structure

Your project should now look like this:

```
my-edgebrain-project/
├── venv/                    # Virtual environment
├── my_first_agent.py       # Your first agent
├── code_generator.py       # Code generation example
├── circle_area.py          # Generated code output
└── requirements.txt        # Dependencies (optional)
```

## Optional: Create requirements.txt

```txt
edgebrain>=0.1.1
ollama
```

Then install with:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Explore Examples**: Look at the examples in the EdgeBrain documentation
2. **Add Tools**: Learn how to add custom tools to your agents
3. **Multi-Agent Systems**: Create multiple agents that work together
4. **Memory Systems**: Add persistent memory to your agents
5. **Custom Integration**: Build your own integrations

## Getting Help

- **Documentation**: Check the full documentation in the `docs/` folder
- **Examples**: Look at example scripts for more complex usage
- **GitHub Issues**: Report problems or ask questions on GitHub
- **Community**: Join discussions and share your projects

## Common Issues

### 1. Import Error

```
ImportError: No module named 'edgebrain'
```

**Solution**: Make sure you installed EdgeBrain and your virtual environment is activated:
```bash
pip install edgebrain
```

### 2. Ollama Connection Error

```
Error: Cannot connect to Ollama
```

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### 3. Model Not Found

```
Error: Model 'qwen2.5:3b' not found
```

**Solution**: Pull the required model:
```bash
ollama pull qwen2.5:3b
```

---

🎉 **Congratulations!** You've successfully set up EdgeBrain from PyPI and created your first agents!
