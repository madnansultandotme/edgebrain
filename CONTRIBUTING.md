# Contributing to EdgeBrain

Thank you for your interest in contributing to the Ollama Agentic Framework! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Standards](#development-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow these guidelines to ensure a welcoming environment for everyone.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Git installed and configured
- Ollama installed and running
- Basic understanding of async/await programming
- Familiarity with AI/ML concepts

### First Contribution

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ollama-agentic-framework.git
   cd ollama-agentic-framework
   ```
3. **Set up the development environment** (see Development Setup below)
4. **Find an issue** to work on or propose a new feature
5. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Install the package in development mode:

```bash
pip install -e .
pip install -r requirements-dev.txt  # If available
```

### 3. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

Run the test suite to ensure everything is working:

```bash
python -m pytest tests/ -v
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve or add documentation
5. **Examples**: Create new examples or improve existing ones
6. **Testing**: Add or improve test coverage

### What We're Looking For

Priority areas for contributions:

- **New Tools**: Implement additional tools for agents
- **Performance Improvements**: Optimize existing code
- **Documentation**: Improve clarity and completeness
- **Examples**: Real-world use cases and tutorials
- **Testing**: Increase test coverage
- **Bug Fixes**: Address reported issues

### What We're Not Looking For

Please avoid:

- Breaking changes without discussion
- Large refactoring without prior approval
- Features that significantly increase complexity
- Code that doesn't follow our standards
- Contributions without tests

## Pull Request Process

### 1. Before You Start

- Check existing issues and PRs to avoid duplication
- For large changes, create an issue first to discuss the approach
- Ensure your idea aligns with the project's goals

### 2. Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run tests** to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
   python -m pytest tests/ --cov=src --cov-report=html
   ```

6. **Run linting**:
   ```bash
   flake8 src/ tests/
   black src/ tests/
   isort src/ tests/
   ```

### 3. Submitting Your PR

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: descriptive commit message"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/descriptive-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Screenshots or examples if applicable
   - Checklist of completed items

### 4. PR Review Process

- All PRs require at least one review from a maintainer
- Address feedback promptly and professionally
- Keep PRs focused and reasonably sized
- Be prepared to make changes based on feedback

### 5. PR Checklist

Before submitting, ensure your PR:

- [ ] Follows the coding standards
- [ ] Includes appropriate tests
- [ ] Updates documentation if needed
- [ ] Passes all existing tests
- [ ] Has a clear, descriptive title
- [ ] References related issues
- [ ] Includes examples if adding new features

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Environment details**:
   - Python version
   - Operating system
   - Ollama version
   - Framework version
3. **Steps to reproduce** the issue
4. **Expected behavior**
5. **Actual behavior**
6. **Error messages** or logs
7. **Minimal code example** if possible

### Feature Requests

For feature requests, provide:

1. **Clear description** of the proposed feature
2. **Use case** explaining why it's needed
3. **Proposed implementation** if you have ideas
4. **Alternatives considered**
5. **Additional context** or examples

### Issue Templates

Use our issue templates when available:

- Bug Report Template
- Feature Request Template
- Documentation Improvement Template

## Development Standards

### Code Style

We follow these coding standards:

1. **PEP 8** for Python code style
2. **Black** for code formatting
3. **isort** for import sorting
4. **Type hints** for all function signatures
5. **Docstrings** for all public methods and classes

### Code Quality

- **Single Responsibility**: Each class/function should have one clear purpose
- **DRY Principle**: Don't repeat yourself
- **SOLID Principles**: Follow object-oriented design principles
- **Error Handling**: Proper exception handling and logging
- **Performance**: Consider performance implications of your code

### Naming Conventions

- **Classes**: PascalCase (e.g., `AgentOrchestrator`)
- **Functions/Methods**: snake_case (e.g., `create_agent`)
- **Variables**: snake_case (e.g., `agent_id`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- **Private methods**: Leading underscore (e.g., `_internal_method`)

### Documentation Standards

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Include type hints for all parameters and return values
- **Comments**: Explain complex logic, not obvious code
- **README updates**: Update README for new features

Example docstring:

```python
async def create_agent(
    self,
    agent_id: str,
    role: str,
    description: str,
    model: Optional[str] = None
) -> Agent:
    """Create and register a new agent.
    
    Args:
        agent_id: Unique identifier for the agent
        role: Role of the agent (e.g., "Research Specialist")
        description: Description of the agent's purpose
        model: Optional specific model to use for the agent
        
    Returns:
        The created Agent instance
        
    Raises:
        ValueError: If agent_id already exists
        ConnectionError: If unable to connect to Ollama
        
    Example:
        >>> agent = await orchestrator.create_agent(
        ...     agent_id="researcher_001",
        ...     role="Research Specialist",
        ...     description="Conducts research and analysis"
        ... )
    """
```

## Testing

### Test Requirements

- **Unit tests** for all new functionality
- **Integration tests** for component interactions
- **Test coverage** should be maintained above 90%
- **Mock external dependencies** (Ollama, databases, etc.)

### Test Structure

```
tests/
├── unit/
│   ├── test_agent.py
│   ├── test_orchestrator.py
│   └── test_tools.py
├── integration/
│   ├── test_agent_integration.py
│   └── test_workflow_integration.py
├── fixtures/
│   ├── mock_data.py
│   └── test_helpers.py
└── conftest.py
```

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestAgent:
    @pytest.fixture
    def mock_ollama_integration(self):
        mock = AsyncMock()
        mock.generate_response.return_value = OllamaResponse(
            content="Test response",
            model="llama3.1"
        )
        return mock
    
    @pytest.mark.asyncio
    async def test_agent_creation(self, mock_ollama_integration):
        """Test agent creation with valid parameters."""
        agent = Agent(
            agent_id="test_agent",
            role="Test Role",
            description="Test description",
            ollama_integration=mock_ollama_integration,
            tool_registry=MagicMock(),
            memory_manager=MagicMock()
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.role == "Test Role"
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_agent.py

# Run with verbose output
python -m pytest -v

# Run only failed tests
python -m pytest --lf
```

## Documentation

### Documentation Types

1. **API Documentation**: Docstrings and type hints
2. **User Guides**: How-to guides and tutorials
3. **Architecture Documentation**: Design and implementation details
4. **Examples**: Working code examples

### Documentation Standards

- **Clear and concise**: Easy to understand
- **Complete**: Cover all functionality
- **Up-to-date**: Keep in sync with code changes
- **Examples**: Include practical examples
- **Cross-references**: Link related concepts

### Building Documentation

```bash
# Generate API documentation
python -m pydoc -w src/

# Build documentation site (if using Sphinx)
cd docs/
make html
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag the release
- [ ] Publish to PyPI (maintainers only)

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: For sensitive issues or private communication

### Resources

- [Project Documentation](docs/)
- [API Reference](docs/api_reference.md)
- [Usage Guide](docs/usage_guide.md)
- [Architecture Overview](docs/architecture.md)

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **Documentation**: Author attribution where appropriate

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Ollama Agentic Framework! Your contributions help make this project better for everyone.

