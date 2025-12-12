# Contributing to Ollama MCP Server

Thank you for your interest in contributing to the Ollama MCP Server! This guide will help you get started with contributing to this project.

## 🎯 Project Goals

Our mission is to provide a **self-contained, enterprise-grade MCP server** for comprehensive Ollama management with:
- Zero external dependencies
- Cross-platform compatibility (Windows/Linux/macOS)
- Professional error handling and troubleshooting
- Clean, maintainable code architecture

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher (required by MCP SDK dependency)
- Git
- Ollama installed and accessible
- MCP-compatible client for testing (e.g., Claude Desktop)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/ollama-mcp-server.git
   cd ollama-mcp-server
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation**
   ```bash
   python -m ollama_mcp.server --help
   ```

## 🛠️ Development Workflow

### Code Standards

We maintain high code quality standards:

#### **Code Formatting**
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

#### **Type Checking**
```bash
# Run mypy type checking
mypy src/
```

#### **Linting**
```bash
# Run flake8 linting
flake8 src/ tests/
```

### Testing

#### **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ollama_mcp

# Run specific test file
pytest tests/test_base_tools.py

# Run with verbose output
pytest -v
```

#### **Test Categories**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test tool interactions with Ollama
- **Platform Tests**: Test cross-platform compatibility
- **Error Tests**: Test error handling and edge cases

### Git Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

4. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Contribution Guidelines

### Code Quality Requirements

#### **Self-Contained Principle**
- **No External MCP Dependencies**: All functionality must be implemented internally
- **Standard Library Preferred**: Use Python standard library when possible
- **Minimal Dependencies**: New dependencies require justification and team approval

#### **Cross-Platform Compatibility**
- **Test on Multiple Platforms**: Windows, Linux, macOS when possible
- **Platform-Specific Code**: Clearly documented and properly conditionally executed
- **Path Handling**: Use `pathlib` for cross-platform path operations

#### **Error Handling Excellence**
- **Specific Exception Types**: Use appropriate exception types (FileNotFoundError, OSError, etc.)
- **Actionable Error Messages**: Include troubleshooting steps in error responses
- **Graceful Degradation**: Non-critical failures shouldn't crash the entire operation

### Code Architecture

#### **Tool Implementation**
```python
# Example tool structure
async def _handle_your_tool(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Tool description with clear purpose"""
    try:
        # Input validation
        required_param = arguments.get("required_param", "")
        if not required_param:
            return error_response("Parameter required", "usage_example")
        
        # Core logic
        result = await your_operation(required_param)
        
        # Success response
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "result": result,
                "next_steps": "What user can do next"
            }, indent=2)
        )]
        
    except SpecificException as e:
        return error_response(f"Specific error: {e}", troubleshooting_steps)
    except Exception as e:
        return error_response(f"Unexpected error: {e}", generic_troubleshooting)
```

#### **Response Format Standards**
```json
{
  "success": true|false,
  "data": "...",
  "error": "Specific error description (if failed)",
  "troubleshooting": {
    "step_1": "First troubleshooting step",
    "step_2": "Second troubleshooting step"
  },
  "next_steps": "What user should do next"
}
```

## 🎯 Areas for Contribution

### High-Priority Areas

#### **Platform Testing**
- Test on different operating systems
- Validate GPU detection across hardware configurations
- Test with various Ollama installation methods

#### **GPU Vendor Support**
- AMD ROCm integration improvements
- Intel Arc GPU support
- Apple Silicon optimization enhancements

#### **Performance Optimization**
- Startup time improvements
- Memory usage optimization
- Response time benchmarking

#### **Error Handling Enhancement**
- Additional edge case coverage
- Improved troubleshooting guidance
- Platform-specific error handling

### Medium-Priority Areas

#### **Documentation**
- Usage examples and tutorials
- Architecture documentation
- API reference improvements

#### **Testing**
- Comprehensive test suite expansion
- Integration test improvements
- Edge case validation

#### **Tooling**
- CI/CD pipeline enhancements
- Development environment improvements
- Code quality automation

## 🐛 Bug Reports

### Before Reporting
1. **Search existing issues** to avoid duplicates
2. **Test with latest version** to ensure bug still exists
3. **Gather system information** (OS, Python version, Ollama version)

### Bug Report Template
```markdown
**Describe the Bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Configure MCP server with...
2. Run tool '...'
3. See error

**Expected Behavior**
What you expected to happen.

**System Information**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python: [e.g., 3.10.0]
- Ollama: [e.g., 0.3.0]
- MCP Client: [e.g., Claude Desktop]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature.

**Problem it Solves**
What problem does this feature address?

**Proposed Solution**
Detailed description of how the feature should work.

**Alternatives Considered**
Other approaches you've considered.

**Implementation Notes**
Technical considerations or constraints.
```

## 📚 Documentation Standards

### Code Documentation
- **Docstrings**: All public functions and classes
- **Type Hints**: Complete type annotations
- **Comments**: Complex logic explanation
- **Examples**: Usage examples in docstrings

### User Documentation
- **Clear Instructions**: Step-by-step guidance
- **Examples**: Real-world usage scenarios
- **Troubleshooting**: Common issues and solutions
- **Platform Notes**: OS-specific considerations

## 🔍 Code Review Process

### Review Criteria
- **Functionality**: Does it work as intended?
- **Code Quality**: Follows project standards?
- **Testing**: Adequate test coverage?
- **Documentation**: Properly documented?
- **Compatibility**: Cross-platform considerations?
- **Performance**: No significant performance regression?

### Review Checklist
- [ ] Code follows formatting standards (Black, isort)
- [ ] Type checking passes (mypy)
- [ ] Tests pass on all supported platforms
- [ ] Documentation updated for new features
- [ ] Error handling follows project patterns
- [ ] No external MCP dependencies introduced

## 🏆 Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **README.md**: Special thanks section

## 📞 Getting Help

- **Development Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Feature Ideas**: GitHub Issues with feature request template
- **General Chat**: GitHub Discussions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Ollama MCP Server better! 🚀
