# Ollama MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A self-contained **Model Context Protocol (MCP) server** for local Ollama management. Features include listing local models, chatting, starting/stopping the server, and a 'local model advisor' to suggest the best local model for a given task. The server is designed to be a robust, dependency-free, and cross-platform tool for managing a local Ollama instance.

## ⚠️ Current Testing Status

**Currently tested on**: Windows 11 with NVIDIA RTX 4090  
**Status**: Beta on Windows, Other Platforms Need Testing  
**Cross-platform code**: Ready for Linux and macOS but requires community testing  
**GPU support**: NVIDIA fully tested, AMD/Intel/Apple Silicon implemented but needs validation

We welcome testers on different platforms and hardware configurations! Please report your experience via GitHub Issues.

## 🎯 Key Features

### 🔧 **Self-Contained Architecture**
- **Zero External Dependencies**: No external MCP servers required
- **MIT License Ready**: All code internally developed and properly licensed
- **Enterprise-Grade**: Professional error handling with actionable troubleshooting

### 🌐 **Universal Compatibility**
- **Cross-Platform**: Windows, Linux, macOS with automatic platform detection
- **Multi-GPU Support**: NVIDIA, AMD, Intel detection with vendor-specific optimizations
- **Smart Installation Discovery**: Automatic Ollama detection across platforms

### ⚡ **Complete Local Ollama Management**
- **Model Operations**: List, suggest, and remove local models.
- **Server Control**: Start and monitor the Ollama server with intelligent process management.
- **Direct Chat**: Communicate with any locally installed model.
- **System Analysis**: Assess hardware compatibility and monitor resources.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/paolodalprato/ollama-mcp-server.git
cd ollama-mcp-server
pip install -e .
```

### Configuration

Add to your MCP client configuration (e.g., Claude Desktop `config.json`):

```json
{
  "mcpServers": {
    "ollama-mcp": {
      "command": "python",
      "args": [
        "X:\\PATH_TO\\ollama-mcp-server\\src\\ollama_mcp\\server.py"
      ],
      "env": {}
    }
  }
}
```

**Note**: Adjust the path to match your installation directory. On Linux/macOS, use forward slashes: `/path/to/ollama-mcp-server/src/ollama_mcp/server.py`

### Requirements

- **Python 3.8+**
- **Ollama installed** and accessible in PATH
- **MCP-compatible client** (Claude Desktop, etc.)

## 🛠️ Available Tools

### **Model Management**
- `list_local_models` - List all locally installed models with their details.
- `local_llm_chat` - Chat directly with any locally installed model.
- `remove_model` - Safely remove a model from local storage.
- `suggest_models` - Recommends the best **locally installed** model for a specific task (e.g., "suggest a model for coding").

### **Server and System Operations**
- `start_ollama_server` - Starts the Ollama server if it's not already running.
- `ollama_health_check` - Performs a comprehensive health check of the Ollama server.
- `system_resource_check` - Analyzes system hardware and resource availability.

### **Diagnostics**
- `test_model_responsiveness` - Checks the responsiveness of a specific local model by sending a test prompt, helping to diagnose performance issues.
- `select_chat_model` - Presents a list of available local models to choose from before starting a chat.

## 💬 How to Interact with Ollama-MCP

Ollama-MCP works **through your MCP client** (like Claude Desktop) - you don't interact with it directly. Instead, you communicate with your MCP client using **natural language**, and the client translates your requests into tool calls.

### **Basic Interaction Pattern**

You speak to your MCP client in natural language, and it automatically uses the appropriate ollama-mcp tools:

```
You: "List my installed Ollama models"
→ Client calls: list_local_models
→ You get: Formatted list of your models

You: "Chat with llama3.2: explain machine learning"  
→ Client calls: local_llm_chat with model="llama3.2" and message="explain machine learning"
→ You get: AI response from your local model

You: "Check if Ollama is running"
→ Client calls: ollama_health_check  
→ You get: Server status and troubleshooting if needed
```

### **Example Interactions**

#### **Model Management**
- *"What models do I have installed?"* → `list_local_models`
- *"I need a model for creative writing, which of my models is best?"* → `suggest_models`
- *"Remove the old mistral model to save space"* → `remove_model`

#### **System Operations**
- *"Start Ollama server"* → `start_ollama_server`
- *"Is my system capable of running large AI models?"* → `system_resource_check`

#### **AI Chat**
- *"Chat with llama3.2: write a Python function to sort a list"* → `local_llm_chat`
- *"Use deepseek-coder to debug this code: [code snippet]"* → `local_llm_chat`
- *"Ask phi3.5 to explain quantum computing simply"* → `local_llm_chat`

### **Key Points**

- **No Direct Commands**: You never call `ollama_health_check()` directly
- **Natural Language**: Speak normally to your MCP client
- **Automatic Tool Selection**: The client chooses the right tool based on your request
- **Conversational**: You can ask follow-up questions and the client maintains context

## 🎯 Real-World Use Cases

### **Daily Development Workflow**
*"I need to work on a coding project. Which of my local models is best for coding? Let's check its performance and then ask it a question."*

This could trigger:
1. `suggest_models` - Recommends the best local model for "coding".
2. `test_model_responsiveness` - Checks if the recommended model is responsive.
3. `local_llm_chat` - Starts a chat with the model.

### **Model Management Session**
*"Show me what models I have and recommend one for writing a story. Then let's clean up any old models I don't need."*

Triggers:
1. `list_local_models` - Current inventory
2. `suggest_models` - Recommends a local model for "writing a story".
3. `remove_model` - Cleanup unwanted models.

### **Troubleshooting Session**
*"Ollama isn't working. Check what's wrong, try to fix it, and test with a simple chat."*

Triggers:
1. `ollama_health_check` - Diagnose issues
2. `start_ollama_server` - Attempt to start server
3. `local_llm_chat` - Verify working with test message

## 🏗️ Architecture

### **Design Principles**
- **Self-Contained**: Zero external MCP server dependencies
- **Fail-Safe**: Comprehensive error handling with actionable guidance
- **Cross-Platform First**: Universal Windows/Linux/macOS compatibility
- **Enterprise Ready**: Professional-grade implementation and documentation

### **Technical Highlights**
- **Internal Process Management**: Advanced subprocess handling with timeout control
- **Multi-GPU Detection**: Platform-specific GPU identification without confusing metrics
- **Intelligent Model Selection**: Fallback to first available model when none specified
- **Progressive Health Monitoring**: Smart server startup detection with detailed feedback

## 📋 System Compatibility

### **Operating Systems**
- **Windows**: Full support with auto-detection in Program Files and AppData ✅ **Tested**
- **Linux**: XDG configuration support with package manager integration ⚠️ **Needs Testing**
- **macOS**: Homebrew detection with Apple Silicon GPU support ⚠️ **Needs Testing**

### **GPU Support**
- **NVIDIA**: Full detection via nvidia-smi with memory and utilization info ✅ **Tested RTX 4090**
- **AMD**: ROCm support via vendor-specific tools ⚠️ **Needs Testing**
- **Intel**: Basic detection via system tools ⚠️ **Needs Testing**
- **Apple Silicon**: M1/M2/M3 detection with unified memory handling ⚠️ **Needs Testing**

### **Hardware Requirements**
- **Minimum**: 4GB RAM, 2GB free disk space
- **Recommended**: 8GB+ RAM, 10GB+ free disk space
- **GPU**: Optional but recommended for model acceleration

## 🔧 Development

### **Project Structure**
```
ollama-mcp-server/
├── src/
│   ├── __init__.py               # Defines the package version
│   └── ollama_mcp/
│       ├── __init__.py           # Makes 'ollama_mcp' a package
│       ├── server.py             # Main MCP server implementation
│       ├── client.py             # Ollama API client
│       ├── config.py             # Configuration management
│       ├── model_manager.py      # Local model operations
│       ├── hardware_checker.py   # System hardware analysis
│       └── ... (and other modules)
├── tests/
│   ├── test_client.py            # Unit tests for the client
│   └── test_tools.py             # Integration tests for tools
├── .gitignore                    # Specifies intentionally untracked files
└── pyproject.toml                # Project configuration and dependencies
```

### **Key Technical Achievements**

#### **Self-Contained Implementation**
- **Challenge**: Eliminated external `desktop-commander` dependency
- **Solution**: Internal process management with advanced subprocess handling
- **Result**: Zero external MCP dependencies, MIT license compatible

#### **Intelligent GPU Detection**
- **Challenge**: Complex VRAM reporting causing user confusion
- **Solution**: Simplified to GPU name display only
- **Result**: Clean, reliable hardware identification

#### **Enterprise Error Handling**
- **Implementation**: 6-level exception framework with specific error types
- **Coverage**: Platform-specific errors, process failures, network issues
- **UX**: Actionable troubleshooting steps for every error scenario

## 🤝 Contributing

We welcome contributions! Areas where help is especially appreciated:

- **Platform Testing**: Different OS and hardware configurations ⭐ **High Priority**
- **GPU Vendor Support**: Additional vendor-specific detection
- **Performance Optimization**: Startup time and resource usage improvements
- **Documentation**: Usage examples and integration guides
- **Testing**: Edge cases and error condition validation

### **Immediate Testing Needs**
- **Linux**: Ubuntu, Fedora, Arch with various GPU configurations
- **macOS**: Intel and Apple Silicon Macs with different Ollama installations
- **GPU Vendors**: AMD ROCm, Intel Arc, Apple unified memory
- **Edge Cases**: Different Python versions, various Ollama installation methods

### **Development Setup**
```bash
git clone https://github.com/paolodalprato/ollama-mcp-server.git
cd ollama-mcp-server

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

## 🐛 Troubleshooting

### **Common Issues**

#### **Ollama Not Found**
```bash
# Verify Ollama installation
ollama --version

# Check PATH configuration
which ollama  # Linux/macOS
where ollama  # Windows
```

#### **Server Startup Failures**
```bash
# Check port availability
netstat -an | grep 11434

# Manual server start for debugging
ollama serve
```

#### **Permission Issues**
- **Windows**: Run as Administrator if needed
- **Linux/macOS**: Check user permissions for service management

### **Platform-Specific Issues**
If you encounter issues on Linux or macOS, please report them via GitHub Issues with:
- Operating system and version
- Python version
- Ollama version and installation method
- GPU hardware (if applicable)
- Complete error output

## 📊 Performance

### **Typical Response Times** *(Windows RTX 4090)*
- **Health Check**: <500ms
- **Model List**: <1 second
- **Server Start**: 1-15 seconds (hardware dependent)
- **Model Chat**: 2-30 seconds (model and prompt dependent)

### **Resource Usage**
- **Memory**: <50MB for MCP server process
- **CPU**: Minimal when idle, scales with operations
- **Storage**: Configuration files and logs only

## 🔐 Security

- **Data Flow**: User → MCP Client (Claude) → ollama-mcp-server → Local Ollama → back through chain

## 👨‍💻 About This Project

This is my first MCP server, created by adapting a personal tool I had developed for my own Ollama management needs.

### **The Problem I Faced**
I started using Claude to interact with Ollama because it allows me to use natural language instead of command-line interfaces. Claude also provides capabilities that Ollama alone doesn't have, particularly intelligent model suggestions based on both my system capabilities and specific needs.

### **My Solution**
I built this MCP server to streamline my own workflow, and then refined it into a stable tool that others might find useful. The design reflects real usage patterns:

- **Self-contained**: No external dependencies that can break
- **Intelligent error handling**: Clear guidance when things go wrong
- **Cross-platform**: Works consistently across different environments  
- **Practical tools**: Features I actually use in daily work

### **Design Philosophy**
I initially developed this for my personal use to manage Ollama models more efficiently. When the MCP protocol became available, I transformed my personal tool into an MCP server to share it with others who might find it useful.

**Development Approach**: This project was developed using "vibe coding" with Claude - an iterative, conversational development process where AI assistance helped refine both the technical implementation and user experience. It's a practical example of AI-assisted development creating tools for AI management.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team**: For the excellent local AI platform
- **MCP Project**: For the Model Context Protocol specification
- **Claude Desktop**: For MCP client implementation and testing platform
- **Community**: For testing, feedback, and contributions

## 📞 Support

- **Bug Reports**: [GitHub Issues](https://github.com/paolodalprato/ollama-mcp-server/issues)
- **Feature Requests**: [GitHub Issues](https://github.com/paolodalprato/ollama-mcp-server/issues)
- **Community Discussion**: [GitHub Discussions](https://github.com/paolodalprato/ollama-mcp-server/discussions)

---

## Changelog

*   **August 2025:** Project refactoring and enhancement by Jules. Overhauled the architecture for modularity, implemented a fully asynchronous client, added a test suite, and refined the tool logic based on a "local-first" philosophy.
*   **July 2025:** Initial version created by Paolo Dalprato.

---

**Status**: Beta on Windows, Other Platforms Need Testing  
**Testing**: Windows 11 + RTX 4090 validated, Linux/macOS require community validation  
**License**: MIT  
**Dependencies**: Zero external MCP servers required
