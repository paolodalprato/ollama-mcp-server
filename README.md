# Ollama MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A self-contained **Model Context Protocol (MCP) server** for comprehensive Ollama management. Zero external dependencies, enterprise-grade error handling, and complete cross-platform compatibility.

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

### ⚡ **Complete Ollama Management**
- **Model Operations**: Download, remove, list models with progress tracking
- **Server Control**: Start and monitor Ollama server with intelligent process management
- **Direct Chat**: Local model communication with fallback model selection
- **System Analysis**: Hardware compatibility assessment and resource monitoring

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
- `list_local_models` - List installed models with details
- `local_llm_chat` - Chat directly with local models
- `download_model` - Download models with progress tracking
- `remove_model` - Safely remove models from storage

### **Server Operations**
- `start_ollama_server` - Start Ollama server (self-contained implementation)
- `ollama_health_check` - Comprehensive server health diagnostics
- `system_resource_check` - Hardware compatibility analysis

### **Advanced Features**
- `suggest_models` - AI-powered model recommendations based on your needs
- `search_available_models` - Search Ollama Hub by category
- `check_download_progress` - Monitor download progress with visual indicators
- `select_chat_model` - Interactive model selection interface

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
- *"Download qwen2.5-coder for coding tasks"* → `download_model`
- *"Remove the old mistral model to save space"* → `remove_model`
- *"Show me coding-focused models I can download"* → `search_available_models`

#### **System Operations**
- *"Start Ollama server"* → `start_ollama_server`
- *"Is my system capable of running large AI models?"* → `system_resource_check`
- *"Recommend a model for creative writing"* → `suggest_models`

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
*"I need to set up local AI for coding. Check my system, recommend a good coding model, download it, and test it with a simple Python question."*

This would automatically trigger:
1. `system_resource_check` - Verify hardware capability
2. `suggest_models` - Recommend coding-focused models
3. `download_model` - Download the recommended model
4. `local_llm_chat` - Test with your Python question

### **Model Management Session**
*"Show me what models I have, see what new coding models are available, and clean up any old models I don't need."*

Triggers:
1. `list_local_models` - Current inventory
2. `search_available_models` - Browse new options
3. `remove_model` - Cleanup unwanted models

### **Troubleshooting Session**
*"Ollama isn't working. Check what's wrong, try to fix it, and test with a simple chat."*

Triggers:
1. `ollama_health_check` - Diagnose issues
2. `start_ollama_server` - Attempt to start server
3. `local_llm_chat` - Verify working with test message

### **Research and Development**
*"I'm working on a machine learning project. Suggest models good for code analysis, download the best one, and help me understand transformer architectures."*

Combines multiple tools for a complete workflow from model selection to technical learning.

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
├── src/ollama_mcp/
│   ├── server.py              # Main MCP server implementation
│   ├── client.py              # Ollama client interface
│   ├── tools/
│   │   ├── base_tools.py      # Essential 4 tools
│   │   └── advanced_tools.py  # Extended 7 tools
│   ├── config.py              # Configuration management
│   ├── model_manager.py       # Model operations
│   ├── job_manager.py         # Background task management
│   └── hardware_checker.py    # System analysis
└── pyproject.toml            # Project configuration
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

**Status**: Beta on Windows, Other Platforms Need Testing  
**Testing**: Windows 11 + RTX 4090 validated, Linux/macOS require community validation  
**License**: MIT  
**Dependencies**: Zero external MCP servers required
