# Ollama MCP Server

Self-contained MCP server for comprehensive Ollama management with zero external dependencies.

## Stack

- **Language**: Python 3.10+
- **Framework**: MCP (Model Context Protocol)
- **Dependencies**: mcp, ollama, psutil, aiofiles, PyYAML
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Linting/Formatting**: black (88 chars), isort, flake8, mypy

## Structure

```
ollama-mcp-server/
├── src/
│   ├── __init__.py               # Re-exports version from ollama_mcp
│   └── ollama_mcp/
│       ├── __init__.py           # Package init, exports main classes
│       ├── server.py             # Main MCP server (entry point)
│       ├── client.py             # Ollama async API client + DateTimeEncoder
│       ├── config.py             # Configuration (env vars, timeout=60s)
│       ├── model_manager.py      # Model CRUD operations
│       ├── hardware_checker.py   # Multi-GPU detection, system info
│       ├── ollama_server_control.py # Server lifecycle (start/stop/restart)
│       └── tools/
│           ├── __init__.py
│           ├── base_tools.py     # 4 core tools (list, chat, health, system)
│           └── advanced_tools.py # 5 advanced tools (suggest, remove, start, select, test)
├── tests/
│   ├── test_client.py            # Client unit tests (7 tests)
│   └── test_tools.py             # Tool integration tests (4 tests)
├── pyproject.toml                # Build config & dependencies
├── README.md                     # User documentation
└── CLAUDE.md                     # Developer reference (this file)
```

## Key Components

### Entry Points
- `server.py:main()` - MCP server startup
- `pyproject.toml` script: `ollama-mcp-server`

### Shared Utilities
- `client.py:DateTimeEncoder` - JSON encoder for datetime objects (used by all tools)
- `client.py:ModelInfo` - Dataclass for model metadata
- `config.py:get_config()` - Loads config from environment variables

### Tools (9 total)
**Base (4)**:
1. `list_local_models` - List installed models
2. `local_llm_chat` - Chat with model (auto-selects if not specified)
3. `ollama_health_check` - Server diagnostics
4. `system_resource_check` - Hardware info (CPU, RAM, GPU)

**Advanced (5)**:
1. `suggest_models` - Recommend model for task (concept-based matching)
2. `remove_model` - Delete model from storage
3. `start_ollama_server` - Start Ollama if not running
4. `select_chat_model` - Interactive model selection
5. `test_model_responsiveness` - Latency test

## Conventions

### Code Style
- Line length: 88 (black default)
- Imports: sorted with isort (black profile)
- Type hints: required (mypy strict mode)
- Docstrings: module-level with version info

### Naming
- Classes: `PascalCase` (OllamaClient, ModelManager)
- Functions/methods: `snake_case` (list_models, health_check)
- Constants: `UPPER_SNAKE_CASE` (CONCEPT_KEYWORDS)
- Private methods: `_leading_underscore` (internal only, not called externally)

### Async Patterns
- All I/O operations are async
- Use `asyncio.wait_for()` for timeouts
- Lazy client initialization (starts even if Ollama offline)
- Graceful degradation with error dicts

### Error Handling Pattern
```python
return {
    "success": False,
    "error": "Description of what went wrong",
    "troubleshooting": {
        "step_1": "First thing to try",
        "step_2": "Second thing to try"
    }
}
```

### Tool Response Pattern
```python
async def handle_tool(name, arguments, client) -> List[TextContent]:
    result = await do_something()
    return [TextContent(
        type="text",
        text=json.dumps(result, cls=DateTimeEncoder, indent=2)
    )]
```

## Configuration

Environment variables (all optional):
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | localhost | Ollama server host |
| `OLLAMA_PORT` | 11434 | Ollama server port |
| `OLLAMA_TIMEOUT` | 60 | Request timeout (seconds) |
| `HARDWARE_ENABLE_GPU_DETECTION` | True | Enable GPU detection |
| `HARDWARE_GPU_MEMORY_FRACTION` | 0.9 | Usable GPU memory fraction |
| `HARDWARE_ENABLE_CPU_FALLBACK` | True | Fall back to CPU if no GPU |
| `HARDWARE_MEMORY_THRESHOLD_GB` | 4.0 | RAM warning threshold |

## Running

```bash
# Install (editable)
pip install -e .

# Run MCP server directly
python src/ollama_mcp/server.py

# Run via entry point
ollama-mcp-server

# Run tests
pytest

# Run tests with coverage
pytest --cov=ollama_mcp

# Format code
black src/ && isort src/

# Type check
mypy src/
```

## Testing Notes

- Tests use `unittest.mock` to mock the ollama library
- `@pytest.mark.asyncio` decorator for async tests
- Mock `ollama.AsyncClient` to avoid real API calls
- Test both success and error paths

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows | ✅ Tested | RTX 4090 validated |
| Linux | ⚠️ Needs testing | Code ready |
| macOS | ⚠️ Needs testing | Apple Silicon detection implemented |

---
*Last updated after code review cleanup.*
