# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-01-02

### Changed
- **Code Cleanup**: Comprehensive code review and refactoring
  - Removed ~600 lines of dead code (`server_manager.py`)
  - Consolidated duplicate `DateTimeEncoder` class into `client.py`
  - Consolidated duplicate `__version__` definition
  - Made `test_model_responsiveness` method public (was `_test_model_responsiveness`)

### Fixed
- **Configuration**: Default timeout now correctly set to 60s (was 30s in code, 60s in docs)
- **Imports**: Removed unused imports (`json` from hardware_checker, `subprocess`/`time`/`datetime` from advanced_tools)
- **Docstrings**: Fixed inconsistent version references and project name in module headers

### Added
- **Developer Documentation**: New `CLAUDE.md` with project conventions, structure, and patterns
- **Code Comments**: Section markers and algorithm documentation for complex functions

### Technical Details
- All 11 unit tests passing
- No breaking changes to public API
- All MCP tools remain fully functional

---

## [0.9.0] - 2025-08-17

### Fixed
- **Critical Bug**: Fixed datetime serialization issue in model listing
  - Resolved `Object of type datetime is not JSON serializable` error
  - All datetime objects now properly converted to ISO format strings
  - Model listing and all other tools now work correctly with Claude Desktop

### Technical Details
- Fixed datetime handling in `list_local_models` tool
- Improved JSON serialization for all API responses
- Enhanced error handling for edge cases
- All tools tested and verified working with Claude Desktop MCP client

### Testing
- Comprehensive test suite run on Windows 11 with RTX 4090
- All 9 tools verified working correctly:
  - ✅ list_local_models
  - ✅ local_llm_chat  
  - ✅ ollama_health_check
  - ✅ system_resource_check
  - ✅ suggest_models
  - ✅ select_chat_model
  - ✅ test_model_responsiveness
  - ✅ start_ollama_server
  - ✅ remove_model

## [Previous Versions]

### [0.8.x] - July-August 2025
- Initial development and architecture setup with Claude AI assistance
- Implementation of core MCP tools
- Cross-platform compatibility framework
- GPU detection and system analysis features

### [Initial Release] - July 2025
- First version by Paolo Dalprato developed with Claude
- Basic Ollama integration and model management
