import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Since we are using an editable install, we can use absolute imports
from ollama_mcp.tools.base_tools import handle_base_tool
from ollama_mcp.tools.advanced_tools import handle_advanced_tool
from ollama_mcp.client import OllamaClient

@pytest.mark.asyncio
@patch('ollama_mcp.hardware_checker.HardwareChecker')
async def test_system_resource_check_tool(MockHardwareChecker):
    """Test that the system_resource_check tool correctly calls the HardwareChecker."""
    # Arrange
    mock_checker_instance = MockHardwareChecker.return_value
    mock_checker_instance.get_system_info = AsyncMock(return_value={
        "success": True,
        "system_info": {"os_name": "Linux"}
    })

    # Act
    result = await handle_base_tool("system_resource_check", {}, None)

    # Assert
    MockHardwareChecker.assert_called_once()
    mock_checker_instance.get_system_info.assert_awaited_once()

    response_text = json.loads(result[0].text)
    assert response_text["success"] is True
    assert response_text["system_info"]["os_name"] == "Linux"

@pytest.mark.asyncio
@pytest.mark.parametrize("language, user_needs", [
    ("en", "i need a model for coding"),
    ("it", "mi serve un modello per lo sviluppo"),
])
async def test_suggest_models_local_advisor_multilingual(language, user_needs):
    """Test that the concept-based suggestion works for different languages."""
    # Arrange
    mock_client = MagicMock(spec=OllamaClient)

    # Use dicts instead of objects - this matches how client.list_models() returns data
    fake_local_models = [
        {"name": "codellama:7b", "size": 1000, "size_human": "1.0 KB", "modified_at": "2024-01-01", "modified_display": "2024-01-01"},
        {"name": "llama3:latest", "size": 2000, "size_human": "2.0 KB", "modified_at": "2024-01-01", "modified_display": "2024-01-01"},
    ]
    mock_client.list_models = AsyncMock(return_value={"success": True, "models": fake_local_models, "count": 2})

    def show_side_effect(model_name):
        if model_name == "codellama:7b":
            return {"success": True, "name": "codellama:7b", "modelfile": "Model for code generation"}
        if model_name == "llama3:latest":
            return {"success": True, "name": "llama3:latest", "modelfile": "General purpose chat model"}
        return {"success": False}

    mock_client.show = AsyncMock(side_effect=show_side_effect)

    # Act
    arguments = {"user_needs": user_needs}
    result = await handle_advanced_tool("suggest_models", arguments, mock_client)

    # Assert
    response_text = json.loads(result[0].text)
    assert response_text["success"] is True

    recommendations = response_text["recommendations"]
    assert len(recommendations) > 0
    assert recommendations[0]["name"] == "codellama:7b"
    assert "coding" in recommendations[0]["reasons"][0]

@pytest.mark.asyncio
async def test_unknown_tool_handling():
    """Test that both base and advanced handlers reject unknown tools."""

    # Test base handler
    result_base = await handle_base_tool("unknown_tool_name", {}, None)
    assert "Unknown base tool: unknown_tool_name" in result_base[0].text

    # Test advanced handler
    result_adv = await handle_advanced_tool("unknown_tool_name", {}, None)
    assert "Unknown advanced tool: unknown_tool_name" in result_adv[0].text
