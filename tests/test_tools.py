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
async def test_suggest_models_local_advisor():
    """Test the suggest_models tool as a local model advisor."""
    # Arrange
    mock_client = MagicMock(spec=OllamaClient)

    class FakeModelInfo:
        def __init__(self, name):
            self.name = name

    # Mock the list of local models
    fake_local_models = [
        FakeModelInfo(name="codellama:7b"),
        FakeModelInfo(name="llama3:latest"),
    ]
    mock_client.list_models = AsyncMock(return_value={"success": True, "models": fake_local_models})

    # Mock the details for each model
    def show_side_effect(model_name):
        if model_name == "codellama:7b":
            return {
                "success": True,
                "name": "codellama:7b",
                "modelfile": "# For coding",
                "details": {"family": "llama", "parameter_size": "7B"}
            }
        if model_name == "llama3:latest":
            return {
                "success": True,
                "name": "llama3:latest",
                "modelfile": "# General purpose chat",
                "details": {"family": "llama", "parameter_size": "8B"}
            }
        return {"success": False}

    mock_client.show = AsyncMock(side_effect=show_side_effect)

    # Act
    # Ask for a model for "coding"
    arguments = {"user_needs": "i need a model for coding"}
    result = await handle_advanced_tool("suggest_models", arguments, mock_client)

    # Assert
    mock_client.list_models.assert_awaited_once()
    assert mock_client.show.await_count == 2

    response_text = json.loads(result[0].text)
    assert response_text["success"] is True

    recommendations = response_text["recommendations"]
    assert len(recommendations) > 0
    # The coding model should be ranked highest
    assert recommendations[0]["name"] == "codellama:7b"
    # Check that one of the reasons mentions "coding"
    assert any("coding" in reason for reason in recommendations[0]["reasons"])

@pytest.mark.asyncio
async def test_unknown_tool_handling():
    """Test that both base and advanced handlers reject unknown tools."""

    # Test base handler
    result_base = await handle_base_tool("unknown_tool_name", {}, None)
    assert "Unknown base tool: unknown_tool_name" in result_base[0].text

    # Test advanced handler
    result_adv = await handle_advanced_tool("unknown_tool_name", {}, None)
    assert "Unknown advanced tool: unknown_tool_name" in result_adv[0].text
