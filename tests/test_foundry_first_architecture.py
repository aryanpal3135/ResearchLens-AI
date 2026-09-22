"""
Tests for Microsoft Foundry-First Architecture in ResearchLens AI.
Validates:
1. AgentRouter resolves tasks to correct Foundry-managed agents.
2. MicrosoftFoundryClient initializes with AIProjectClient and DefaultAzureCredential.
3. MicrosoftFoundryClient.discover_agent queries cloud agent definitions.
4. MicrosoftFoundryClient.run_agent routes execution to Foundry Prompt Agent protocols.
5. Zero silent fallback: explicit errors when Foundry is unreachable.
6. ResearchLensAgent dispatches tasks to appropriate Foundry agents.
"""

import pytest
from unittest.mock import MagicMock, patch
from config.settings import settings
from services.agent_router import AgentRouter, AgentTaskType, AgentRoute
from services.foundry_client import MicrosoftFoundryClient
from services.foundry_agent import ResearchLensAgent


def test_agent_router_resolution():
    """Validates that AgentRouter maps tasks to designated Foundry Prompt Agents."""
    router = AgentRouter(
        research_agent_name="researchmate-gpt4-1-mini",
        research_agent_version="1",
        chat_agent_name="Paper-Chat-Agent",
        chat_agent_version="2",
    )

    # Chat task -> Paper-Chat-Agent (v2)
    chat_route = router.resolve(AgentTaskType.PAPER_CHAT)
    assert chat_route.agent_name == "Paper-Chat-Agent"
    assert chat_route.version == "2"

    # Research tasks -> researchmate-gpt4-1-mini (v1)
    for task in [
        AgentTaskType.DEEP_ANALYSIS,
        AgentTaskType.COMPARISON,
        AgentTaskType.RESEARCH_GAPS,
        AgentTaskType.RESEARCH_QUESTIONS,
        AgentTaskType.LITERATURE_REVIEW,
    ]:
        route = router.resolve(task)
        assert route.agent_name == "researchmate-gpt4-1-mini"
        assert route.version == "1"


def test_foundry_client_configuration():
    """Validates MicrosoftFoundryClient configuration properties and endpoint checking."""
    client = MicrosoftFoundryClient(
        project_endpoint="https://researchmate-resource.services.ai.azure.com/api/projects/researchmate",
        research_agent_name="researchmate-gpt4-1-mini",
        chat_agent_name="Paper-Chat-Agent",
    )

    assert client.is_configured is True
    assert client.project_endpoint == "https://researchmate-resource.services.ai.azure.com/api/projects/researchmate"
    assert client.research_agent_name == "researchmate-gpt4-1-mini"
    assert client.chat_agent_name == "Paper-Chat-Agent"


def test_foundry_client_unconfigured_rejection():
    """Validates client rejects invalid or missing project endpoint without silent bypass."""
    client = MicrosoftFoundryClient(project_endpoint="")
    assert client.is_configured is False

    res = client.run_agent(agent_name="Paper-Chat-Agent", user_prompt="Hello")
    assert res["success"] is False
    assert res["status"] == "unconfigured"
    assert "not configured" in res["error"].lower()


def test_foundry_client_agent_discovery_mock():
    """Validates discover_agent queries the Foundry project agents catalog."""
    mock_proj_client = MagicMock()
    mock_agent_record = MagicMock()
    mock_agent_record.id = "agent_id_123"
    mock_agent_record.state = "active"
    mock_agent_record.versions = {"1": MagicMock()}
    mock_proj_client.agents.get.return_value = mock_agent_record

    client = MicrosoftFoundryClient(
        project_endpoint="https://researchmate-resource.services.ai.azure.com/api/projects/researchmate",
        project_client=mock_proj_client,
    )

    discovery = client.discover_agent("researchmate-gpt4-1-mini")
    assert discovery["found"] is True
    assert discovery["name"] == "researchmate-gpt4-1-mini"
    assert discovery["state"] == "active"
    mock_proj_client.agents.get.assert_called_once_with(agent_name="researchmate-gpt4-1-mini")


def test_foundry_client_run_agent_execution_mock():
    """Validates run_agent acquires the agent's OpenAI protocol client and executes chat completions."""
    mock_proj_client = MagicMock()
    mock_openai_client = MagicMock()
    mock_proj_client.get_openai_client.return_value = mock_openai_client

    mock_choice = MagicMock()
    mock_choice.message.content = "Synthesized research answer from Foundry Agent."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 45
    mock_openai_client.chat.completions.create.return_value = mock_response

    client = MicrosoftFoundryClient(
        project_endpoint="https://researchmate-resource.services.ai.azure.com/api/projects/researchmate",
        project_client=mock_proj_client,
    )

    result = client.run_agent(
        agent_name="Paper-Chat-Agent",
        user_prompt="Explain attention mechanism",
        system_prompt="Custom instruction",
        temperature=0.3,
        max_tokens=1000,
    )

    assert result["success"] is True
    assert result["content"] == "Synthesized research answer from Foundry Agent."
    assert result["agent_name"] == "Paper-Chat-Agent"
    assert result["tokens_prompt"] == 150
    assert result["tokens_completion"] == 45
    mock_proj_client.get_openai_client.assert_called_once_with(agent_name="Paper-Chat-Agent")


def test_foundry_client_zero_silent_fallback_on_error():
    """Validates that if Foundry throws an error, it is returned explicitly with NO silent fallback."""
    mock_proj_client = MagicMock()
    mock_openai_client = MagicMock()
    mock_proj_client.get_openai_client.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("Foundry protocol connection timeout")

    client = MicrosoftFoundryClient(
        project_endpoint="https://researchmate-resource.services.ai.azure.com/api/projects/researchmate",
        project_client=mock_proj_client,
    )

    result = client.run_agent(
        agent_name="researchmate-gpt4-1-mini",
        user_prompt="Compare papers",
    )

    assert result["success"] is False
    assert result["status"] == "error"
    assert "Foundry protocol connection timeout" in result["error"]
    assert result["content"] == ""


def test_researchlens_agent_multi_agent_routing():
    """Validates ResearchLensAgent routes tasks to the proper Foundry agent."""
    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.run_agent.return_value = {
        "success": True,
        "content": "Result from Foundry Agent",
        "agent_name": "mocked",
        "model": "gpt-4.1-mini",
        "latency_ms": 100.0,
    }

    agent = ResearchLensAgent(foundry_client=mock_client)

    # Invoke chat QA task
    agent._invoke_foundry_agent(
        task_type=AgentTaskType.PAPER_CHAT,
        user_prompt="Chat query",
    )
    mock_client.run_agent.assert_called_with(
        agent_name="Paper-Chat-Agent",
        user_prompt="Chat query",
        system_prompt=None,
        temperature=0.2,
        max_tokens=4000,
        timeout=60.0,
    )

    # Invoke literature review task
    agent._invoke_foundry_agent(
        task_type=AgentTaskType.LITERATURE_REVIEW,
        user_prompt="Review query",
    )
    mock_client.run_agent.assert_called_with(
        agent_name="researchmate-gpt4-1-mini",
        user_prompt="Review query",
        system_prompt=None,
        temperature=0.2,
        max_tokens=4000,
        timeout=60.0,
    )
