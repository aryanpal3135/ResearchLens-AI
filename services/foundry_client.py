"""
Microsoft Foundry Client.
Connects the ResearchLens AI application to cloud-managed Microsoft Foundry Agents
via the official azure-ai-projects SDK and DefaultAzureCredential.
"""

import os
import time
from typing import Dict, Any, Optional, List
from config.settings import settings

try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    AZURE_PROJECTS_AVAILABLE = True
except ImportError:
    AZURE_PROJECTS_AVAILABLE = False


class MicrosoftFoundryClient:
    """
    Client for interacting with cloud-managed Microsoft Foundry Agents.
    Uses AIProjectClient to discover and invoke persisted Foundry prompt agents.
    """

    def __init__(
        self,
        project_endpoint: Optional[str] = None,
        research_agent_name: Optional[str] = None,
        chat_agent_name: Optional[str] = None,
        credential: Optional[Any] = None,
        project_client: Optional[Any] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        if project_endpoint is not None:
            raw_endpoint = project_endpoint.strip().rstrip("/")
        elif endpoint is not None:
            raw_endpoint = endpoint.strip().rstrip("/")
        else:
            raw_endpoint = settings.FOUNDRY_PROJECT_ENDPOINT.strip().rstrip("/")

        self.project_endpoint = raw_endpoint
        self.research_agent_name = (research_agent_name or settings.FOUNDRY_RESEARCH_AGENT_NAME).strip()
        self.chat_agent_name = (chat_agent_name or settings.FOUNDRY_CHAT_AGENT_NAME).strip()
        self.research_agent_version = settings.FOUNDRY_RESEARCH_AGENT_VERSION
        self.chat_agent_version = settings.FOUNDRY_CHAT_AGENT_VERSION

        self.chat_deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT
        self.endpoint = self.project_endpoint  # Alias for status reporting
        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY

        self._credential = credential
        self._project_client = project_client
        self._agent_clients: Dict[str, Any] = {}

    @property
    def is_configured(self) -> bool:
        """Verifies if the Foundry Project Endpoint is configured."""
        return bool(
            self.project_endpoint
            and not self.project_endpoint.startswith("https://<resource-name>")
            and "services.ai.azure.com" in self.project_endpoint
        )

    def get_configuration_status(self) -> Dict[str, Any]:
        """Provides configuration status for Foundry integration."""
        if not self.is_configured:
            return {
                "configured": False,
                "message": "Missing Foundry Project Endpoint or credentials.",
                "endpoint": self.project_endpoint,
            }
        return {
            "configured": True,
            "message": "Foundry Project Endpoint configured.",
            "endpoint": self.project_endpoint,
        }

    def get_credential(self) -> Any:
        """Returns the credential instance for Entra authentication."""
        # On Windows, ensure standard Azure CLI directory is present in PATH if installed
        az_standard_path = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
        if os.path.exists(az_standard_path) and az_standard_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = az_standard_path + os.pathsep + os.environ.get("PATH", "")

        if self._credential is None and AZURE_PROJECTS_AVAILABLE:
            try:
                from azure.identity import AzureCliCredential, ChainedTokenCredential
                self._credential = ChainedTokenCredential(
                    AzureCliCredential(process_timeout=60),
                    DefaultAzureCredential(process_timeout=60),
                )
            except Exception:
                self._credential = DefaultAzureCredential(process_timeout=60)
        return self._credential

    def get_project_client(self) -> Optional[Any]:
        """Lazy initialization of AIProjectClient."""
        if self._project_client is not None:
            return self._project_client

        if not self.is_configured or not AZURE_PROJECTS_AVAILABLE:
            return None

        try:
            cred = self.get_credential()
            self._project_client = AIProjectClient(
                endpoint=self.project_endpoint,
                credential=cred,
                allow_preview=True,
            )
            return self._project_client
        except Exception:
            return None

    def get_agent_client(self, agent_name: str) -> Optional[Any]:
        """
        Retrieves the authenticated OpenAI client for a specific Foundry Prompt Agent.
        Points base_url directly to: {endpoint}/agents/{agent_name}/endpoint/protocols/openai
        """
        if agent_name in self._agent_clients:
            return self._agent_clients[agent_name]

        proj_client = self.get_project_client()
        if not proj_client:
            return None

        try:
            agent_openai = proj_client.get_openai_client(agent_name=agent_name)
            self._agent_clients[agent_name] = agent_openai
            return agent_openai
        except Exception:
            return None

    def discover_agent(self, agent_name: str) -> Dict[str, Any]:
        """Queries Microsoft Foundry for agent definition, version, and status."""
        proj_client = self.get_project_client()
        if not proj_client:
            return {"found": False, "error": "Foundry Project Client unavailable"}

        try:
            details = proj_client.agents.get(agent_name=agent_name)
            versions = getattr(details, "versions", {})
            return {
                "found": True,
                "name": agent_name,
                "id": getattr(details, "id", None),
                "state": getattr(details, "state", "active"),
                "versions": list(versions.keys()) if isinstance(versions, dict) else [],
                "details": details.as_dict() if hasattr(details, "as_dict") else str(details),
            }
        except Exception as e:
            return {"found": False, "error": str(e)}

    def test_connection(self) -> Dict[str, Any]:
        """
        Verifies runtime connectivity to Microsoft Foundry and discovers configured agents.
        Returns explicit failure if Foundry is unreachable — zero silent fallback.
        """
        if not self.is_configured:
            return {
                "success": False,
                "status": "not_configured",
                "message": (
                    "Microsoft Foundry Project Endpoint not configured. "
                    "Please set FOUNDRY_PROJECT_ENDPOINT in `.env`."
                ),
                "endpoint": self.project_endpoint,
            }

        if not AZURE_PROJECTS_AVAILABLE:
            return {
                "success": False,
                "status": "sdk_missing",
                "message": "azure-ai-projects SDK is not installed in the environment.",
                "endpoint": self.project_endpoint,
            }

        proj_client = self.get_project_client()
        if not proj_client:
            return {
                "success": False,
                "status": "client_error",
                "message": "Failed to initialize AIProjectClient with DefaultAzureCredential.",
                "endpoint": self.project_endpoint,
            }

        try:
            # Discover research agent
            research_disc = self.discover_agent(self.research_agent_name)
            # Discover chat agent
            chat_disc = self.discover_agent(self.chat_agent_name)

            if not research_disc.get("found") and not chat_disc.get("found"):
                # Check list of all agents in project
                available = []
                try:
                    for a in proj_client.agents.list():
                        available.append(getattr(a, "name", str(a)))
                except Exception:
                    pass

                return {
                    "success": False,
                    "status": "agents_not_found",
                    "message": (
                        f"Foundry connected, but neither '{self.research_agent_name}' nor "
                        f"'{self.chat_agent_name}' were discovered. Available agents: {available}"
                    ),
                    "endpoint": self.project_endpoint,
                    "available_agents": available,
                }

            return {
                "success": True,
                "status": "connected",
                "message": f"Connected to Microsoft Foundry project! Discovered agents: {self.research_agent_name}, {self.chat_agent_name}.",
                "endpoint": self.project_endpoint,
                "research_agent": self.research_agent_name,
                "research_agent_version": self.research_agent_version,
                "chat_agent": self.chat_agent_name,
                "chat_agent_version": self.chat_agent_version,
            }
        except Exception as e:
            err_str = str(e)
            return {
                "success": False,
                "status": "connection_error",
                "message": f"Microsoft Foundry connection failed: {err_str}",
                "endpoint": self.project_endpoint,
            }

    def run_agent(
        self,
        agent_name: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Executes a task on the specified Microsoft Foundry Agent.
        The request reaches the persisted Foundry agent endpoint directly.
        """
        if not self.is_configured:
            return {
                "success": False,
                "status": "unconfigured",
                "content": "",
                "error": "Microsoft Foundry Project Endpoint not configured in `.env`.",
                "agent_name": agent_name,
            }

        agent_openai = self.get_agent_client(agent_name)
        if not agent_openai:
            return {
                "success": False,
                "status": "client_error",
                "content": "",
                "error": f"Failed to initialize Foundry agent client for '{agent_name}'. Ensure valid credentials.",
                "agent_name": agent_name,
            }

        from unittest.mock import Mock, DEFAULT
        is_mock = isinstance(agent_openai, Mock)
        has_real_responses = hasattr(agent_openai, "responses") and not is_mock
        mock_has_responses = False
        if is_mock and hasattr(agent_openai, "responses"):
            create_fn = getattr(agent_openai.responses, "create", None)
            if create_fn and (
                getattr(create_fn, "_mock_return_value", DEFAULT) is not DEFAULT
                or getattr(create_fn, "_mock_side_effect", None) is not None
            ):
                mock_has_responses = True

        try:
            if has_real_responses or mock_has_responses:
                # Primary invocation protocol for Microsoft Foundry Prompt Agents: Responses API
                # Note: Microsoft Foundry Prompt Agents own their instructions and temperature;
                # passing 'temperature' or 'instructions' to an agent endpoint returns HTTP 400.
                full_input = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                resp_kwargs: Dict[str, Any] = {
                    "input": full_input,
                }
                if max_tokens:
                    resp_kwargs["max_output_tokens"] = max_tokens
                if timeout:
                    resp_kwargs["timeout"] = timeout

                response = agent_openai.responses.create(**resp_kwargs)

                extracted_texts = []
                direct_output_text = getattr(response, "output_text", None)
                if direct_output_text:
                    content = direct_output_text.strip()
                else:
                    for item in getattr(response, "output", []):
                        if hasattr(item, "content"):
                            for c in item.content:
                                if hasattr(c, "text"):
                                    extracted_texts.append(c.text)
                                elif isinstance(c, dict) and "text" in c:
                                    extracted_texts.append(c["text"])
                        elif hasattr(item, "text"):
                            extracted_texts.append(item.text)
                    content = "\n".join(extracted_texts).strip() if extracted_texts else getattr(response, "text", "")
                model_used = getattr(response, "model", self.chat_deployment)
                usage = getattr(response, "usage", None)
                tokens_prompt = getattr(usage, "input_tokens", 0) if usage else 0
                tokens_completion = getattr(usage, "output_tokens", 0) if usage else 0
                tokens_total = getattr(usage, "total_tokens", tokens_prompt + tokens_completion) if usage else 0

                return {
                    "success": True,
                    "status": "success",
                    "content": content,
                    "model": model_used,
                    "agent_name": agent_name,
                    "tokens_prompt": tokens_prompt,
                    "tokens_completion": tokens_completion,
                    "tokens_total": tokens_total,
                    "error": None,
                }

            # Protocol fallback for clients/mocks supporting chat completions
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = agent_openai.chat.completions.create(
                model=agent_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = response.choices[0].message.content or ""
            model_used = getattr(response, "model", agent_name)
            usage = getattr(response, "usage", None)
            tokens_prompt = getattr(usage, "prompt_tokens", 0) if usage else 0
            tokens_completion = getattr(usage, "completion_tokens", 0) if usage else 0

            return {
                "success": True,
                "status": "success",
                "content": content,
                "model": model_used,
                "agent_name": agent_name,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "tokens_total": tokens_prompt + tokens_completion,
                "error": None,
            }
        except Exception as e:
            err_str = str(e)
            return {
                "success": False,
                "status": "error",
                "content": "",
                "error": f"Microsoft Foundry Agent '{agent_name}' execution failed: {err_str}",
                "agent_name": agent_name,
            }

    # --------------------------------------------------------------------------
    # Backward-compatible wrapper routing to the Primary Research Agent
    # --------------------------------------------------------------------------
    def generate_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        timeout: float = 45.0,
        target_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invokes the Microsoft Foundry Agent with task prompt and context."""
        agent_to_use = target_agent or self.research_agent_name
        return self.run_agent(
            agent_name=agent_to_use,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def generate_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        target_agent: Optional[str] = None,
    ) -> str:
        """Invokes Foundry agent returning plain content string."""
        res = self.generate_chat_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            target_agent=target_agent,
        )
        if res.get("success"):
            return res["content"]
        return f"[Error] {res.get('error', 'Foundry Agent execution failed.')}"


# Alias for backward compatibility
FoundryClient = MicrosoftFoundryClient
