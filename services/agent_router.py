"""
Microsoft Foundry Agent Router for ResearchLens AI.
Provides clean application-level task-to-agent routing for cloud-managed Foundry agents.
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
from config.settings import settings


class AgentTaskType(str, Enum):
    """Categorical research task types routed to specialized Foundry agents."""
    DEEP_ANALYSIS = "deep_analysis"
    PAPER_CHAT = "paper_chat"
    COMPARISON = "comparison"
    RESEARCH_GAPS = "research_gaps"
    RESEARCH_QUESTIONS = "research_questions"
    LITERATURE_REVIEW = "literature_review"


@dataclass
class AgentRoute:
    """Target Microsoft Foundry agent destination for an application task."""
    task_type: AgentTaskType
    agent_name: str
    agent_version: Optional[str]
    description: str

    @property
    def version(self) -> Optional[str]:
        return self.agent_version


class AgentRouter:
    """
    Centralized agent router matching user tasks to persisted Foundry agents.
    Routes are dynamically configured via environment variables / settings.
    """

    def __init__(
        self,
        research_agent_name: Optional[str] = None,
        research_agent_version: Optional[str] = None,
        chat_agent_name: Optional[str] = None,
        chat_agent_version: Optional[str] = None,
    ):
        self.research_agent_name = (research_agent_name or settings.FOUNDRY_RESEARCH_AGENT_NAME).strip()
        self.research_agent_version = (research_agent_version or settings.FOUNDRY_RESEARCH_AGENT_VERSION).strip()
        self.chat_agent_name = (chat_agent_name or settings.FOUNDRY_CHAT_AGENT_NAME).strip()
        self.chat_agent_version = (chat_agent_version or settings.FOUNDRY_CHAT_AGENT_VERSION).strip()

    def resolve(self, task_type: AgentTaskType) -> AgentRoute:
        """Resolves task type to the appropriate Microsoft Foundry agent route."""
        if task_type == AgentTaskType.PAPER_CHAT:
            return AgentRoute(
                task_type=task_type,
                agent_name=self.chat_agent_name,
                agent_version=self.chat_agent_version,
                description="Interactive conversational QA and paper explanation agent.",
            )
        else:
            return AgentRoute(
                task_type=task_type,
                agent_name=self.research_agent_name,
                agent_version=self.research_agent_version,
                description="Primary research analysis, synthesis, gap detection, and literature review agent.",
            )

    def get_route_map(self) -> Dict[str, AgentRoute]:
        """Returns the full task routing configuration."""
        return {t.value: self.resolve(t) for t in AgentTaskType}
