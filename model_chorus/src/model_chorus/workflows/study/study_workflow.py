"""
STUDY workflow for persona-based collaborative research.

This module implements the StudyWorkflow which provides multi-persona investigation
with role-based orchestration, conversation memory, and systematic exploration.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from ...core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from ...core.conversation import ConversationMemory
from ...core.registry import WorkflowRegistry
from ...core.role_orchestration import (
    RoleOrchestrator,
    ModelRole,
    OrchestrationPattern,
    OrchestrationResult,
)
from ...providers import ModelProvider, GenerationRequest, GenerationResponse
from ...core.models import ConversationMessage
from ...core.progress import emit_workflow_start, emit_workflow_complete

logger = logging.getLogger(__name__)


@WorkflowRegistry.register("study")
class StudyWorkflow(BaseWorkflow):
    """
    Persona-based collaborative research workflow.

    This workflow implements systematic investigation through multiple personas
    with distinct expertise, enabling collaborative exploration of complex topics
    through role-based orchestration and conversation threading.

    Architecture:
    - Multi-persona role orchestration
    - Conversation threading and memory
    - Systematic hypothesis exploration
    - Persona-specific expertise and perspectives

    Key Features:
    - Role-based investigation with distinct personas
    - Conversation threading for multi-turn exploration
    - Systematic knowledge building
    - Collaborative analysis and synthesis
    - Inherits conversation support from BaseWorkflow

    The StudyWorkflow is ideal for:
    - Complex topic exploration requiring multiple perspectives
    - Research investigations with specialized knowledge domains
    - Collaborative problem analysis
    - Systematic learning and knowledge building
    - Multi-faceted topic investigation

    Workflow Pattern:
    1. **Persona Assignment**: Assign specialized roles for investigation
    2. **Collaborative Exploration**: Personas investigate from their perspectives
    3. **Synthesis**: Combine insights into comprehensive understanding

    Example:
        >>> from model_chorus.providers import ClaudeProvider
        >>> from model_chorus.workflows.study import StudyWorkflow
        >>> from model_chorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = StudyWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # Conduct research
        >>> result = await workflow.run(
        ...     "Explore the implementation patterns of authentication systems"
        ... )
        >>> print(result.synthesis)
        >>>
        >>> # Continue investigation
        >>> result2 = await workflow.run(
        ...     "Dive deeper into OAuth 2.0 flow patterns",
        ...     continuation_id=result.metadata.get('thread_id')
        ... )
    """

    def __init__(
        self,
        provider: ModelProvider,
        fallback_providers: Optional[List[ModelProvider]] = None,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize StudyWorkflow with a primary provider.

        Args:
            provider: ModelProvider instance to use for the workflow
            fallback_providers: Optional list of fallback providers
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn conversations

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        super().__init__(
            name="Study",
            description="Persona-based collaborative research workflow",
            config=config,
            conversation_memory=conversation_memory
        )
        self.provider = provider
        self.fallback_providers = fallback_providers or []

        logger.info(f"StudyWorkflow initialized with provider: {provider.provider_name}")

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        skip_provider_check: bool = False,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute persona-based research workflow.

        This method orchestrates multi-persona investigation with role-based
        collaboration and systematic exploration. Supports conversation continuation
        for iterative research sessions.

        Args:
            prompt: The research topic or question to investigate
            continuation_id: Optional thread ID to continue existing research
            files: Optional list of file paths to include in context
            skip_provider_check: Skip provider availability check
            **kwargs: Additional parameters (e.g., personas, max_turns, temperature)

        Returns:
            WorkflowResult containing:
                - synthesis: Comprehensive research findings
                - steps: Individual persona contributions
                - metadata: Thread ID, personas used, investigation parameters

        Raises:
            ValueError: If prompt is empty
            Exception: If provider generation fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Check provider availability
        if not skip_provider_check:
            has_available, available, unavailable = await self.check_provider_availability(
                self.provider, self.fallback_providers
            )

            if not has_available:
                from ...providers.cli_provider import ProviderUnavailableError
                error_msg = "No providers available for study workflow:\n"
                for name, error in unavailable:
                    error_msg += f"  - {name}: {error}\n"
                raise ProviderUnavailableError(
                    "all",
                    error_msg,
                    [
                        "Check installations: model-chorus list-providers --check",
                        "Install missing providers or update .model-chorusrc"
                    ]
                )

            if unavailable and logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Some providers unavailable: {[n for n, _ in unavailable]}")
                logger.info(f"Will use available providers: {available}")

        # Get or create thread ID
        thread_id = continuation_id or str(uuid.uuid4())

        # Retrieve conversation history if continuing
        history = []
        if continuation_id and self.conversation_memory:
            thread = self.conversation_memory.get_thread(continuation_id)
            if thread:
                history = thread.messages
                logger.info(f"Loaded {len(history)} messages from thread {continuation_id}")

        # Emit workflow start event
        emit_workflow_start(
            workflow_type="study",
            prompt=prompt,
            thread_id=thread_id
        )

        # Create workflow result
        result = WorkflowResult(success=False)

        try:
            # PHASE 1: Setup personas (to be implemented)
            personas = self._setup_personas(kwargs.get('personas'))
            logger.info(f"Study workflow using {len(personas)} personas")

            # PHASE 2: Investigation loop (to be implemented)
            investigation_steps = await self._conduct_investigation(
                prompt=prompt,
                personas=personas,
                history=history,
                thread_id=thread_id,
                **kwargs
            )

            # PHASE 3: Synthesis (to be implemented)
            synthesis = await self._synthesize_findings(investigation_steps, **kwargs)

            # Build result
            result.success = True
            result.steps = investigation_steps
            result.synthesis = synthesis
            result.metadata = {
                "thread_id": thread_id,
                "workflow_type": "study",
                "personas_used": [p.get('name', 'unknown') for p in personas],
                "investigation_rounds": len(investigation_steps),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"Study workflow completed successfully. Thread: {thread_id}")

        except Exception as e:
            logger.error(f"StudyWorkflow execution failed: {e}")
            result.success = False
            result.error = str(e)

        finally:
            # Emit workflow complete event
            emit_workflow_complete(
                workflow_type="study",
                success=result.success,
                thread_id=result.metadata.get("thread_id") if result.metadata else None
            )

        self._result = result
        return result

    def _setup_personas(self, personas: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Setup investigation personas.

        Args:
            personas: Optional list of persona configurations

        Returns:
            List of persona configurations with defaults applied
        """
        # Default personas if none provided
        if not personas:
            return [
                {
                    "name": "Researcher",
                    "expertise": "systematic investigation and analysis",
                    "role": "primary investigator"
                },
                {
                    "name": "Critic",
                    "expertise": "identifying assumptions and edge cases",
                    "role": "critical reviewer"
                }
            ]
        return personas

    async def _conduct_investigation(
        self,
        prompt: str,
        personas: List[Dict[str, Any]],
        history: List[ConversationMessage],
        thread_id: str,
        **kwargs
    ) -> List[WorkflowStep]:
        """
        Conduct persona-based investigation.

        This is the main investigation loop where personas explore the topic.

        Args:
            prompt: Research topic or question
            personas: List of persona configurations
            history: Conversation history
            thread_id: Thread ID for conversation tracking
            **kwargs: Additional parameters

        Returns:
            List of investigation steps from persona contributions
        """
        steps = []

        # TODO: Implement actual persona invocation and investigation
        # For now, create placeholder step
        step = WorkflowStep(
            step_number=1,
            content="Investigation flow skeleton in place. Persona invocation to be implemented.",
            model=self.provider.provider_name,
            metadata={
                "phase": "investigation",
                "personas_ready": len(personas)
            }
        )
        steps.append(step)

        return steps

    async def _synthesize_findings(
        self,
        steps: List[WorkflowStep],
        **kwargs
    ) -> str:
        """
        Synthesize findings from investigation steps.

        Args:
            steps: Investigation steps from persona contributions
            **kwargs: Additional parameters

        Returns:
            Synthesized findings and conclusions
        """
        # TODO: Implement actual synthesis logic
        # For now, return basic synthesis
        if not steps:
            return "No investigation findings to synthesize."

        return (
            f"Study workflow investigation skeleton complete.\n"
            f"Completed {len(steps)} investigation step(s).\n"
            f"Persona invocation and synthesis logic ready for implementation."
        )
