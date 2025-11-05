"""
ThinkDeep workflow for extended reasoning and systematic investigation.

This module implements the ThinkDeepWorkflow which provides multi-step
investigation with hypothesis tracking, evidence collection, and confidence
progression across conversation turns.
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

from ..core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from ..core.conversation import ConversationMemory
from ..providers import ModelProvider, GenerationRequest, GenerationResponse
from ..core.models import (
    ConversationMessage,
    ThinkDeepState,
    Hypothesis,
    InvestigationStep,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class ThinkDeepWorkflow(BaseWorkflow):
    """
    Extended reasoning workflow with systematic investigation and hypothesis tracking.

    This workflow provides multi-step investigation capabilities where hypotheses
    are formed, tested, and refined across conversation turns. It maintains state
    including hypothesis evolution, investigation steps, confidence levels, and
    relevant files examined.

    Key features:
    - Single provider with extended reasoning
    - Hypothesis tracking and evolution
    - Investigation step progression
    - Confidence level tracking
    - File examination history
    - State persistence across turns via conversation threading

    The ThinkDeepWorkflow is ideal for:
    - Complex problem analysis requiring systematic investigation
    - Debugging scenarios with hypothesis testing
    - Architecture decisions with evidence-based reasoning
    - Security analysis with confidence tracking
    - Any task requiring methodical, step-by-step investigation

    Example:
        >>> from modelchorus.providers import ClaudeProvider
        >>> from modelchorus.workflows import ThinkDeepWorkflow
        >>> from modelchorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = ThinkDeepWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # First step (creates new investigation)
        >>> result1 = await workflow.run(
        ...     "Why is authentication failing intermittently?",
        ...     files=["src/auth.py", "tests/test_auth.py"]
        ... )
        >>> thread_id = result1.metadata.get('thread_id')
        >>>
        >>> # Follow-up investigation (continues thread with state)
        >>> result2 = await workflow.run(
        ...     "Check if it's related to async/await patterns",
        ...     continuation_id=thread_id,
        ...     files=["src/services/user.py"]
        ... )
        >>>
        >>> # Check investigation state
        >>> state = workflow.get_investigation_state(thread_id)
        >>> print(f"Hypotheses: {len(state.hypotheses)}")
        >>> print(f"Confidence: {state.current_confidence}")
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize ThinkDeepWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for investigation
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn investigations

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        super().__init__(
            name="ThinkDeep",
            description="Extended reasoning with systematic investigation and hypothesis tracking",
            config=config,
            conversation_memory=conversation_memory
        )
        self.provider = provider

        logger.info(f"ThinkDeepWorkflow initialized with provider: {provider.provider_name}")

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute ThinkDeep investigation workflow.

        This method orchestrates a systematic investigation, maintaining state
        across turns when continuation_id is provided. Each turn can add new
        hypotheses, record investigation steps, update confidence levels, and
        track examined files.

        Args:
            prompt: The investigation prompt/query
            continuation_id: Optional thread ID to continue an existing investigation
            files: Optional list of file paths to examine during investigation
            **kwargs: Additional parameters passed to provider.generate()
                     (e.g., temperature, max_tokens, system_prompt)

        Returns:
            WorkflowResult containing:
                - success: True if investigation step succeeded
                - synthesis: The model's investigation findings
                - steps: Investigation steps with findings and confidence
                - metadata: thread_id, investigation state, and progress info

        Raises:
            Exception: If provider.generate() fails

        Example:
            >>> # Start new investigation
            >>> result = await workflow.run(
            ...     "Investigate authentication bug",
            ...     files=["src/auth.py"]
            ... )
            >>>
            >>> # Continue investigation
            >>> result2 = await workflow.run(
            ...     "Test hypothesis about async race condition",
            ...     continuation_id=result.metadata['thread_id'],
            ...     files=["src/services/user.py"]
            ... )
        """
        logger.info(
            f"Starting ThinkDeep investigation - prompt length: {len(prompt)}, "
            f"continuation: {continuation_id is not None}, "
            f"files: {len(files) if files else 0}"
        )

        # Generate or use thread ID
        if continuation_id:
            thread_id = continuation_id
        else:
            # Create new thread if conversation memory available
            if self.conversation_memory:
                thread_id = self.conversation_memory.create_thread(
                    workflow_name=self.name
                )
            else:
                thread_id = str(uuid.uuid4())

        # Initialize result
        result = WorkflowResult(success=False)

        try:
            # Load or initialize investigation state
            state = self._get_or_create_state(thread_id)

            # Build the full prompt with investigation context
            full_prompt = self._build_investigation_prompt(
                prompt, thread_id, state, files
            )

            # Create generation request
            request = GenerationRequest(
                prompt=full_prompt,
                continuation_id=thread_id,
                **kwargs
            )

            logger.info(f"Sending investigation request to provider: {self.provider.provider_name}")

            # Generate response from provider
            response: GenerationResponse = await self.provider.generate(request)

            logger.info(
                f"Received investigation response from {self.provider.provider_name}: "
                f"{len(response.content)} chars"
            )

            # Add user message to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "user",
                    prompt,
                    files=files,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name
                )

            # Add assistant response to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "assistant",
                    response.content,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name,
                    model_name=response.model
                )

            # Create and add investigation step
            step_number = len(state.steps) + 1
            investigation_step = InvestigationStep(
                step_number=step_number,
                findings=self._extract_findings(response.content),
                files_checked=files if files else [],
                confidence=state.current_confidence  # Will be updated with confidence extraction later
            )
            state.steps.append(investigation_step)

            # Track files examined
            if files:
                for file in files:
                    if file not in state.relevant_files:
                        state.relevant_files.append(file)

            # Save updated state
            self._save_state(thread_id, state)

            # Build successful result
            result.success = True
            result.synthesis = response.content
            result.add_step(
                step_number=len(state.steps) + 1,
                content=response.content,
                model=response.model
            )

            # Add metadata
            result.metadata.update({
                'thread_id': thread_id,
                'provider': self.provider.provider_name,
                'model': response.model,
                'usage': response.usage,
                'stop_reason': response.stop_reason,
                'is_continuation': continuation_id is not None,
                'investigation_step': len(state.steps) + 1,
                'hypotheses_count': len(state.hypotheses),
                'confidence': state.current_confidence,
                'files_examined': len(state.relevant_files)
            })

            logger.info(f"ThinkDeep investigation step completed for thread: {thread_id}")

        except Exception as e:
            logger.error(f"ThinkDeep investigation failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.metadata['thread_id'] = thread_id

        # Store result
        self._result = result
        return result

    def _get_or_create_state(self, thread_id: str) -> ThinkDeepState:
        """
        Get existing investigation state or create new one.

        Args:
            thread_id: Thread ID to load state from

        Returns:
            ThinkDeepState for this investigation
        """
        if not self.conversation_memory:
            return ThinkDeepState()

        thread = self.get_thread(thread_id)
        if not thread or not thread.state:
            return ThinkDeepState()

        # Try to parse state from thread.state dict
        try:
            state_data = thread.state.get('thinkdeep', {})
            return ThinkDeepState(**state_data)
        except Exception as e:
            logger.warning(f"Failed to load state from thread {thread_id}: {e}")
            return ThinkDeepState()

    def _save_state(self, thread_id: str, state: ThinkDeepState) -> None:
        """
        Save investigation state to thread.

        Args:
            thread_id: Thread ID to save state to
            state: ThinkDeepState to save
        """
        if not self.conversation_memory:
            return

        thread = self.get_thread(thread_id)
        if not thread:
            logger.warning(f"Thread {thread_id} not found, cannot save state")
            return

        # Save state to thread.state dict
        thread.state['thinkdeep'] = state.model_dump()
        self.conversation_memory.save_thread(thread)

    def _build_investigation_prompt(
        self,
        prompt: str,
        thread_id: str,
        state: ThinkDeepState,
        files: Optional[List[str]] = None
    ) -> str:
        """
        Build investigation prompt with conversation history, state, and file context.

        Args:
            prompt: Current investigation prompt
            thread_id: Thread ID to load history from
            state: Current investigation state
            files: Optional list of file paths to include in context

        Returns:
            Full prompt string with investigation context
        """
        context_parts = []

        # Add investigation state summary if exists
        if state.hypotheses or state.steps:
            context_parts.append("Current investigation state:\n")

            if state.hypotheses:
                context_parts.append(f"\nHypotheses ({len(state.hypotheses)}):\n")
                for i, hyp in enumerate(state.hypotheses, 1):
                    context_parts.append(
                        f"{i}. [{hyp.status.upper()}] {hyp.hypothesis}\n"
                    )
                    if hyp.evidence:
                        context_parts.append(f"   Evidence: {', '.join(hyp.evidence[:3])}\n")

            if state.steps:
                context_parts.append(f"\nInvestigation steps completed: {len(state.steps)}\n")
                context_parts.append(f"Current confidence: {state.current_confidence}\n")

            if state.relevant_files:
                context_parts.append(
                    f"\nFiles examined: {', '.join(state.relevant_files[:5])}"
                )
                if len(state.relevant_files) > 5:
                    context_parts.append(f" (and {len(state.relevant_files) - 5} more)")
                context_parts.append("\n")

        # Add file contents if provided
        if files:
            context_parts.append("\nFile context for this step:\n")
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    context_parts.append(f"\n--- File: {file_path} ---\n")
                    context_parts.append(file_content)
                    context_parts.append(f"\n--- End of {file_path} ---\n")
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    context_parts.append(f"\n--- File: {file_path} (Failed to read: {e}) ---\n")

        # Add conversation history if available
        if self.conversation_memory:
            messages = self.resume_conversation(thread_id)
            if messages:
                context_parts.append("\nPrevious investigation turns:\n")
                for msg in messages:
                    role_label = msg.role.upper()
                    # Truncate long messages in history
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    context_parts.append(f"{role_label}: {content}\n")

        # Add current prompt
        if context_parts:
            context_parts.append(f"\nCurrent investigation step:\n{prompt}")
            full_prompt = "\n".join(context_parts)
        else:
            full_prompt = prompt

        logger.debug(
            f"Built investigation prompt - "
            f"hypotheses: {len(state.hypotheses)}, "
            f"steps: {len(state.steps)}, "
            f"files: {len(files) if files else 0}, "
            f"total length: {len(full_prompt)}"
        )

        return full_prompt

    def _extract_findings(self, response_content: str) -> str:
        """
        Extract key findings from investigation response.

        This is a simple extraction that takes the first paragraph or
        first 500 characters as the key findings summary. Future enhancement
        could use structured output parsing or LLM-based extraction.

        Args:
            response_content: Full response content from the model

        Returns:
            Extracted findings summary
        """
        # Simple extraction: take first paragraph or first 500 chars
        lines = response_content.strip().split('\n\n')
        if lines:
            first_paragraph = lines[0].strip()
            if len(first_paragraph) > 500:
                return first_paragraph[:497] + "..."
            return first_paragraph

        # Fallback: truncate to 500 chars
        if len(response_content) > 500:
            return response_content[:497] + "..."
        return response_content.strip()

    def get_investigation_state(self, thread_id: str) -> Optional[ThinkDeepState]:
        """
        Get the current investigation state for a thread.

        Args:
            thread_id: Thread ID to get state for

        Returns:
            ThinkDeepState if available, None otherwise
        """
        if not self.conversation_memory:
            return None

        return self._get_or_create_state(thread_id)

    def add_hypothesis(
        self,
        thread_id: str,
        hypothesis_text: str,
        evidence: Optional[List[str]] = None
    ) -> bool:
        """
        Add a new hypothesis to an investigation.

        Args:
            thread_id: Thread ID of the investigation
            hypothesis_text: The hypothesis statement
            evidence: Optional initial evidence supporting the hypothesis

        Returns:
            True if hypothesis was added, False if investigation not found
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return False

        hypothesis = Hypothesis(
            hypothesis=hypothesis_text,
            evidence=evidence if evidence else [],
            status="active"
        )
        state.hypotheses.append(hypothesis)
        self._save_state(thread_id, state)

        logger.info(f"Added hypothesis to investigation {thread_id}: {hypothesis_text}")
        return True

    def update_hypothesis(
        self,
        thread_id: str,
        hypothesis_text: str,
        new_evidence: Optional[List[str]] = None,
        new_status: Optional[str] = None
    ) -> bool:
        """
        Update an existing hypothesis with new evidence or status.

        Args:
            thread_id: Thread ID of the investigation
            hypothesis_text: The hypothesis to update (matched by text)
            new_evidence: Optional new evidence to add
            new_status: Optional new status ("active", "disproven", "validated")

        Returns:
            True if hypothesis was updated, False if not found
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return False

        # Find hypothesis by text
        hypothesis_found = False
        for hyp in state.hypotheses:
            if hyp.hypothesis == hypothesis_text:
                hypothesis_found = True
                if new_evidence:
                    hyp.evidence.extend(new_evidence)
                if new_status and new_status in ["active", "disproven", "validated"]:
                    hyp.status = new_status
                break

        if hypothesis_found:
            self._save_state(thread_id, state)
            logger.info(f"Updated hypothesis in investigation {thread_id}: {hypothesis_text}")

        return hypothesis_found

    def validate_hypothesis(self, thread_id: str, hypothesis_text: str) -> bool:
        """
        Mark a hypothesis as validated.

        Args:
            thread_id: Thread ID of the investigation
            hypothesis_text: The hypothesis to validate (matched by text)

        Returns:
            True if hypothesis was validated, False if not found
        """
        return self.update_hypothesis(thread_id, hypothesis_text, new_status="validated")

    def disprove_hypothesis(self, thread_id: str, hypothesis_text: str) -> bool:
        """
        Mark a hypothesis as disproven.

        Args:
            thread_id: Thread ID of the investigation
            hypothesis_text: The hypothesis to disprove (matched by text)

        Returns:
            True if hypothesis was disproven, False if not found
        """
        return self.update_hypothesis(thread_id, hypothesis_text, new_status="disproven")

    def get_active_hypotheses(self, thread_id: str) -> List[Hypothesis]:
        """
        Get all active (not disproven or validated) hypotheses.

        Args:
            thread_id: Thread ID of the investigation

        Returns:
            List of active hypotheses
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return []

        return [h for h in state.hypotheses if h.status == "active"]

    def get_all_hypotheses(self, thread_id: str) -> List[Hypothesis]:
        """
        Get all hypotheses regardless of status.

        Args:
            thread_id: Thread ID of the investigation

        Returns:
            List of all hypotheses
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return []

        return state.hypotheses

    def update_confidence(
        self,
        thread_id: str,
        new_confidence: str
    ) -> bool:
        """
        Update the current confidence level for an investigation.

        Args:
            thread_id: Thread ID of the investigation
            new_confidence: New confidence level (ConfidenceLevel value)

        Returns:
            True if confidence was updated, False if investigation not found
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return False

        # Validate confidence level
        valid_levels = [level.value for level in ConfidenceLevel]
        if new_confidence not in valid_levels:
            logger.warning(
                f"Invalid confidence level '{new_confidence}'. "
                f"Valid levels: {valid_levels}"
            )
            return False

        old_confidence = state.current_confidence
        state.current_confidence = new_confidence
        self._save_state(thread_id, state)

        logger.info(
            f"Updated confidence for investigation {thread_id}: "
            f"{old_confidence} → {new_confidence}"
        )
        return True

    def get_confidence(self, thread_id: str) -> Optional[str]:
        """
        Get the current confidence level for an investigation.

        Args:
            thread_id: Thread ID of the investigation

        Returns:
            Current confidence level or None if investigation not found
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return None

        return state.current_confidence

    def is_investigation_complete(self, thread_id: str) -> bool:
        """
        Determine if investigation has reached completion criteria.

        An investigation is considered complete when:
        - Confidence level is "certain" OR "almost_certain"
        - At least one hypothesis exists
        - At least one investigation step has been completed

        Args:
            thread_id: Thread ID of the investigation

        Returns:
            True if investigation meets completion criteria, False otherwise
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return False

        # Check confidence level
        high_confidence = state.current_confidence in [
            ConfidenceLevel.CERTAIN.value,
            ConfidenceLevel.ALMOST_CERTAIN.value
        ]

        # Check if any work has been done
        has_hypotheses = len(state.hypotheses) > 0
        has_steps = len(state.steps) > 0

        is_complete = high_confidence and has_hypotheses and has_steps

        if is_complete:
            logger.info(
                f"Investigation {thread_id} meets completion criteria: "
                f"confidence={state.current_confidence}, "
                f"hypotheses={len(state.hypotheses)}, "
                f"steps={len(state.steps)}"
            )

        return is_complete

    def get_investigation_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the investigation progress.

        Args:
            thread_id: Thread ID of the investigation

        Returns:
            Dictionary with investigation summary or None if not found
        """
        state = self._get_or_create_state(thread_id)
        if not state:
            return None

        active_hypotheses = [h for h in state.hypotheses if h.status == "active"]
        validated_hypotheses = [h for h in state.hypotheses if h.status == "validated"]
        disproven_hypotheses = [h for h in state.hypotheses if h.status == "disproven"]

        return {
            "thread_id": thread_id,
            "confidence": state.current_confidence,
            "is_complete": self.is_investigation_complete(thread_id),
            "total_steps": len(state.steps),
            "total_hypotheses": len(state.hypotheses),
            "active_hypotheses": len(active_hypotheses),
            "validated_hypotheses": len(validated_hypotheses),
            "disproven_hypotheses": len(disproven_hypotheses),
            "files_examined": len(state.relevant_files)
        }

    def get_provider(self) -> ModelProvider:
        """
        Get the configured provider.

        Returns:
            The ModelProvider instance used by this workflow
        """
        return self.provider

    def validate_config(self) -> bool:
        """
        Validate the workflow configuration.

        Checks that the provider is properly configured and has a valid API key.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.provider is None:
            logger.error("Provider is None")
            return False

        if not self.provider.validate_api_key():
            logger.warning(f"Provider {self.provider.provider_name} API key validation failed")
            return False

        return True

    def __repr__(self) -> str:
        """String representation of the workflow."""
        memory_status = "with memory" if self.conversation_memory else "no memory"
        return (
            f"ThinkDeepWorkflow(provider='{self.provider.provider_name}', "
            f"{memory_status})"
        )
