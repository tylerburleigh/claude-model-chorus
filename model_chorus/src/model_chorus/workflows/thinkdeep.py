"""
ThinkDeep workflow for extended reasoning and systematic investigation.

This module implements the ThinkDeepWorkflow which provides multi-step
investigation with hypothesis tracking, evidence collection, and confidence
progression across conversation turns.
"""

import logging
import uuid
from typing import Any, Literal, cast

from ..core.base_workflow import BaseWorkflow, WorkflowResult
from ..core.conversation import ConversationMemory
from ..core.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)
from ..core.progress import emit_progress, emit_workflow_complete, emit_workflow_start
from ..providers import GenerationRequest, GenerationResponse, ModelProvider

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
        >>> from model_chorus.providers import ClaudeProvider
        >>> from model_chorus.workflows import ThinkDeepWorkflow
        >>> from model_chorus.core.conversation import ConversationMemory
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
        fallback_providers: list[ModelProvider] | None = None,
        config: dict[str, Any] | None = None,
        conversation_memory: ConversationMemory | None = None,
        expert_provider: ModelProvider | None = None,
    ):
        """
        Initialize ThinkDeepWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for investigation
            fallback_providers: Optional list of fallback providers to try if primary fails
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn investigations
            expert_provider: Optional secondary ModelProvider for expert validation

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        super().__init__(
            name="ThinkDeep",
            description="Extended reasoning with systematic investigation and hypothesis tracking",
            config=config,
            conversation_memory=conversation_memory,
        )
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.expert_provider = expert_provider

        # Get expert validation config (default: enabled if expert_provider provided)
        if config:
            self.enable_expert_validation = config.get(
                "enable_expert_validation", expert_provider is not None
            )
        else:
            self.enable_expert_validation = expert_provider is not None

        logger.info(
            f"ThinkDeepWorkflow initialized with provider: {provider.provider_name}, "
            f"expert_provider: {expert_provider.provider_name if expert_provider else 'None'}, "
            f"expert_validation: {self.enable_expert_validation}"
        )

    async def run(  # type: ignore[override,no-untyped-def]
        self,
        step: str,
        step_number: int,
        total_steps: int,
        next_step_required: bool,
        findings: str,
        hypothesis: str | None = None,
        confidence: str = "exploring",
        continuation_id: str | None = None,
        files: list[str] | None = None,
        relevant_files: list[str] | None = None,
        thinking_mode: str = "medium",
        skip_provider_check: bool = False,
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute ThinkDeep investigation workflow with explicit multi-step control.

        This method orchestrates a systematic investigation with explicit step management.
        Each invocation represents one investigation step with specific findings, hypothesis,
        and confidence level. State is maintained across steps via continuation_id.

        Args:
            step: Investigation step description (what to investigate in this step)
            step_number: Current step index (starts at 1)
            total_steps: Estimated total number of steps in investigation
            next_step_required: Whether more investigation steps are needed
            findings: What was discovered in this step
            hypothesis: Optional current working theory about the problem
            confidence: Confidence level (exploring, low, medium, high, very_high, almost_certain, certain)
            continuation_id: Optional thread ID to continue an existing investigation
            files: Optional list of file paths examined in this step
            thinking_mode: Reasoning depth (minimal, low, medium, high, max)
            skip_provider_check: Skip provider availability check (faster startup)
            **kwargs: Additional parameters passed to provider.generate()
                     (e.g., system_prompt)

        Returns:
            WorkflowResult containing:
                - success: True if investigation step succeeded
                - synthesis: The model's investigation analysis
                - steps: Investigation steps with findings and confidence
                - metadata: continuation_id, step info, and progress

        Raises:
            Exception: If provider.generate() fails

        Example:
            >>> # Start new investigation (step 1)
            >>> result = await workflow.run(
            ...     step="Investigate authentication failures",
            ...     step_number=1,
            ...     total_steps=3,
            ...     next_step_required=True,
            ...     findings="5% of requests failing with 401 errors",
            ...     confidence="exploring",
            ...     files=["src/auth.py"]
            ... )
            >>>
            >>> # Continue investigation (step 2)
            >>> result2 = await workflow.run(
            ...     step="Test hypothesis about async race condition",
            ...     step_number=2,
            ...     total_steps=3,
            ...     next_step_required=True,
            ...     findings="Found missing await in token validation",
            ...     hypothesis="Race condition in async token validation",
            ...     confidence="medium",
            ...     continuation_id=result.metadata['continuation_id'],
            ...     files=["src/services/token.py"]
            ... )
        """
        logger.info(
            f"Starting ThinkDeep investigation step {step_number}/{total_steps} - "
            f"continuation: {continuation_id is not None}, "
            f"confidence: {confidence}, "
            f"files: {len(files) if files else 0}, "
            f"relevant_files: {len(relevant_files) if relevant_files else 0}"
        )

        # Check provider availability
        if not skip_provider_check:
            has_available, available, unavailable = (
                await self.check_provider_availability(
                    self.provider, self.fallback_providers
                )
            )

            if not has_available:
                from ..providers.cli_provider import ProviderUnavailableError

                error_msg = "No providers available for ThinkDeep investigation:\n"
                for name, error in unavailable:
                    error_msg += f"  - {name}: {error}\n"
                raise ProviderUnavailableError(
                    "all",
                    error_msg,
                    [
                        "Check installations: model-chorus list-providers --check",
                        "Install missing providers or update .model-chorusrc",
                    ],
                )

            if unavailable and logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Some providers unavailable: {[n for n, _ in unavailable]}"
                )
                logger.info(f"Will use available providers: {available}")

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

            # Build the full prompt with multi-step investigation context
            full_prompt = self._build_investigation_prompt(
                step=step,
                step_number=step_number,
                total_steps=total_steps,
                findings=findings,
                hypothesis=hypothesis,
                confidence=confidence,
                thread_id=thread_id,
                state=state,
                files=files,
                relevant_files=relevant_files,
                thinking_mode=thinking_mode,
            )

            # Create generation request
            request = GenerationRequest(
                prompt=full_prompt, continuation_id=thread_id, **kwargs
            )

            logger.info(
                f"Sending investigation request to provider: {self.provider.provider_name}"
            )

            # Emit workflow start (for step 1) or progress update
            if step_number == 1:
                emit_workflow_start("thinkdeep")

            # Generate response from provider with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed before success: {', '.join(failed)}")
                emit_progress(f"Failed providers: {', '.join(failed)}")
            emit_progress(f"Using provider: {used_provider}")

            logger.info(
                f"Received investigation response from {used_provider}: "
                f"{len(response.content)} chars"
            )

            # Add user message to conversation history (step description + findings)
            if self.conversation_memory:
                user_message = (
                    f"Step {step_number}/{total_steps}: {step}\nFindings: {findings}"
                )
                if hypothesis:
                    user_message += f"\nHypothesis: {hypothesis}"
                message_files = self._merge_file_lists(files, relevant_files)
                self.add_message(
                    thread_id,
                    "user",
                    user_message,
                    files=message_files or None,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name,
                )

            # Add assistant response to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "assistant",
                    response.content,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name,
                    model_name=response.model,
                )

            # Create and add investigation step with explicit parameters
            investigation_step = InvestigationStep(
                step_number=step_number,
                findings=findings,
                files_checked=files if files else [],
                confidence=confidence,
            )
            state.steps.append(investigation_step)

            # Update current confidence
            state.current_confidence = confidence

            # Add/update hypothesis if provided
            if hypothesis:
                # Check if this hypothesis already exists
                existing_hyp = next(
                    (h for h in state.hypotheses if h.hypothesis == hypothesis), None
                )
                if existing_hyp:
                    # Update existing hypothesis status based on confidence
                    if confidence in ["very_high", "almost_certain", "certain"]:
                        existing_hyp.status = "validated"
                    else:
                        existing_hyp.status = "active"
                else:
                    # Add new hypothesis
                    new_hyp = Hypothesis(
                        hypothesis=hypothesis,
                        status="active",
                        evidence=[findings],  # Add current findings as evidence
                    )
                    state.hypotheses.append(new_hyp)

            # Track files examined
            combined_files = self._merge_file_lists(files, relevant_files)
            if combined_files:
                for file in combined_files:
                    if file not in state.relevant_files:
                        state.relevant_files.append(file)

            # Perform expert validation if configured and confidence not "certain"
            expert_validation = await self._perform_expert_validation(
                thread_id, state, response.content, **kwargs
            )

            # Save updated state (after expert validation)
            self._save_state(thread_id, state)

            # Build successful result
            result.success = True
            result.synthesis = response.content
            result.add_step(
                step_number=len(state.steps) + 1,
                content=response.content,
                model=response.model,
            )

            # Add expert validation to result if performed
            if expert_validation and self.expert_provider is not None:
                result.add_step(
                    step_number=len(state.steps) + 2,
                    content=expert_validation,
                    model=f"{self.expert_provider.provider_name} (expert validation)",
                )

            # Add metadata with continuation_id for next step
            result.metadata.update(
                {
                    "thread_id": thread_id,  # Keep for backward compatibility
                    "continuation_id": thread_id,  # Primary ID for multi-step continuation
                    "provider": self.provider.provider_name,
                    "model": response.model,
                    "usage": response.usage,
                    "stop_reason": response.stop_reason,
                    "step_number": step_number,
                    "total_steps": total_steps,
                    "next_step_required": next_step_required,
                    "confidence": confidence,
                    "hypothesis": hypothesis,
                    "findings": findings,
                    "files_examined_this_step": len(files) if files else 0,
                    "total_files_examined": len(state.relevant_files),
                    "total_hypotheses": len(state.hypotheses),
                    "expert_validation_performed": expert_validation is not None,
                    "relevant_files_this_step": self._merge_file_lists(
                        None, relevant_files
                    ),
                    "relevant_files": list(state.relevant_files),
                }
            )

            logger.info(
                f"ThinkDeep investigation step {step_number}/{total_steps} completed for thread: {thread_id}"
            )

            # Emit workflow complete (for final step)
            if not next_step_required or step_number == total_steps:
                emit_workflow_complete("thinkdeep")

        except Exception as e:
            logger.error(f"ThinkDeep investigation failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.metadata["thread_id"] = thread_id

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
            state_data = thread.state.get("thinkdeep", {})
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
        thread.state["thinkdeep"] = state.model_dump()
        self.conversation_memory._save_thread(thread)

    @staticmethod
    def _merge_file_lists(
        files: list[str] | None, relevant_files: list[str] | None
    ) -> list[str]:
        """
        Merge and deduplicate file paths from the examined files list and the
        relevant files list while preserving order.

        Args:
            files: Files directly examined in this step
            relevant_files: Additional files identified as relevant

        Returns:
            Combined list of unique file paths
        """
        merged: list[str] = []
        for source_list in (files or []), (relevant_files or []):
            for file_path in source_list:
                if file_path and file_path not in merged:
                    merged.append(file_path)
        return merged

    def _build_investigation_prompt(
        self,
        step: str,
        step_number: int,
        total_steps: int,
        findings: str,
        hypothesis: str | None,
        confidence: str,
        thread_id: str,
        state: ThinkDeepState,
        files: list[str] | None = None,
        relevant_files: list[str] | None = None,
        thinking_mode: str = "medium",
    ) -> str:
        """
        Build investigation prompt with multi-step context.

        Args:
            step: Investigation step description
            step_number: Current step index
            total_steps: Estimated total investigation steps
            findings: What was discovered in this step
            hypothesis: Optional current working theory
            confidence: Confidence level
            thread_id: Thread ID to load history from
            state: Current investigation state
            files: Optional list of file paths to include in context
            relevant_files: Optional list of additional relevant file paths provided
            thinking_mode: Reasoning depth

        Returns:
            Full prompt string with structured investigation context
        """
        context_parts = []

        # Add multi-step investigation context
        context_parts.append(
            f"# Systematic Investigation - Step {step_number}/{total_steps}\n"
        )
        context_parts.append(f"\n## Current Step\n{step}\n")
        context_parts.append(f"\n## Findings So Far\n{findings}\n")

        if hypothesis:
            context_parts.append(f"\n## Current Hypothesis\n{hypothesis}\n")

        context_parts.append(f"\n## Confidence Level\n{confidence}\n")
        context_parts.append(f"\n## Thinking Mode\n{thinking_mode}\n")

        # Add investigation state summary if continuing
        if state.steps:
            context_parts.append(
                f"\n## Previous Investigation Steps ({len(state.steps)})\n"
            )
            for i, prev_step in enumerate(
                state.steps[-3:], start=max(1, len(state.steps) - 2)
            ):
                context_parts.append(f"\nStep {i}:\n")
                context_parts.append(
                    f"  Findings: {prev_step.findings[:150]}...\n"
                    if len(prev_step.findings) > 150
                    else f"  Findings: {prev_step.findings}\n"
                )
                context_parts.append(f"  Confidence: {prev_step.confidence}\n")

        if state.relevant_files:
            context_parts.append(
                f"\n## Files Examined Previously\n{', '.join(state.relevant_files[:10])}"
            )
            if len(state.relevant_files) > 10:
                context_parts.append(f" (and {len(state.relevant_files) - 10} more)")
            context_parts.append("\n")

        if relevant_files:
            additional_files = []
            for file_path in relevant_files:
                if file_path not in additional_files:
                    additional_files.append(file_path)
            if additional_files:
                context_parts.append(
                    "\n## Additional Relevant Files Referenced This Step\n"
                )
                context_parts.append(", ".join(additional_files))
                context_parts.append("\n")

        # Add file contents if provided for this step
        if files:
            context_parts.append("\n## Files for Analysis in This Step\n")
            for file_path in files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        file_content = f.read()
                    context_parts.append(f"\n--- File: {file_path} ---\n")
                    context_parts.append(file_content)
                    context_parts.append(f"\n--- End of {file_path} ---\n")
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    context_parts.append(
                        f"\n--- File: {file_path} (Failed to read: {e}) ---\n"
                    )

        # Add instruction for analysis
        context_parts.append("\n## Task\n")
        context_parts.append(
            "Analyze the investigation step, findings, and hypothesis. Provide:"
        )
        context_parts.append("\n1. Assessment of the current hypothesis")
        context_parts.append("\n2. Analysis of the findings")
        context_parts.append("\n3. Recommendations for next steps")
        context_parts.append("\n4. Updated confidence assessment if warranted\n")

        full_prompt = "\n".join(context_parts)

        logger.debug(
            f"Built investigation prompt - "
            f"step {step_number}/{total_steps}, "
            f"confidence: {confidence}, "
            f"files: {len(files) if files else 0}, "
            f"relevant_files: {len(relevant_files) if relevant_files else 0}, "
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
        lines = response_content.strip().split("\n\n")
        if lines:
            first_paragraph = lines[0].strip()
            if len(first_paragraph) > 500:
                return first_paragraph[:497] + "..."
            return first_paragraph

        # Fallback: truncate to 500 chars
        if len(response_content) > 500:
            return response_content[:497] + "..."
        return response_content.strip()

    async def _perform_expert_validation(
        self, thread_id: str, state: ThinkDeepState, primary_response: str, **kwargs
    ) -> str | None:
        """
        Perform optional expert validation with a different model.

        This method is called when confidence is not "certain" and an expert
        provider is configured. The expert provider reviews the investigation
        findings and provides validation or additional insights.

        Args:
            thread_id: Thread ID of the investigation
            state: Current investigation state
            primary_response: Response from the primary model
            **kwargs: Additional parameters passed to expert provider.generate()

        Returns:
            Expert validation response or None if validation not needed/available
        """
        # Skip if expert validation disabled or no expert provider
        if not self.enable_expert_validation or not self.expert_provider:
            return None

        # Skip if confidence is already "certain"
        if state.current_confidence == ConfidenceLevel.CERTAIN.value:
            logger.info("Skipping expert validation - confidence is 'certain'")
            return None

        logger.info(
            f"Performing expert validation for investigation {thread_id} "
            f"(confidence: {state.current_confidence})"
        )

        # Build expert validation prompt
        validation_prompt = self._build_expert_validation_prompt(
            state, primary_response
        )

        try:
            # Create generation request for expert
            request = GenerationRequest(
                prompt=validation_prompt, continuation_id=thread_id, **kwargs
            )

            logger.info(
                f"Sending expert validation request to provider: "
                f"{self.expert_provider.provider_name}"
            )

            # Generate expert response
            expert_response: GenerationResponse = await self.expert_provider.generate(
                request
            )

            logger.info(
                f"Received expert validation from {self.expert_provider.provider_name}: "
                f"{len(expert_response.content)} chars"
            )

            # Add expert validation to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "assistant",
                    f"[Expert Validation from {self.expert_provider.provider_name}]\n\n{expert_response.content}",
                    workflow_name=self.name,
                    model_provider=self.expert_provider.provider_name,
                    model_name=expert_response.model,
                )

            return expert_response.content

        except Exception as e:
            logger.error(f"Expert validation failed: {e}", exc_info=True)
            return None

    def _build_expert_validation_prompt(
        self, state: ThinkDeepState, primary_response: str
    ) -> str:
        """
        Build prompt for expert validation.

        Args:
            state: Current investigation state
            primary_response: Response from primary model

        Returns:
            Expert validation prompt
        """
        prompt_parts = [
            "You are acting as an expert reviewer for an ongoing investigation.",
            "Review the investigation findings and provide validation, additional insights, or corrections.\n",
            "Current Investigation State:",
            f"- Confidence Level: {state.current_confidence}",
            f"- Hypotheses: {len(state.hypotheses)}",
            f"- Steps Completed: {len(state.steps)}",
            f"- Files Examined: {len(state.relevant_files)}\n",
        ]

        # Add hypotheses summary
        if state.hypotheses:
            prompt_parts.append("Hypotheses:")
            for i, hyp in enumerate(state.hypotheses, 1):
                prompt_parts.append(f"{i}. [{hyp.status.upper()}] {hyp.hypothesis}")
            prompt_parts.append("")

        # Add primary model's latest findings
        prompt_parts.extend(
            [
                "Latest Investigation Findings:",
                primary_response,
                "",
                "Expert Validation Task:",
                "1. Assess the validity of the hypotheses and findings",
                "2. Identify any gaps or alternative explanations",
                "3. Suggest additional investigation steps if needed",
                "4. Recommend whether to increase or maintain confidence level",
                "",
                "Provide your expert assessment:",
            ]
        )

        return "\n".join(prompt_parts)

    def get_investigation_state(self, thread_id: str) -> ThinkDeepState | None:
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
        self, thread_id: str, hypothesis_text: str, evidence: list[str] | None = None
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
            status="active",
        )
        state.hypotheses.append(hypothesis)
        self._save_state(thread_id, state)

        logger.info(f"Added hypothesis to investigation {thread_id}: {hypothesis_text}")
        return True

    def update_hypothesis(
        self,
        thread_id: str,
        hypothesis_text: str,
        new_evidence: list[str] | None = None,
        new_status: str | None = None,
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
                    hyp.status = cast(
                        Literal["active", "disproven", "validated"], new_status
                    )
                break

        if hypothesis_found:
            self._save_state(thread_id, state)
            logger.info(
                f"Updated hypothesis in investigation {thread_id}: {hypothesis_text}"
            )

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
        return self.update_hypothesis(
            thread_id, hypothesis_text, new_status="validated"
        )

    def disprove_hypothesis(self, thread_id: str, hypothesis_text: str) -> bool:
        """
        Mark a hypothesis as disproven.

        Args:
            thread_id: Thread ID of the investigation
            hypothesis_text: The hypothesis to disprove (matched by text)

        Returns:
            True if hypothesis was disproven, False if not found
        """
        return self.update_hypothesis(
            thread_id, hypothesis_text, new_status="disproven"
        )

    def get_active_hypotheses(self, thread_id: str) -> list[Hypothesis]:
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

    def get_all_hypotheses(self, thread_id: str) -> list[Hypothesis]:
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

    def update_confidence(self, thread_id: str, new_confidence: str) -> bool:
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

    def get_confidence(self, thread_id: str) -> str | None:
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
            ConfidenceLevel.ALMOST_CERTAIN.value,
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

    def get_investigation_summary(self, thread_id: str) -> dict[str, Any] | None:
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
            "files_examined": len(state.relevant_files),
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
            logger.warning(
                f"Provider {self.provider.provider_name} API key validation failed"
            )
            return False

        return True

    def __repr__(self) -> str:
        """String representation of the workflow."""
        memory_status = "with memory" if self.conversation_memory else "no memory"
        return (
            f"ThinkDeepWorkflow(provider='{self.provider.provider_name}', "
            f"{memory_status})"
        )
