"""
Argument workflow for structured reasoning and dialectical analysis.

This module implements the ArgumentWorkflow which provides systematic analysis
of claims, arguments, and reasoning through a structured workflow that examines
multiple perspectives and assesses argument strength.
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

from ...core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from ...core.conversation import ConversationMemory
from ...core.registry import WorkflowRegistry
from ...providers import ModelProvider, GenerationRequest, GenerationResponse
from ...core.models import ConversationMessage

logger = logging.getLogger(__name__)


@WorkflowRegistry.register("argument")
class ArgumentWorkflow(BaseWorkflow):
    """
    Structured argument analysis workflow for dialectical reasoning.

    This workflow provides systematic analysis of claims and arguments through
    a multi-step process that examines supporting evidence, counter-arguments,
    and overall argument strength. It's designed for critical thinking, debate
    preparation, and decision-making scenarios.

    Key features:
    - Single provider (focused analysis from one perspective)
    - Multi-step analysis: claim analysis, evidence, counter-arguments, assessment
    - Conversation threading via continuation_id
    - Inherits conversation support from BaseWorkflow
    - Structured dialectical reasoning

    The ArgumentWorkflow is ideal for:
    - Analyzing the strength of arguments and claims
    - Debate preparation and research
    - Critical thinking and decision-making support
    - Examining multiple perspectives on a topic
    - Identifying weaknesses in reasoning

    Workflow Steps:
    1. **Claim Analysis**: Understand and clarify the core argument
    2. **Supporting Evidence**: Identify evidence and reasons supporting the claim
    3. **Counter-Arguments**: Identify objections and opposing viewpoints
    4. **Strength Assessment**: Evaluate overall argument quality and validity

    Example:
        >>> from modelchorus.providers import ClaudeProvider
        >>> from modelchorus.workflows import ArgumentWorkflow
        >>> from modelchorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = ArgumentWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # Analyze an argument
        >>> result = await workflow.run(
        ...     "Universal basic income would reduce poverty and inequality"
        ... )
        >>> print(result.synthesis)
        >>>
        >>> # Continue analysis with follow-up
        >>> result2 = await workflow.run(
        ...     "What about the impact on work incentives?",
        ...     continuation_id=result.metadata.get('thread_id')
        ... )
        >>> print(result2.synthesis)
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize ArgumentWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for argument analysis
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn conversations

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        super().__init__(
            name="Argument",
            description="Structured argument analysis with dialectical reasoning",
            config=config,
            conversation_memory=conversation_memory
        )
        self.provider = provider

        logger.info(f"ArgumentWorkflow initialized with provider: {provider.provider_name}")

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute argument analysis workflow with optional conversation continuation.

        This method performs a structured analysis of the argument or claim provided
        in the prompt. If a continuation_id is provided and conversation_memory is
        available, the conversation history will be loaded and included in the context.

        Args:
            prompt: The argument, claim, or question to analyze
            continuation_id: Optional thread ID to continue an existing conversation
            files: Optional list of file paths to include in conversation context
            **kwargs: Additional parameters passed to provider.generate()
                     (e.g., temperature, max_tokens, system_prompt)

        Returns:
            WorkflowResult containing:
                - success: True if analysis succeeded
                - synthesis: Combined analysis from all steps
                - steps: Four steps (claim analysis, evidence, counter-arguments, assessment)
                - metadata: thread_id, model info, and analysis details

        Raises:
            Exception: If provider.generate() fails

        Example:
            >>> # Fresh analysis
            >>> result = await workflow.run(
            ...     "Remote work increases productivity"
            ... )
            >>>
            >>> # Continuation
            >>> result2 = await workflow.run(
            ...     "How does this vary by industry?",
            ...     continuation_id=result.metadata['thread_id']
            ... )
        """
        logger.info(
            f"Starting argument workflow - prompt length: {len(prompt)}, "
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
            # Build the full prompt with conversation history and file context if available
            full_prompt = self._build_prompt_with_history(prompt, thread_id, files)

            # Step 1: Claim Analysis
            claim_analysis = await self._analyze_claim(full_prompt, thread_id, **kwargs)
            result.add_step(
                step_number=1,
                content=claim_analysis.content,
                model=claim_analysis.model,
                step_name="Claim Analysis"
            )

            # Step 2: Supporting Evidence
            supporting_evidence = await self._gather_supporting_evidence(
                full_prompt, claim_analysis.content, thread_id, **kwargs
            )
            result.add_step(
                step_number=2,
                content=supporting_evidence.content,
                model=supporting_evidence.model,
                step_name="Supporting Evidence"
            )

            # Step 3: Counter-Arguments
            counter_arguments = await self._gather_counter_arguments(
                full_prompt, claim_analysis.content, thread_id, **kwargs
            )
            result.add_step(
                step_number=3,
                content=counter_arguments.content,
                model=counter_arguments.model,
                step_name="Counter-Arguments"
            )

            # Step 4: Strength Assessment
            strength_assessment = await self._assess_argument_strength(
                full_prompt,
                claim_analysis.content,
                supporting_evidence.content,
                counter_arguments.content,
                thread_id,
                **kwargs
            )
            result.add_step(
                step_number=4,
                content=strength_assessment.content,
                model=strength_assessment.model,
                step_name="Strength Assessment"
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

            # Add synthesis as assistant response to conversation history
            synthesis = self.synthesize(result.steps)
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "assistant",
                    synthesis,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name,
                    model_name=claim_analysis.model
                )

            # Build successful result
            result.success = True
            result.synthesis = synthesis

            # Add metadata
            result.metadata.update({
                'thread_id': thread_id,
                'provider': self.provider.provider_name,
                'model': claim_analysis.model,
                'is_continuation': continuation_id is not None,
                'conversation_length': self._get_conversation_length(thread_id),
                'steps_completed': 4
            })

            logger.info(f"Argument workflow completed successfully for thread: {thread_id}")

        except Exception as e:
            logger.error(f"Argument workflow failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.metadata['thread_id'] = thread_id

        # Store result
        self._result = result
        return result

    async def _analyze_claim(
        self,
        prompt: str,
        thread_id: str,
        **kwargs
    ) -> GenerationResponse:
        """
        Step 1: Analyze and clarify the core claim or argument.

        Args:
            prompt: The argument or claim to analyze
            thread_id: Thread ID for context
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse with claim analysis
        """
        analysis_prompt = f"""Analyze the following argument or claim:

{prompt}

Please:
1. Identify the core claim being made
2. Clarify any ambiguous terms or concepts
3. Identify key assumptions underlying the argument
4. Restate the argument in clear, precise terms

Provide a structured analysis of the claim."""

        request = GenerationRequest(
            prompt=analysis_prompt,
            continuation_id=thread_id,
            **kwargs
        )

        return await self.provider.generate(request)

    async def _gather_supporting_evidence(
        self,
        original_prompt: str,
        claim_analysis: str,
        thread_id: str,
        **kwargs
    ) -> GenerationResponse:
        """
        Step 2: Identify evidence and reasoning supporting the claim.

        Args:
            original_prompt: Original user prompt
            claim_analysis: Results from claim analysis step
            thread_id: Thread ID for context
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse with supporting evidence
        """
        evidence_prompt = f"""Based on this claim analysis:

{claim_analysis}

Please identify and explain:
1. Main arguments and evidence supporting this claim
2. Logical reasoning that supports the conclusion
3. Real-world examples or data that back up the claim
4. Theoretical frameworks that support this position

Provide a comprehensive analysis of the supporting side."""

        request = GenerationRequest(
            prompt=evidence_prompt,
            continuation_id=thread_id,
            **kwargs
        )

        return await self.provider.generate(request)

    async def _gather_counter_arguments(
        self,
        original_prompt: str,
        claim_analysis: str,
        thread_id: str,
        **kwargs
    ) -> GenerationResponse:
        """
        Step 3: Identify objections and counter-arguments to the claim.

        Args:
            original_prompt: Original user prompt
            claim_analysis: Results from claim analysis step
            thread_id: Thread ID for context
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse with counter-arguments
        """
        counter_prompt = f"""Based on this claim analysis:

{claim_analysis}

Please identify and explain:
1. Main objections and counter-arguments to this claim
2. Weaknesses in the reasoning or evidence
3. Alternative explanations or perspectives
4. Potential flaws in the underlying assumptions

Provide a comprehensive analysis of the opposing side."""

        request = GenerationRequest(
            prompt=counter_prompt,
            continuation_id=thread_id,
            **kwargs
        )

        return await self.provider.generate(request)

    async def _assess_argument_strength(
        self,
        original_prompt: str,
        claim_analysis: str,
        supporting_evidence: str,
        counter_arguments: str,
        thread_id: str,
        **kwargs
    ) -> GenerationResponse:
        """
        Step 4: Assess overall argument strength and validity.

        Args:
            original_prompt: Original user prompt
            claim_analysis: Results from claim analysis step
            supporting_evidence: Results from supporting evidence step
            counter_arguments: Results from counter-arguments step
            thread_id: Thread ID for context
            **kwargs: Additional generation parameters

        Returns:
            GenerationResponse with strength assessment
        """
        assessment_prompt = f"""Based on the complete argument analysis:

CLAIM ANALYSIS:
{claim_analysis}

SUPPORTING EVIDENCE:
{supporting_evidence}

COUNTER-ARGUMENTS:
{counter_arguments}

Please provide:
1. Overall assessment of argument strength (strong, moderate, weak)
2. Key strengths of the argument
3. Key weaknesses or vulnerabilities
4. Conditions under which the argument is most/least valid
5. Final judgment on the argument's persuasiveness

Provide a balanced, critical assessment."""

        request = GenerationRequest(
            prompt=assessment_prompt,
            continuation_id=thread_id,
            **kwargs
        )

        return await self.provider.generate(request)

    def _build_prompt_with_history(
        self,
        prompt: str,
        thread_id: str,
        files: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt with conversation history and file context if available.

        Args:
            prompt: Current user prompt
            thread_id: Thread ID to load history from
            files: Optional list of file paths to include in context

        Returns:
            Full prompt string with conversation history and file contents prepended
        """
        context_parts = []

        # Add file contents if provided
        if files:
            context_parts.append("File context:\n")
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

        # If no conversation memory, return prompt with file context only
        if not self.conversation_memory:
            if context_parts:
                context_parts.append(f"\n{prompt}")
                return "\n".join(context_parts)
            return prompt

        # Try to load conversation history
        messages = self.resume_conversation(thread_id)

        # Add conversation history
        if messages:
            context_parts.append("\nPrevious conversation:\n")
            for msg in messages:
                role_label = msg.role.upper()
                context_parts.append(f"{role_label}: {msg.content}\n")

                # Include file references from history if present
                if msg.files:
                    context_parts.append(f"  [Referenced files: {', '.join(msg.files)}]\n")

            # Add current user prompt with USER: prefix when there's history
            context_parts.append(f"\nUSER: {prompt}")
        else:
            # No history - just add prompt (with file context if any)
            if context_parts:
                context_parts.append(f"\n{prompt}")
            else:
                return prompt

        full_prompt = "\n".join(context_parts)

        logger.debug(
            f"Built prompt with {len(messages) if messages else 0} previous messages, "
            f"{len(files) if files else 0} files, total length: {len(full_prompt)}"
        )

        return full_prompt

    def _get_conversation_length(self, thread_id: str) -> int:
        """
        Get the number of messages in a conversation thread.

        Args:
            thread_id: Thread ID to check

        Returns:
            Number of messages in the thread, or 0 if not available
        """
        if not self.conversation_memory:
            return 0

        thread = self.get_thread(thread_id)
        if not thread:
            return 0

        return len(thread.messages)

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
            f"ArgumentWorkflow(provider='{self.provider.provider_name}', "
            f"{memory_status})"
        )
