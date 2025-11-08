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
from ...core.role_orchestration import (
    RoleOrchestrator,
    ModelRole,
    OrchestrationPattern,
    OrchestrationResult,
)
from ...providers import ModelProvider, GenerationRequest, GenerationResponse
from ...core.models import ConversationMessage, ArgumentMap, ArgumentPerspective
from ...core.progress import emit_workflow_start, emit_stage, emit_workflow_complete

logger = logging.getLogger(__name__)


@WorkflowRegistry.register("argument")
class ArgumentWorkflow(BaseWorkflow):
    """
    Role-based dialectical reasoning workflow using RoleOrchestrator.

    This workflow implements structured argument analysis through role-based
    orchestration, where different AI roles (Creator, Skeptic, Moderator) examine
    an argument from multiple perspectives to produce balanced dialectical analysis.

    Architecture:
    - Uses RoleOrchestrator for sequential role execution
    - Creator role: Generates strong thesis advocating FOR the position (Step 1)
    - Skeptic role: Provides critical rebuttal AGAINST the position (Step 2)
    - Moderator role: Synthesizes perspectives into balanced analysis (Step 3)

    Current Implementation Status:
    - Step 1 (Creator): ✓ Implemented - Generates thesis with supporting arguments
    - Step 2 (Skeptic): ✓ Implemented - Provides critical rebuttal and counter-arguments
    - Step 3 (Moderator): ✓ Implemented - Synthesizes both perspectives into balanced analysis

    Key Features:
    - Role-based orchestration using RoleOrchestrator
    - Sequential execution pattern (roles build on each other's outputs)
    - Stance-driven prompts (for/against/neutral)
    - Conversation threading via continuation_id
    - Inherits conversation support from BaseWorkflow
    - Structured dialectical reasoning

    The ArgumentWorkflow is ideal for:
    - Analyzing the strength of arguments and claims
    - Debate preparation and research
    - Critical thinking and decision-making support
    - Examining multiple perspectives systematically
    - Identifying both strengths and weaknesses in reasoning

    Workflow Steps (when complete):
    1. **Creator Role (Thesis Generation)**: Build strong case FOR the position
    2. **Skeptic Role (Critical Rebuttal)**: Challenge with counter-arguments
    3. **Moderator Role (Synthesis)**: Integrate perspectives into balanced assessment

    Example:
        >>> from model_chorus.providers import ClaudeProvider
        >>> from model_chorus.workflows import ArgumentWorkflow
        >>> from model_chorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = ArgumentWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # Analyze an argument (all three roles execute)
        >>> result = await workflow.run(
        ...     "Universal basic income would reduce poverty"
        ... )
        >>> print(result.steps[0].content)  # Creator's thesis
        >>> print(result.steps[1].content)  # Skeptic's rebuttal
        >>> print(result.steps[2].content)  # Moderator's synthesis
        >>> print(result.metadata['roles_executed'])  # ['creator', 'skeptic', 'moderator']
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
        fallback_providers: Optional[List[ModelProvider]] = None,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize ArgumentWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for argument analysis
            fallback_providers: Optional list of fallback providers to try if primary fails
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
        self.fallback_providers = fallback_providers or []

        logger.info(f"ArgumentWorkflow initialized with provider: {provider.provider_name}")

    def _create_creator_role(self) -> ModelRole:
        """
        Create Creator role for thesis generation (Step 1 of ARGUMENT workflow).

        The Creator role is responsible for generating a strong initial thesis
        with supporting arguments. This role advocates FOR the position, establishing
        the foundation for the Skeptic's counter-arguments.

        Returns:
            ModelRole configured for Creator with "for" stance

        Example:
            >>> creator = workflow._create_creator_role()
            >>> print(creator.role, creator.stance)
            creator for
        """
        return ModelRole(
            role="creator",
            model=self.provider.provider_name,
            stance="for",
            stance_prompt=(
                "You are a thoughtful argument creator. Your role is to construct a STRONG, "
                "well-reasoned thesis that supports the given position. Present compelling "
                "evidence, logical reasoning, and anticipate potential objections. Be persuasive "
                "and thorough in building your case."
            ),
            system_prompt=(
                "You are an expert at constructing compelling arguments. Focus on:\n"
                "1. Clearly stating the core thesis\n"
                "2. Providing strong supporting evidence and reasoning\n"
                "3. Identifying key assumptions underlying the argument\n"
                "4. Anticipating and addressing potential counter-arguments\n"
                "5. Maintaining intellectual rigor while being persuasive"
            ),
            temperature=0.7,  # Balanced creativity and coherence
            metadata={
                "step": 1,
                "step_name": "Thesis Generation",
                "role_type": "advocate",
            },
        )

    def _create_skeptic_role(self) -> ModelRole:
        """
        Create Skeptic role for critical rebuttal (Step 2 of ARGUMENT workflow).

        The Skeptic role is responsible for providing critical analysis and
        counter-arguments to the Creator's thesis. This role advocates AGAINST
        the position, identifying weaknesses, flaws, and alternative perspectives.

        Returns:
            ModelRole configured for Skeptic with "against" stance

        Example:
            >>> skeptic = workflow._create_skeptic_role()
            >>> print(skeptic.role, skeptic.stance)
            skeptic against
        """
        return ModelRole(
            role="skeptic",
            model=self.provider.provider_name,
            stance="against",
            stance_prompt=(
                "You are a critical skeptic. Your role is to provide a STRONG rebuttal "
                "to the thesis presented. Challenge assumptions, identify logical flaws, "
                "present counter-evidence, and articulate the strongest possible case AGAINST "
                "the position. Be rigorous and thorough in your critique."
            ),
            system_prompt=(
                "You are an expert at critical analysis and identifying argument weaknesses. Focus on:\n"
                "1. Challenging the core thesis and its underlying assumptions\n"
                "2. Identifying logical fallacies or gaps in reasoning\n"
                "3. Presenting counter-evidence and alternative explanations\n"
                "4. Highlighting potential negative consequences or risks\n"
                "5. Articulating the strongest counter-arguments with intellectual rigor"
            ),
            temperature=0.7,  # Balanced creativity and coherence
            metadata={
                "step": 2,
                "step_name": "Critical Rebuttal",
                "role_type": "critic",
            },
        )

    def _create_moderator_role(self) -> ModelRole:
        """
        Create Moderator role for balanced synthesis (Step 3 of ARGUMENT workflow).

        The Moderator role is responsible for synthesizing the Creator's thesis
        and the Skeptic's rebuttal into a balanced, nuanced analysis. This role
        takes a NEUTRAL stance, weighing both perspectives fairly and producing
        a comprehensive assessment that acknowledges strengths and weaknesses.

        Returns:
            ModelRole configured for Moderator with "neutral" stance

        Example:
            >>> moderator = workflow._create_moderator_role()
            >>> print(moderator.role, moderator.stance)
            moderator neutral
        """
        return ModelRole(
            role="moderator",
            model=self.provider.provider_name,
            stance="neutral",
            stance_prompt=(
                "You are an impartial moderator and synthesizer. Your role is to integrate "
                "the thesis and rebuttal into a balanced, comprehensive analysis. Acknowledge "
                "the valid points from both perspectives, identify areas of agreement and "
                "disagreement, and provide nuanced insights. Be fair, thorough, and intellectually "
                "honest in your synthesis."
            ),
            system_prompt=(
                "You are an expert at synthesizing diverse perspectives into balanced analysis. Focus on:\n"
                "1. Summarizing the key points from both the thesis and rebuttal\n"
                "2. Identifying areas where both perspectives have merit\n"
                "3. Highlighting unresolved tensions or genuine disagreements\n"
                "4. Providing nuanced conclusions that acknowledge complexity\n"
                "5. Offering actionable insights or recommendations where appropriate\n"
                "6. Maintaining intellectual rigor while being accessible and clear"
            ),
            temperature=0.7,  # Balanced creativity and coherence
            metadata={
                "step": 3,
                "step_name": "Balanced Synthesis",
                "role_type": "synthesizer",
            },
        )

    def _generate_argument_map(
        self,
        prompt: str,
        creator_response,
        skeptic_response,
        moderator_response,
        synthesis: str,
        metadata: Dict[str, Any]
    ) -> ArgumentMap:
        """
        Generate structured ArgumentMap from role responses.

        Creates an ArgumentMap containing all three perspectives (Creator, Skeptic,
        Moderator) with their stances, content, and metadata. This provides
        programmatic access to the dialectical analysis.

        Args:
            prompt: Original argument topic/claim
            creator_response: Response from Creator role
            skeptic_response: Response from Skeptic role
            moderator_response: Response from Moderator role
            synthesis: Final synthesis text
            metadata: Workflow metadata

        Returns:
            ArgumentMap with all perspectives structured

        Example:
            >>> arg_map = workflow._generate_argument_map(
            ...     "Universal basic income reduces poverty",
            ...     creator_response,
            ...     skeptic_response,
            ...     moderator_response,
            ...     "After examining...",
            ...     {"thread_id": "123"}
            ... )
            >>> print(arg_map.perspectives[0].role)  # 'creator'
        """
        # Create perspective for Creator
        creator_perspective = ArgumentPerspective(
            role="creator",
            stance="for",
            content=creator_response.content,
            key_points=[],  # Could be extracted via NLP in future
            model=creator_response.model,
            metadata={"step": 1, "step_name": "Thesis Generation"}
        )

        # Create perspective for Skeptic
        skeptic_perspective = ArgumentPerspective(
            role="skeptic",
            stance="against",
            content=skeptic_response.content,
            key_points=[],  # Could be extracted via NLP in future
            model=skeptic_response.model,
            metadata={"step": 2, "step_name": "Critical Rebuttal"}
        )

        # Create perspective for Moderator
        moderator_perspective = ArgumentPerspective(
            role="moderator",
            stance="neutral",
            content=moderator_response.content,
            key_points=[],  # Could be extracted via NLP in future
            model=moderator_response.model,
            metadata={"step": 3, "step_name": "Balanced Synthesis"}
        )

        # Create ArgumentMap with all perspectives
        argument_map = ArgumentMap(
            topic=prompt,
            perspectives=[creator_perspective, skeptic_perspective, moderator_perspective],
            synthesis=synthesis,
            metadata=metadata
        )

        return argument_map

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        skip_provider_check: bool = False,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute argument analysis workflow using role-based orchestration.

        This method performs structured dialectical analysis through three sequential
        roles in the complete ARGUMENT workflow (Creator → Skeptic → Moderator).

        The workflow uses RoleOrchestrator to coordinate role-based execution:
        - Creator role: Generates thesis advocating FOR the position
        - Skeptic role: Provides critical rebuttal AGAINST the position
        - Moderator role: Synthesizes perspectives into balanced analysis

        Args:
            prompt: The argument, claim, or question to analyze
            continuation_id: Optional thread ID to continue an existing conversation
            files: Optional list of file paths to include in conversation context
            skip_provider_check: Skip provider availability check (faster startup)
            **kwargs: Additional parameters passed to RoleOrchestrator
                     (e.g., temperature, max_tokens)

        Returns:
            WorkflowResult containing:
                - success: True if analysis succeeded
                - synthesis: Combined analysis from all role perspectives
                - steps: Three steps (Creator thesis, Skeptic rebuttal, Moderator synthesis)
                - metadata: thread_id, model info, role execution details, and argument_map
                  - argument_map: ArgumentMap with structured perspectives and synthesis

        Raises:
            Exception: If role orchestration fails

        Example:
            >>> # Fresh analysis
            >>> result = await workflow.run(
            ...     "Universal basic income would reduce poverty"
            ... )
            >>> print(result.steps[0].content)  # Creator's thesis
            >>> print(result.steps[1].content)  # Skeptic's rebuttal
            >>> print(result.steps[2].content)  # Moderator's synthesis
            >>>
            >>> # Access structured ArgumentMap
            >>> arg_map = result.metadata['argument_map']
            >>> creator = arg_map.get_perspective('creator')
            >>> print(creator.stance)  # 'for'
            >>> print(arg_map.synthesis)  # Final balanced synthesis
            >>>
            >>> # Continuation
            >>> result2 = await workflow.run(
            ...     "What about work incentives?",
            ...     continuation_id=result.metadata['thread_id']
            ... )
        """
        logger.info(
            f"Starting ARGUMENT workflow - prompt length: {len(prompt)}, "
            f"continuation: {continuation_id is not None}, "
            f"files: {len(files) if files else 0}"
        )

        # Check provider availability
        if not skip_provider_check:
            has_available, available, unavailable = await self.check_provider_availability(
                self.provider, self.fallback_providers
            )

            if not has_available:
                from ...providers.cli_provider import ProviderUnavailableError
                error_msg = "No providers available for argument analysis:\n"
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

            # Create roles for ARGUMENT workflow
            creator_role = self._create_creator_role()  # Step 1: Thesis generation
            skeptic_role = self._create_skeptic_role()  # Step 2: Critical rebuttal
            moderator_role = self._create_moderator_role()  # Step 3: Balanced synthesis

            # Set up provider map for orchestrator
            provider_map = {
                self.provider.provider_name: self.provider
            }

            # Create orchestrator with all three roles (SEQUENTIAL pattern)
            orchestrator = RoleOrchestrator(
                roles=[creator_role, skeptic_role, moderator_role],
                provider_map=provider_map,
                pattern=OrchestrationPattern.SEQUENTIAL,
            )

            logger.info("Executing ARGUMENT workflow with Creator → Skeptic → Moderator roles...")

            # Emit workflow start
            emit_workflow_start("argument", "15-30s")

            # Execute orchestration (Creator generates thesis, Skeptic rebuts, Moderator synthesizes)
            emit_stage("Creator")
            orchestration_result: OrchestrationResult = await orchestrator.execute(
                base_prompt=full_prompt,
                context=None,  # No prior context for first step
            )

            # Extract role responses from orchestration result
            # role_responses is a list of (role_name, response) tuples
            if len(orchestration_result.role_responses) >= 3:
                _, creator_response = orchestration_result.role_responses[0]
                _, skeptic_response = orchestration_result.role_responses[1]
                _, moderator_response = orchestration_result.role_responses[2]

                # Add Creator's thesis as Step 1
                result.add_step(
                    step_number=1,
                    content=creator_response.content,
                    model=creator_response.model,
                    role="creator",
                    step_name="Thesis Generation (Creator)"
                )

                logger.info(f"Creator role generated thesis: {len(creator_response.content)} chars")

                # Add Skeptic's rebuttal as Step 2
                result.add_step(
                    step_number=2,
                    content=skeptic_response.content,
                    model=skeptic_response.model,
                    role="skeptic",
                    step_name="Critical Rebuttal (Skeptic)"
                )

                logger.info(f"Skeptic role generated rebuttal: {len(skeptic_response.content)} chars")

                # Add Moderator's synthesis as Step 3
                result.add_step(
                    step_number=3,
                    content=moderator_response.content,
                    model=moderator_response.model,
                    role="moderator",
                    step_name="Balanced Synthesis (Moderator)"
                )

                logger.info(f"Moderator role generated synthesis: {len(moderator_response.content)} chars")
            else:
                raise ValueError(
                    f"Expected 3 role responses, got {len(orchestration_result.role_responses)}"
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

            # Generate synthesis from role outputs
            synthesis = self.synthesize(result.steps)

            # Add synthesis as assistant response to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "assistant",
                    synthesis,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name,
                    model_name=creator_response.model
                )

            # Build successful result
            result.success = True
            result.synthesis = synthesis

            # Generate structured ArgumentMap
            base_metadata = {
                'thread_id': thread_id,
                'provider': self.provider.provider_name,
                'model': creator_response.model,
                'is_continuation': continuation_id is not None,
                'conversation_length': self._get_conversation_length(thread_id),
                'workflow_pattern': 'role_orchestration',
                'orchestration_pattern': OrchestrationPattern.SEQUENTIAL.value,
                'roles_executed': ['creator', 'skeptic', 'moderator'],
                'steps_completed': 3,  # Creator + Skeptic + Moderator
            }

            argument_map = self._generate_argument_map(
                prompt=prompt,
                creator_response=creator_response,
                skeptic_response=skeptic_response,
                moderator_response=moderator_response,
                synthesis=synthesis,
                metadata=base_metadata
            )

            # Add metadata including ArgumentMap
            result.metadata.update(base_metadata)
            result.metadata['argument_map'] = argument_map

            logger.info(f"ARGUMENT workflow completed successfully for thread: {thread_id}")
            logger.info(f"Creator role: {len(creator_response.content)} chars thesis")
            logger.info(f"Skeptic role: {len(skeptic_response.content)} chars rebuttal")
            logger.info(f"Moderator role: {len(moderator_response.content)} chars synthesis")

            # Emit workflow complete
            emit_workflow_complete("argument")

        except Exception as e:
            logger.error(f"Argument workflow failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.metadata['thread_id'] = thread_id

        # Store result
        self._result = result
        return result

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
