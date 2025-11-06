"""
Ideate workflow for brainstorming and creative idea generation.

This module implements the IdeateWorkflow which provides systematic ideation
through structured brainstorming sessions with creative prompting.
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


@WorkflowRegistry.register("ideate")
class IdeateWorkflow(BaseWorkflow):
    """
    Creative ideation workflow for brainstorming and idea generation.

    This workflow implements structured brainstorming through multiple rounds
    of creative idea generation, helping users explore diverse solutions and
    innovative approaches to problems or challenges.

    Architecture:
    - Single-model creative generation with high temperature
    - Multiple ideation rounds for diverse perspectives
    - Conversation threading for iterative refinement
    - Focus on quantity and creativity over evaluation

    Key Features:
    - Multi-round idea generation
    - Creative prompting strategies
    - Conversation threading via continuation_id
    - Inherits conversation support from BaseWorkflow
    - Structured brainstorming approach

    The IdeateWorkflow is ideal for:
    - Brainstorming new features or solutions
    - Creative problem-solving
    - Exploring innovative approaches
    - Generating diverse perspectives on challenges
    - Early-stage product ideation

    Workflow Pattern:
    1. **Initial Ideation**: Generate diverse initial ideas
    2. **Expansion**: Explore variations and combinations
    3. **Refinement**: Develop promising concepts further

    Example:
        >>> from modelchorus.providers import ClaudeProvider
        >>> from modelchorus.workflows import IdeateWorkflow
        >>> from modelchorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = IdeateWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # Generate ideas
        >>> result = await workflow.run(
        ...     "How can we improve user onboarding?"
        ... )
        >>> print(result.synthesis)
        >>>
        >>> # Refine specific ideas
        >>> result2 = await workflow.run(
        ...     "Expand on the gamification idea",
        ...     continuation_id=result.metadata.get('thread_id')
        ... )
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize IdeateWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for ideation
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn conversations

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        super().__init__(
            name="Ideate",
            description="Creative ideation and brainstorming workflow",
            config=config,
            conversation_memory=conversation_memory
        )
        self.provider = provider

        logger.info(f"IdeateWorkflow initialized with provider: {provider.provider_name}")

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute ideation workflow with creative prompting.

        This method handles brainstorming sessions with enhanced creative prompting
        to generate diverse and innovative ideas. Supports conversation continuation
        for iterative refinement.

        Args:
            prompt: The topic or problem to ideate on
            continuation_id: Optional thread ID to continue an existing ideation session
            files: Optional list of file paths to include in context
            **kwargs: Additional parameters passed to provider.generate()
                     (e.g., temperature, max_tokens, num_ideas)

        Returns:
            WorkflowResult containing:
                - synthesis: Combined ideas and recommendations
                - steps: Individual ideation steps
                - metadata: Thread ID, model info, ideation parameters

        Raises:
            ValueError: If prompt is empty
            Exception: If provider generation fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Get or create thread ID
        thread_id = continuation_id or str(uuid.uuid4())

        # Retrieve conversation history if continuing
        history = []
        if continuation_id and self.conversation_memory:
            thread = self.conversation_memory.get_thread(continuation_id)
            if thread:
                history = thread.messages
                logger.info(f"Loaded {len(history)} messages from thread {continuation_id}")

        # Prepare creative system prompt
        system_prompt = kwargs.get('system_prompt', self._get_ideation_system_prompt())

        # Set high temperature for creativity (unless overridden)
        temperature = kwargs.get('temperature', 0.9)

        # Prepare user message with ideation framing
        user_message = self._frame_ideation_prompt(prompt, is_continuation=bool(history))

        # Create generation request
        request = GenerationRequest(
            messages=history + [ConversationMessage(role="user", content=user_message)],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 4096)
        )

        logger.info(f"Executing ideation for prompt: {prompt[:100]}...")

        try:
            # Generate ideas
            response: GenerationResponse = await self.provider.generate(request)

            # Create workflow step
            step = WorkflowStep(
                step_number=len(history) // 2 + 1,
                title="Ideation Round",
                content=response.content,
                metadata={
                    "model": self.provider.provider_name,
                    "temperature": temperature,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            )

            # Update conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_message(
                    thread_id,
                    ConversationMessage(role="user", content=prompt)
                )
                self.conversation_memory.add_message(
                    thread_id,
                    ConversationMessage(role="assistant", content=response.content)
                )

            # Create workflow result
            result = WorkflowResult(
                synthesis=response.content,
                steps=[step],
                metadata={
                    'thread_id': thread_id,
                    'model': self.provider.provider_name,
                    'workflow': 'ideate',
                    'temperature': temperature,
                    'round': len(history) // 2 + 1
                }
            )

            logger.info(f"Ideation completed successfully. Thread: {thread_id}")
            return result

        except Exception as e:
            logger.error(f"Ideation workflow failed: {e}")
            raise

    def _get_ideation_system_prompt(self) -> str:
        """
        Get the system prompt for creative ideation.

        Returns:
            System prompt optimized for brainstorming and idea generation
        """
        return """You are a creative ideation expert specializing in brainstorming and innovation.

Your role is to generate diverse, creative, and practical ideas to help users explore solutions.

Guidelines for ideation:
1. **Quantity over quality**: Generate many ideas without premature filtering
2. **Diverse perspectives**: Explore different angles and approaches
3. **Build on context**: If this is a continuation, reference and expand previous ideas
4. **Be specific**: Provide concrete, actionable suggestions
5. **Think boldly**: Include innovative and unconventional ideas alongside practical ones
6. **Structured output**: Organize ideas clearly with explanations

Focus on being creative, generative, and helpful. Defer criticism and evaluation for later stages."""

    def _frame_ideation_prompt(self, prompt: str, is_continuation: bool) -> str:
        """
        Frame the user's prompt for optimal ideation.

        Args:
            prompt: The user's original prompt
            is_continuation: Whether this is continuing a previous session

        Returns:
            Enhanced prompt optimized for ideation
        """
        if is_continuation:
            return f"""Continue brainstorming on the topic we've been discussing.

User request: {prompt}

Build on our previous ideas and explore new directions."""
        else:
            return f"""Let's brainstorm creative ideas on the following topic:

{prompt}

Generate diverse, innovative ideas. Think broadly and explore multiple approaches."""

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate ideation workflow configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Currently no required config parameters
        # Future: could validate num_rounds, min_ideas, etc.
        return True

    def get_provider(self) -> ModelProvider:
        """
        Get the provider used by this workflow.

        Returns:
            ModelProvider instance
        """
        return self.provider
