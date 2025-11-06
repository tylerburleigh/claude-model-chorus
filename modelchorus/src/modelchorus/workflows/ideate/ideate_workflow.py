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
from ...core.role_orchestration import (
    RoleOrchestrator,
    ModelRole,
    OrchestrationPattern,
    OrchestrationResult,
)
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

    def _create_brainstormer_role(self, model_name: str, perspective: str) -> ModelRole:
        """
        Create a brainstormer role for parallel ideation.

        Args:
            model_name: Name of the model to use
            perspective: The perspective this brainstormer should take
                        (e.g., 'practical', 'innovative', 'user-focused')

        Returns:
            ModelRole configured for creative brainstorming
        """
        perspective_prompts = {
            'practical': (
                "Focus on pragmatic, implementable ideas that can be executed "
                "with existing resources. Emphasize feasibility and near-term value."
            ),
            'innovative': (
                "Think boldly and push boundaries. Explore cutting-edge approaches, "
                "emerging technologies, and unconventional solutions."
            ),
            'user-focused': (
                "Prioritize user experience and user needs. Generate ideas that "
                "directly improve user satisfaction, engagement, and value."
            ),
            'technical': (
                "Consider technical architecture and implementation details. "
                "Focus on scalability, performance, and technical excellence."
            ),
            'business': (
                "Think about business impact, ROI, and strategic alignment. "
                "Emphasize ideas that drive revenue, reduce costs, or create competitive advantage."
            ),
        }

        stance_prompt = perspective_prompts.get(
            perspective,
            "Generate diverse, creative ideas from your unique perspective."
        )

        return ModelRole(
            role=f"brainstormer-{perspective}",
            model=model_name,
            stance="neutral",  # No for/against in ideation
            stance_prompt=stance_prompt,
            system_prompt=self._get_ideation_system_prompt(),
            temperature=0.9,  # High temperature for creativity
            metadata={
                "perspective": perspective,
                "role_type": "ideator",
            },
        )

    async def run_parallel_brainstorming(
        self,
        prompt: str,
        provider_map: Dict[str, ModelProvider],
        perspectives: Optional[List[str]] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute parallel divergent brainstorming with multiple models.

        This method uses RoleOrchestrator to run multiple models in parallel,
        each bringing a different perspective to the brainstorming session.
        This creates diverse, creative output by combining different viewpoints.

        Args:
            prompt: The topic or problem to ideate on
            provider_map: Dictionary mapping model names to provider instances
                         (e.g., {"claude": claude_provider, "gemini": gemini_provider})
            perspectives: List of perspectives to use (default: ['practical', 'innovative', 'user-focused'])
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            WorkflowResult containing:
                - synthesis: Combined ideas from all perspectives
                - steps: Individual brainstorming outputs from each model
                - metadata: Thread ID, models used, perspectives

        Raises:
            ValueError: If prompt is empty or provider_map is empty
            Exception: If orchestration fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not provider_map:
            raise ValueError("Provider map cannot be empty")

        # Default perspectives
        if perspectives is None:
            perspectives = ['practical', 'innovative', 'user-focused']

        # Ensure we don't have more perspectives than providers
        if len(perspectives) > len(provider_map):
            perspectives = perspectives[:len(provider_map)]

        # Create roles for each perspective
        roles = []
        model_names = list(provider_map.keys())
        for i, perspective in enumerate(perspectives):
            model_name = model_names[i % len(model_names)]
            role = self._create_brainstormer_role(model_name, perspective)
            roles.append(role)

        logger.info(
            f"Starting parallel brainstorming with {len(roles)} perspectives: {perspectives}"
        )

        # Create orchestrator with PARALLEL pattern
        orchestrator = RoleOrchestrator(
            roles=roles,
            provider_map=provider_map,
            pattern=OrchestrationPattern.PARALLEL
        )

        # Execute parallel brainstorming
        try:
            orchestration_result: OrchestrationResult = await orchestrator.execute(
                prompt=prompt,
                conversation_history=[]
            )

            # Convert orchestration steps to workflow steps
            steps = []
            for i, (role_name, response) in enumerate(orchestration_result.role_responses):
                perspective = perspectives[i] if i < len(perspectives) else "general"
                step = WorkflowStep(
                    step_number=i + 1,
                    title=f"Brainstorming ({perspective})",
                    content=response.content,
                    metadata={
                        "perspective": perspective,
                        "role": role_name,
                        "model": roles[i].model,
                        "tokens": response.usage.total_tokens if response.usage else 0
                    }
                )
                steps.append(step)

            # Create synthesis of all perspectives
            synthesis = self._synthesize_brainstorming_results(steps, perspectives)

            # Create workflow result
            result = WorkflowResult(
                synthesis=synthesis,
                steps=steps,
                metadata={
                    'workflow': 'ideate-parallel',
                    'perspectives': perspectives,
                    'models_used': [role.model for role in roles],
                    'pattern': 'parallel',
                    'total_ideas': len(steps)
                }
            )

            logger.info(f"Parallel brainstorming completed with {len(steps)} perspectives")
            return result

        except Exception as e:
            logger.error(f"Parallel brainstorming failed: {e}")
            raise

    def _synthesize_brainstorming_results(
        self,
        steps: List[WorkflowStep],
        perspectives: List[str]
    ) -> str:
        """
        Synthesize parallel brainstorming results into a cohesive summary.

        Args:
            steps: List of brainstorming steps from different perspectives
            perspectives: List of perspective names

        Returns:
            Synthesized summary combining all perspectives
        """
        synthesis_parts = [
            "# Parallel Brainstorming Results\n",
            f"\nGenerated {len(steps)} sets of ideas from different perspectives:\n"
        ]

        for i, step in enumerate(steps):
            perspective = perspectives[i] if i < len(perspectives) else "general"
            synthesis_parts.append(f"\n## {perspective.title()} Perspective\n")
            synthesis_parts.append(step.content)

        synthesis_parts.append(
            "\n\n---\n\n"
            "**Next Steps**: Review the ideas above from each perspective. "
            "Identify common themes, unique innovations, and ideas worth developing further."
        )

        return "\n".join(synthesis_parts)

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
