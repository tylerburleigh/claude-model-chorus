"""
Ideate workflow for brainstorming and creative idea generation.

This module implements the IdeateWorkflow which provides systematic ideation
through structured brainstorming sessions with creative prompting.
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
        >>> from model_chorus.providers import ClaudeProvider
        >>> from model_chorus.workflows import IdeateWorkflow
        >>> from model_chorus.core.conversation import ConversationMemory
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
        fallback_providers: Optional[List[ModelProvider]] = None,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize IdeateWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for ideation
            fallback_providers: Optional list of fallback providers to try if primary fails
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
        self.fallback_providers = fallback_providers or []

        logger.info(f"IdeateWorkflow initialized with provider: {provider.provider_name}")

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        skip_provider_check: bool = False,
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
            skip_provider_check: Skip provider availability check (faster startup)
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

        # Check provider availability
        if not skip_provider_check:
            has_available, available, unavailable = await self.check_provider_availability(
                self.provider, self.fallback_providers
            )

            if not has_available:
                from ...providers.cli_provider import ProviderUnavailableError
                error_msg = "No providers available for ideation:\n"
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

        # Prepare creative system prompt
        system_prompt = kwargs.get('system_prompt', self._get_ideation_system_prompt())

        # Set high temperature for creativity (unless overridden)
        temperature = kwargs.get('temperature', 0.9)

        # Prepare user message with ideation framing
        user_message = self._frame_ideation_prompt(prompt, is_continuation=bool(history))

        # Prepend conversation history if it exists
        if history:
            history_str = "\n".join(
                [f"{msg.role.upper()}: {msg.content}" for msg in history]
            )
            user_message = f"Previous conversation:\n{history_str}\n\n{user_message}"

        # Create generation request
        request = GenerationRequest(
            prompt=user_message,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 4096),
            continuation_id=thread_id if continuation_id else None
        )

        logger.info(f"Executing ideation for prompt: {prompt[:100]}...")

        try:
            # Generate ideas with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed before success: {', '.join(failed)}")

            # Create workflow step
            step = WorkflowStep(
                step_number=len(history) // 2 + 1,
                content=response.content,
                model=self.provider.provider_name,
                metadata={
                    "title": "Ideation Round",
                    "temperature": temperature,
                    "tokens_used": response.usage.get('total_tokens', 0) if response.usage else 0
                }
            )

            # Update conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_message(
                    thread_id=thread_id,
                    role="user",
                    content=prompt
                )
                self.conversation_memory.add_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=response.content,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name
                )

            # Create workflow result
            result = WorkflowResult(
                success=True,
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

            # Emit workflow complete
            emit_workflow_complete("ideate")

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
                base_prompt=prompt
            )

            # Convert orchestration steps to workflow steps
            steps = []
            for i, (role_name, response) in enumerate(orchestration_result.role_responses):
                perspective = perspectives[i] if i < len(perspectives) else "general"
                step = WorkflowStep(
                    step_number=i + 1,
                    content=response.content,
                    model=roles[i].model,
                    metadata={
                        "title": f"Brainstorming ({perspective})",
                        "perspective": perspective,
                        "role": role_name,
                        "tokens": response.usage.get('total_tokens', 0) if response.usage else 0
                    }
                )
                steps.append(step)

            # Create synthesis of all perspectives
            synthesis = self._synthesize_brainstorming_results(steps, perspectives)

            # Create workflow result
            result = WorkflowResult(success=True, 
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

    async def run_convergent_analysis(
        self,
        brainstorming_result: WorkflowResult,
        scoring_criteria: Optional[List[str]] = None,
        num_clusters: Optional[int] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute convergent analysis on parallel brainstorming results.

        This method takes diverse ideas from parallel brainstorming and performs:
        1. Idea extraction from perspective-based outputs
        2. Clustering of similar ideas into themes
        3. Multi-criteria scoring and ranking

        Args:
            brainstorming_result: WorkflowResult from run_parallel_brainstorming()
            scoring_criteria: List of criteria to score on (default: ['feasibility', 'impact', 'novelty'])
            num_clusters: Target number of clusters (default: auto-determine 3-7)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            WorkflowResult containing:
                - synthesis: Formatted analysis with clusters, scores, and recommendations
                - steps: [extraction, clustering, scoring]
                - metadata: Analysis parameters and summary statistics

        Raises:
            ValueError: If brainstorming_result is None or has no steps
            Exception: If analysis fails
        """
        if brainstorming_result is None or not brainstorming_result.steps:
            raise ValueError("Brainstorming result must have steps to analyze")

        # Default scoring criteria
        if scoring_criteria is None:
            scoring_criteria = ['feasibility', 'impact', 'novelty']

        logger.info(
            f"Starting convergent analysis on {len(brainstorming_result.steps)} brainstorming outputs"
        )

        # Step 1: Extract ideas from brainstorming results
        extraction_result = await self._extract_ideas(brainstorming_result)

        # Step 2: Cluster similar ideas
        clustering_result = await self._cluster_ideas(
            extraction_result,
            num_clusters=num_clusters,
            **kwargs
        )

        # Step 3: Score ideas/clusters
        scoring_result = await self._score_ideas(
            clustering_result,
            scoring_criteria=scoring_criteria,
            **kwargs
        )

        # Create synthesis of analysis
        synthesis = self._synthesize_convergent_analysis(
            extraction_result,
            clustering_result,
            scoring_result,
            scoring_criteria
        )

        # Combine all steps
        steps = [
            extraction_result,
            clustering_result,
            scoring_result
        ]

        # Create workflow result
        result = WorkflowResult(success=True, 
            synthesis=synthesis,
            steps=steps,
            metadata={
                'workflow': 'ideate-convergent',
                'num_ideas_extracted': extraction_result.metadata.get('num_ideas', 0),
                'num_clusters': clustering_result.metadata.get('num_clusters', 0),
                'scoring_criteria': scoring_criteria,
                'source_perspectives': brainstorming_result.metadata.get('perspectives', [])
            }
        )

        logger.info("Convergent analysis completed successfully")
        return result

    async def _extract_ideas(
        self,
        brainstorming_result: WorkflowResult,
        **kwargs
    ) -> WorkflowStep:
        """
        Extract individual ideas from brainstorming results.

        Parses the narrative outputs from each perspective and extracts
        discrete ideas with source metadata.

        Args:
            brainstorming_result: WorkflowResult from parallel brainstorming
            **kwargs: Additional parameters for provider

        Returns:
            WorkflowStep containing extracted ideas with metadata
        """
        logger.info("Extracting ideas from brainstorming results")

        # Compile all brainstorming content
        perspectives_content = []
        for step in brainstorming_result.steps:
            perspective = step.metadata.get('perspective', 'unknown')
            perspectives_content.append({
                'perspective': perspective,
                'content': step.content,
                'model': step.metadata.get('model', 'unknown')
            })

        # Create extraction prompt
        extraction_prompt = self._create_extraction_prompt(perspectives_content)

        # Set low temperature for structured extraction
        temperature = kwargs.get('temperature', 0.3)

        # Create generation request
        request = GenerationRequest(
            prompt=extraction_prompt,
            system_prompt=self._get_extraction_system_prompt(),
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 3000)
        )

        try:
            # Execute extraction with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed for idea extraction: {', '.join(failed)}")

            # Parse extracted ideas from response
            ideas = self._parse_extracted_ideas(response.content, perspectives_content)

            # Create workflow step
            step = WorkflowStep(
                step_number=1,
                content=response.content,
                model=self.provider.provider_name,
                metadata={
                    'title': 'Idea Extraction',
                    'num_ideas': len(ideas),
                    'perspectives_analyzed': len(perspectives_content),
                    'extracted_ideas': ideas,
                    'temperature': temperature
                }
            )

            logger.info(f"Extracted {len(ideas)} ideas from brainstorming results")
            return step

        except Exception as e:
            logger.error(f"Idea extraction failed: {e}")
            raise

    def _create_extraction_prompt(
        self,
        perspectives_content: List[Dict[str, str]]
    ) -> str:
        """
        Create prompt for extracting individual ideas.

        Args:
            perspectives_content: List of perspective outputs

        Returns:
            Formatted extraction prompt
        """
        prompt_parts = [
            "You are analyzing brainstorming results from multiple perspectives. "
            "Extract individual, discrete ideas from the following outputs.\n\n"
            "For each idea:\n"
            "1. State it clearly and concisely (1-2 sentences)\n"
            "2. Note which perspective(s) it came from\n"
            "3. Give it a brief descriptive label\n\n"
            "Brainstorming Results:\n\n"
        ]

        for i, pc in enumerate(perspectives_content, 1):
            prompt_parts.append(f"--- Perspective {i}: {pc['perspective'].title()} ---\n")
            prompt_parts.append(f"{pc['content']}\n\n")

        prompt_parts.append(
            "---\n\n"
            "Now extract all discrete ideas. Format each as:\n\n"
            "**[IDEA-{number}] {Brief Label}** (from {perspective})\n"
            "{1-2 sentence description}\n\n"
            "Extract all unique ideas, combining duplicates where appropriate."
        )

        return "".join(prompt_parts)

    def _get_extraction_system_prompt(self) -> str:
        """
        Get system prompt for idea extraction.

        Returns:
            System prompt for extraction task
        """
        return """You are an expert at analyzing brainstorming sessions and extracting discrete ideas.

Your task is to:
1. Read through all perspectives carefully
2. Identify each unique idea mentioned
3. Extract ideas clearly and concisely
4. Combine duplicate ideas from different perspectives
5. Preserve the source perspective for each idea

Be thorough but focused. Extract concrete ideas, not vague statements.
Use the requested format exactly."""

    def _parse_extracted_ideas(
        self,
        extraction_content: str,
        perspectives_content: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Parse extracted ideas from LLM response.

        Args:
            extraction_content: LLM response with extracted ideas
            perspectives_content: Original perspective outputs

        Returns:
            List of idea dictionaries with metadata
        """
        ideas = []

        # Split by idea markers
        lines = extraction_content.split('\n')
        current_idea = None

        for line in lines:
            line = line.strip()

            # Check for idea header like "**[IDEA-1] Label** (from perspective)"
            if line.startswith('**[IDEA-') and ']' in line:
                # Save previous idea if exists
                if current_idea and current_idea.get('description'):
                    ideas.append(current_idea)

                # Parse new idea header
                try:
                    # Extract components
                    idea_num = line.split('[IDEA-')[1].split(']')[0]

                    # Extract label (between ] and **)
                    label_part = line.split(']')[1].split('**')[0].strip()

                    # Extract perspective (between parentheses)
                    perspective = 'unknown'
                    if '(' in line and ')' in line:
                        perspective = line.split('(from ')[-1].split(')')[0].strip()

                    current_idea = {
                        'id': f"idea-{idea_num}",
                        'label': label_part,
                        'perspective': perspective,
                        'description': ''
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse idea header: {line} - {e}")
                    continue

            elif current_idea is not None and line and not line.startswith('**[IDEA-'):
                # Accumulate description lines
                if current_idea['description']:
                    current_idea['description'] += ' ' + line
                else:
                    current_idea['description'] = line

        # Don't forget the last idea
        if current_idea and current_idea.get('description'):
            ideas.append(current_idea)

        logger.info(f"Parsed {len(ideas)} ideas from extraction content")
        return ideas

    async def _cluster_ideas(
        self,
        extraction_step: WorkflowStep,
        num_clusters: Optional[int] = None,
        **kwargs
    ) -> WorkflowStep:
        """
        Cluster similar ideas into thematic groups.

        Uses LLM to identify common themes and group related ideas.

        Args:
            extraction_step: WorkflowStep from idea extraction
            num_clusters: Target number of clusters (None = auto-determine)
            **kwargs: Additional parameters for provider

        Returns:
            WorkflowStep containing clustered ideas with metadata
        """
        logger.info("Clustering ideas into thematic groups")

        # Get extracted ideas from metadata
        ideas = extraction_step.metadata.get('extracted_ideas', [])

        if not ideas:
            raise ValueError("No ideas found in extraction step")

        # Create clustering prompt
        clustering_prompt = self._create_clustering_prompt(ideas, num_clusters)

        # Set moderate temperature for creative but structured clustering
        temperature = kwargs.get('temperature', 0.5)

        # Create generation request
        request = GenerationRequest(
            prompt=clustering_prompt,
            system_prompt=self._get_clustering_system_prompt(),
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 3000)
        )

        try:
            # Execute clustering with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed for idea clustering: {', '.join(failed)}")

            # Parse clusters from response
            clusters = self._parse_clusters(response.content, ideas)

            # Create workflow step
            step = WorkflowStep(
                step_number=2,
                content=response.content,
                model=self.provider.provider_name,
                metadata={
                    'title': 'Idea Clustering',
                    'num_clusters': len(clusters),
                    'num_ideas': len(ideas),
                    'clusters': clusters,
                    'temperature': temperature
                }
            )

            logger.info(f"Clustered {len(ideas)} ideas into {len(clusters)} themes")
            return step

        except Exception as e:
            logger.error(f"Idea clustering failed: {e}")
            raise

    def _create_clustering_prompt(
        self,
        ideas: List[Dict[str, Any]],
        num_clusters: Optional[int]
    ) -> str:
        """
        Create prompt for clustering ideas.

        Args:
            ideas: List of extracted ideas
            num_clusters: Target number of clusters (None = auto)

        Returns:
            Formatted clustering prompt
        """
        cluster_guidance = (
            f"Organize these into exactly {num_clusters} thematic clusters."
            if num_clusters
            else "Organize these into 3-7 thematic clusters (choose the number that makes most sense)."
        )

        prompt_parts = [
            f"You are clustering brainstormed ideas into themes. {cluster_guidance}\n\n"
            "Ideas to cluster:\n\n"
        ]

        for idea in ideas:
            prompt_parts.append(
                f"**{idea['id'].upper()}**: {idea['label']}\n"
                f"{idea['description']}\n"
                f"(from {idea['perspective']})\n\n"
            )

        prompt_parts.append(
            "---\n\n"
            "Now create thematic clusters. For each cluster:\n\n"
            "**[CLUSTER-{number}] {Theme Name}**\n"
            "{Brief description of this theme}\n\n"
            "Ideas in this cluster:\n"
            "- {IDEA-ID}: {Brief note on why it fits}\n"
            "- {IDEA-ID}: {Brief note on why it fits}\n\n"
            "Focus on meaningful themes. Ideas can fit multiple themes - assign to best fit."
        )

        return "".join(prompt_parts)

    def _get_clustering_system_prompt(self) -> str:
        """
        Get system prompt for idea clustering.

        Returns:
            System prompt for clustering task
        """
        return """You are an expert at thematic analysis and clustering ideas.

Your task is to:
1. Identify common themes across the ideas
2. Group related ideas into coherent clusters
3. Name each cluster with a clear, descriptive theme
4. Explain why ideas belong in each cluster

Guidelines:
- Create 3-7 clusters (fewer is better if ideas are very related)
- Each cluster should have a clear, distinct theme
- Assign each idea to exactly one cluster (best fit)
- Provide meaningful cluster names
- Explain the grouping logic

Use the requested format exactly."""

    def _parse_clusters(
        self,
        clustering_content: str,
        ideas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse clusters from LLM response.

        Args:
            clustering_content: LLM response with clusters
            ideas: Original list of ideas

        Returns:
            List of cluster dictionaries
        """
        clusters = []

        lines = clustering_content.split('\n')
        current_cluster = None

        for line in lines:
            line = line.strip()

            # Check for cluster header like "**[CLUSTER-1] Theme Name**"
            if line.startswith('**[CLUSTER-') and ']' in line:
                # Save previous cluster if exists
                if current_cluster and current_cluster.get('ideas'):
                    clusters.append(current_cluster)

                # Parse new cluster header
                try:
                    cluster_num = line.split('[CLUSTER-')[1].split(']')[0]
                    theme_name = line.split(']')[1].split('**')[0].strip()

                    current_cluster = {
                        'id': f"cluster-{cluster_num}",
                        'theme': theme_name,
                        'description': '',
                        'ideas': []
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse cluster header: {line} - {e}")
                    continue

            elif current_cluster is not None:
                # Check for idea assignment like "- IDEA-1: reason"
                if line.startswith('- ') and 'IDEA-' in line.upper():
                    try:
                        # Extract idea ID (case-insensitive)
                        idea_part = line[2:].split(':')[0].strip().lower()

                        # Extract reason
                        reason = ':'.join(line[2:].split(':')[1:]).strip() if ':' in line else ''

                        current_cluster['ideas'].append({
                            'idea_id': idea_part,
                            'reason': reason
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse idea assignment: {line} - {e}")

                # Accumulate description (lines before "Ideas in this cluster:")
                elif (
                    'ideas in this cluster' not in line.lower()
                    and not line.startswith('**[CLUSTER-')
                    and line
                    and not current_cluster.get('ideas')
                ):
                    if current_cluster['description']:
                        current_cluster['description'] += ' ' + line
                    else:
                        current_cluster['description'] = line

        # Don't forget the last cluster
        if current_cluster and current_cluster.get('ideas'):
            clusters.append(current_cluster)

        logger.info(f"Parsed {len(clusters)} clusters from clustering content")
        return clusters

    async def _score_ideas(
        self,
        clustering_step: WorkflowStep,
        scoring_criteria: List[str],
        **kwargs
    ) -> WorkflowStep:
        """
        Score ideas/clusters based on multiple criteria.

        Uses LLM to evaluate each cluster against criteria and assign scores.

        Args:
            clustering_step: WorkflowStep from idea clustering
            scoring_criteria: List of criteria to score on
            **kwargs: Additional parameters for provider

        Returns:
            WorkflowStep containing scored clusters with metadata
        """
        logger.info(f"Scoring clusters on criteria: {scoring_criteria}")

        # Get clusters from metadata
        clusters = clustering_step.metadata.get('clusters', [])

        if not clusters:
            raise ValueError("No clusters found in clustering step")

        # Create scoring prompt
        scoring_prompt = self._create_scoring_prompt(clusters, scoring_criteria)

        # Set low temperature for consistent scoring
        temperature = kwargs.get('temperature', 0.3)

        # Create generation request
        request = GenerationRequest(
            prompt=scoring_prompt,
            system_prompt=self._get_scoring_system_prompt(),
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 3000)
        )

        try:
            # Execute scoring with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed for idea scoring: {', '.join(failed)}")

            # Parse scores from response
            scored_clusters = self._parse_scores(response.content, clusters, scoring_criteria)

            # Create workflow step
            step = WorkflowStep(
                step_number=3,
                content=response.content,
                model=self.provider.provider_name,
                metadata={
                    'title': 'Idea Scoring',
                    'num_clusters': len(scored_clusters),
                    'scoring_criteria': scoring_criteria,
                    'scored_clusters': scored_clusters,
                    'temperature': temperature
                }
            )

            logger.info(f"Scored {len(scored_clusters)} clusters on {len(scoring_criteria)} criteria")
            return step

        except Exception as e:
            logger.error(f"Idea scoring failed: {e}")
            raise

    def _create_scoring_prompt(
        self,
        clusters: List[Dict[str, Any]],
        scoring_criteria: List[str]
    ) -> str:
        """
        Create prompt for scoring clusters.

        Args:
            clusters: List of clusters with ideas
            scoring_criteria: List of criteria names

        Returns:
            Formatted scoring prompt
        """
        # Define scoring rubric
        criteria_descriptions = {
            'feasibility': 'How practical and achievable is this? (1=very difficult, 5=very easy)',
            'impact': 'How much value/improvement would this create? (1=minimal, 5=transformative)',
            'novelty': 'How innovative and unique is this? (1=common, 5=groundbreaking)',
            'effort': 'How much work would this require? (1=extensive, 5=minimal)',
            'risk': 'How risky is this to implement? (1=very risky, 5=very safe)',
            'user_value': 'How much would users benefit? (1=minimal, 5=huge benefit)'
        }

        prompt_parts = [
            f"You are evaluating idea clusters on {len(scoring_criteria)} criteria. "
            f"Score each cluster on a scale of 1-5 for each criterion.\n\n"
            "Scoring Criteria:\n"
        ]

        for criterion in scoring_criteria:
            description = criteria_descriptions.get(
                criterion,
                f"Evaluate {criterion} (1=low, 5=high)"
            )
            prompt_parts.append(f"- **{criterion.title()}**: {description}\n")

        prompt_parts.append("\n\nClusters to Score:\n\n")

        for cluster in clusters:
            prompt_parts.append(f"**{cluster['id'].upper()}**: {cluster['theme']}\n")
            prompt_parts.append(f"{cluster['description']}\n")
            prompt_parts.append(f"Ideas: {len(cluster.get('ideas', []))}\n\n")

        prompt_parts.append(
            "---\n\n"
            "Now score each cluster. Format:\n\n"
            "**[SCORE-{cluster-id}] {Theme Name}**\n\n"
            "Scores:\n"
        )

        for criterion in scoring_criteria:
            prompt_parts.append(f"- {criterion.title()}: {'{score}'}/5 - {'{brief explanation}'}\n")

        prompt_parts.append(
            "\n"
            "Overall Score: {average}/5\n"
            "Recommendation: {High Priority|Medium Priority|Low Priority}\n\n"
        )

        return "".join(prompt_parts)

    def _get_scoring_system_prompt(self) -> str:
        """
        Get system prompt for idea scoring.

        Returns:
            System prompt for scoring task
        """
        return """You are an expert at evaluating and prioritizing ideas.

Your task is to:
1. Score each cluster on the given criteria (1-5 scale)
2. Provide brief justification for each score
3. Calculate overall score (average of criteria)
4. Recommend priority level based on scores

Guidelines:
- Be objective and consistent in scoring
- Consider trade-offs between criteria
- Provide specific reasoning for scores
- Use the full 1-5 range appropriately
- High Priority: Overall score >= 4.0
- Medium Priority: Overall score 3.0-3.9
- Low Priority: Overall score < 3.0

Use the requested format exactly."""

    def _parse_scores(
        self,
        scoring_content: str,
        clusters: List[Dict[str, Any]],
        scoring_criteria: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse scores from LLM response.

        Args:
            scoring_content: LLM response with scores
            clusters: Original list of clusters
            scoring_criteria: List of criteria names

        Returns:
            List of scored cluster dictionaries
        """
        scored_clusters = []

        lines = scoring_content.split('\n')
        current_score = None

        for line in lines:
            line = line.strip()

            # Check for score header like "**[SCORE-cluster-1] Theme Name**"
            if line.startswith('**[SCORE-') and ']' in line:
                # Save previous scored cluster if exists
                if current_score and current_score.get('scores'):
                    scored_clusters.append(current_score)

                # Parse new score header
                try:
                    cluster_id = line.split('[SCORE-')[1].split(']')[0]
                    theme_name = line.split(']')[1].split('**')[0].strip()

                    current_score = {
                        'cluster_id': cluster_id,
                        'theme': theme_name,
                        'scores': {},
                        'overall_score': 0.0,
                        'recommendation': 'Unknown'
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse score header: {line} - {e}")
                    continue

            elif current_score is not None:
                # Parse individual criterion scores like "- Feasibility: 4/5 - explanation"
                if line.startswith('- ') and ':' in line and '/5' in line:
                    try:
                        criterion_part = line[2:].split(':')[0].strip().lower()
                        score_part = line.split(':')[1].strip()

                        # Extract numeric score
                        score_str = score_part.split('/')[0].strip()
                        score_value = float(score_str)

                        # Extract explanation (after the score)
                        explanation = score_part.split('-', 1)[1].strip() if '-' in score_part else ''

                        current_score['scores'][criterion_part] = {
                            'score': score_value,
                            'explanation': explanation
                        }
                    except Exception as e:
                        logger.warning(f"Failed to parse criterion score: {line} - {e}")

                # Parse overall score like "Overall Score: 4.2/5"
                elif 'overall score:' in line.lower():
                    try:
                        score_part = line.split(':')[1].strip()
                        overall_str = score_part.split('/')[0].strip()
                        current_score['overall_score'] = float(overall_str)
                    except Exception as e:
                        logger.warning(f"Failed to parse overall score: {line} - {e}")

                # Parse recommendation like "Recommendation: High Priority"
                elif 'recommendation:' in line.lower():
                    try:
                        rec_part = line.split(':')[1].strip()
                        current_score['recommendation'] = rec_part
                    except Exception as e:
                        logger.warning(f"Failed to parse recommendation: {line} - {e}")

        # Don't forget the last scored cluster
        if current_score and current_score.get('scores'):
            scored_clusters.append(current_score)

        logger.info(f"Parsed scores for {len(scored_clusters)} clusters")
        return scored_clusters

    def _synthesize_convergent_analysis(
        self,
        extraction_step: WorkflowStep,
        clustering_step: WorkflowStep,
        scoring_step: WorkflowStep,
        scoring_criteria: List[str]
    ) -> str:
        """
        Synthesize convergent analysis results into formatted output.

        Args:
            extraction_step: WorkflowStep from idea extraction
            clustering_step: WorkflowStep from clustering
            scoring_step: WorkflowStep from scoring
            scoring_criteria: List of scoring criteria used

        Returns:
            Formatted synthesis with clusters, scores, and recommendations
        """
        ideas = extraction_step.metadata.get('extracted_ideas', [])
        clusters = clustering_step.metadata.get('clusters', [])
        scored_clusters = scoring_step.metadata.get('scored_clusters', [])

        # Create lookup for scores by cluster_id
        score_lookup = {sc['cluster_id']: sc for sc in scored_clusters}

        # Sort clusters by overall score (descending)
        sorted_scored = sorted(
            scored_clusters,
            key=lambda x: x.get('overall_score', 0),
            reverse=True
        )

        synthesis_parts = [
            "# Convergent Idea Analysis\n",
            f"\nAnalyzed {len(ideas)} ideas from brainstorming session\n",
            f"Organized into {len(clusters)} thematic clusters\n",
            f"Scored on {len(scoring_criteria)} criteria: {', '.join(scoring_criteria)}\n",
            "\n---\n\n"
        ]

        # High Priority Clusters
        high_priority = [sc for sc in sorted_scored if sc.get('overall_score', 0) >= 4.0]
        if high_priority:
            synthesis_parts.append("## High Priority Clusters\n\n")
            for sc in high_priority:
                synthesis_parts.append(self._format_scored_cluster(sc, clusters, ideas))

        # Medium Priority Clusters
        medium_priority = [
            sc for sc in sorted_scored
            if 3.0 <= sc.get('overall_score', 0) < 4.0
        ]
        if medium_priority:
            synthesis_parts.append("\n## Medium Priority Clusters\n\n")
            for sc in medium_priority:
                synthesis_parts.append(self._format_scored_cluster(sc, clusters, ideas))

        # Low Priority Clusters
        low_priority = [sc for sc in sorted_scored if sc.get('overall_score', 0) < 3.0]
        if low_priority:
            synthesis_parts.append("\n## Low Priority Clusters\n\n")
            for sc in low_priority:
                synthesis_parts.append(self._format_scored_cluster(sc, clusters, ideas))

        # Add recommendations
        synthesis_parts.append("\n---\n\n## Recommended Next Steps\n\n")

        if high_priority:
            synthesis_parts.append(
                f"1. **Prioritize High-Scoring Themes**: Focus on the {len(high_priority)} "
                f"high-priority cluster(s) for immediate action\n\n"
            )

        synthesis_parts.append(
            "2. **Develop Detailed Plans**: For top-ranked ideas, create implementation plans\n\n"
        )

        synthesis_parts.append(
            "3. **Prototype and Test**: Start with feasible, high-impact ideas for quick wins\n\n"
        )

        if medium_priority:
            synthesis_parts.append(
                f"4. **Consider Medium Priority**: Evaluate the {len(medium_priority)} medium-priority "
                f"cluster(s) for longer-term planning\n\n"
            )

        return "".join(synthesis_parts)

    def _format_scored_cluster(
        self,
        scored_cluster: Dict[str, Any],
        all_clusters: List[Dict[str, Any]],
        all_ideas: List[Dict[str, Any]]
    ) -> str:
        """
        Format a single scored cluster for display.

        Args:
            scored_cluster: Scored cluster dictionary
            all_clusters: All clusters (for looking up details)
            all_ideas: All ideas (for looking up details)

        Returns:
            Formatted cluster section
        """
        cluster_id = scored_cluster['cluster_id']
        theme = scored_cluster['theme']
        overall_score = scored_cluster.get('overall_score', 0)
        recommendation = scored_cluster.get('recommendation', 'Unknown')

        # Find full cluster details
        cluster_details = next(
            (c for c in all_clusters if c['id'] == cluster_id),
            {}
        )

        parts = [
            f"### {theme}\n\n",
            f"**Overall Score**: {overall_score:.1f}/5.0 ({recommendation})\n\n"
        ]

        # Add individual scores
        parts.append("**Scores**:\n")
        for criterion, score_data in scored_cluster.get('scores', {}).items():
            score_val = score_data.get('score', 0)
            explanation = score_data.get('explanation', '')
            parts.append(f"- {criterion.title()}: {score_val}/5 - {explanation}\n")

        parts.append("\n")

        # Add cluster description
        if cluster_details.get('description'):
            parts.append(f"**Description**: {cluster_details['description']}\n\n")

        # Add ideas in cluster
        cluster_ideas = cluster_details.get('ideas', [])
        if cluster_ideas:
            parts.append(f"**Ideas in this cluster** ({len(cluster_ideas)}):\n")
            for ci in cluster_ideas[:5]:  # Limit to first 5 for brevity
                idea_id = ci.get('idea_id', '')
                reason = ci.get('reason', '')

                # Find idea details
                idea = next(
                    (i for i in all_ideas if i['id'] == idea_id),
                    {}
                )

                if idea:
                    parts.append(f"- **{idea.get('label', idea_id)}**: {reason}\n")

            if len(cluster_ideas) > 5:
                parts.append(f"  ... and {len(cluster_ideas) - 5} more\n")

        parts.append("\n")

        return "".join(parts)

    async def run_complete_ideation(
        self,
        prompt: str,
        provider_map: Dict[str, ModelProvider],
        perspectives: Optional[List[str]] = None,
        scoring_criteria: Optional[List[str]] = None,
        num_clusters: Optional[int] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute complete two-step ideation workflow.

        Combines divergent brainstorming with convergent analysis for
        comprehensive ideation from multiple perspectives with scoring.

        This is a convenience method that runs:
        1. Divergent Step: Parallel brainstorming from multiple perspectives
        2. Convergent Step: Extraction, clustering, and scoring of ideas

        Args:
            prompt: The topic or problem to ideate on
            provider_map: Dictionary mapping model names to provider instances
            perspectives: List of perspectives for brainstorming (default: ['practical', 'innovative', 'user-focused'])
            scoring_criteria: Criteria for scoring ideas (default: ['feasibility', 'impact', 'novelty'])
            num_clusters: Target number of clusters (default: auto-determine 3-7)
            **kwargs: Additional parameters passed to both steps

        Returns:
            WorkflowResult containing:
                - synthesis: Final convergent analysis with prioritized clusters
                - steps: All steps from both divergent and convergent phases
                - metadata: Combined metadata from both workflow phases

        Raises:
            ValueError: If prompt is empty or provider_map is empty
            Exception: If either workflow phase fails

        Example:
            >>> from model_chorus.providers import ClaudeProvider, GeminiProvider
            >>> from model_chorus.workflows import IdeateWorkflow
            >>>
            >>> # Create providers
            >>> claude = ClaudeProvider()
            >>> gemini = GeminiProvider()
            >>> provider_map = {"claude": claude, "gemini": gemini}
            >>>
            >>> # Create workflow
            >>> workflow = IdeateWorkflow(claude)  # One provider for convergent phase
            >>>
            >>> # Run complete ideation
            >>> result = await workflow.run_complete_ideation(
            ...     prompt="How can we improve our API documentation?",
            ...     provider_map=provider_map,
            ...     perspectives=['practical', 'innovative', 'user-focused'],
            ...     scoring_criteria=['feasibility', 'impact', 'user_value']
            ... )
            >>>
            >>> # Access prioritized results
            >>> print(result.synthesis)  # Shows high/medium/low priority clusters
            >>> for step in result.steps:
            ...     print(f"{step.title}: {step.metadata}")
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not provider_map:
            raise ValueError("Provider map cannot be empty")

        logger.info("Starting complete ideation workflow (divergent + convergent)")

        # Emit workflow start
        emit_workflow_start("ideate", "20-45s")

        # Step 1: Divergent brainstorming with multiple perspectives
        logger.info("Phase 1: Divergent brainstorming")
        brainstorming_result = await self.run_parallel_brainstorming(
            prompt=prompt,
            provider_map=provider_map,
            perspectives=perspectives,
            **kwargs
        )

        # Step 2: Convergent analysis
        logger.info("Phase 2: Convergent analysis")
        convergent_result = await self.run_convergent_analysis(
            brainstorming_result=brainstorming_result,
            scoring_criteria=scoring_criteria,
            num_clusters=num_clusters,
            **kwargs
        )

        # Combine steps from both phases
        all_steps = brainstorming_result.steps + convergent_result.steps

        # Create combined metadata
        combined_metadata = {
            'workflow': 'ideate-complete',
            'divergent_phase': brainstorming_result.metadata,
            'convergent_phase': convergent_result.metadata,
            'total_steps': len(all_steps),
            'perspectives': perspectives or ['practical', 'innovative', 'user-focused'],
            'scoring_criteria': scoring_criteria or ['feasibility', 'impact', 'novelty']
        }

        # Create final result with convergent analysis as synthesis
        result = WorkflowResult(success=True, 
            synthesis=convergent_result.synthesis,
            steps=all_steps,
            metadata=combined_metadata
        )

        logger.info("Complete ideation workflow finished successfully")
        return result

    def run_interactive_selection(
        self,
        convergent_result: WorkflowResult,
        max_selections: Optional[int] = None
    ) -> WorkflowResult:
        """
        Run interactive CLI selection of idea clusters.

        Presents scored clusters to the user and allows interactive selection
        of which clusters to pursue. This is the third step in the complete
        IDEATE workflow pattern.

        Args:
            convergent_result: WorkflowResult from run_convergent_analysis()
            max_selections: Maximum number of clusters user can select (default: unlimited)

        Returns:
            WorkflowResult containing:
                - synthesis: Summary of selected clusters
                - steps: Original convergent steps + selection step
                - metadata: Selected cluster IDs and selection metadata

        Raises:
            ValueError: If convergent_result is invalid
            IOError: If running in non-interactive environment

        Example:
            >>> # After running convergent analysis
            >>> convergent_result = await workflow.run_convergent_analysis(brainstorming_result)
            >>>
            >>> # Interactively select clusters
            >>> selection_result = workflow.run_interactive_selection(convergent_result)
            >>>
            >>> # Access selected clusters
            >>> selected = selection_result.metadata['selected_clusters']
            >>> print(f"Selected {len(selected)} clusters for implementation")
        """
        if not convergent_result or not convergent_result.steps:
            raise ValueError("Convergent result must have steps")

        # Get scored clusters from convergent analysis
        scoring_step = None
        for step in convergent_result.steps:
            if step.metadata.get('title') == 'Idea Scoring':
                scoring_step = step
                break

        if not scoring_step:
            raise ValueError("No scoring step found in convergent result")

        scored_clusters = scoring_step.metadata.get('scored_clusters', [])

        if not scored_clusters:
            raise ValueError("No scored clusters found")

        logger.info(f"Starting interactive selection for {len(scored_clusters)} clusters")

        # Display clusters and get user selection
        selected_cluster_ids = self._display_and_select_clusters(
            scored_clusters,
            max_selections=max_selections
        )

        # Get full cluster details for selected IDs
        clustering_step = None
        for step in convergent_result.steps:
            if step.metadata.get('title') == 'Idea Clustering':
                clustering_step = step
                break

        all_clusters = clustering_step.metadata.get('clusters', []) if clustering_step else []

        # Extract selected clusters with full details
        selected_clusters = []
        for cluster_id in selected_cluster_ids:
            # Find scored cluster
            scored = next(
                (sc for sc in scored_clusters if sc['cluster_id'] == cluster_id),
                None
            )
            # Find full cluster details
            cluster = next(
                (c for c in all_clusters if c['id'] == cluster_id),
                {}
            )

            if scored:
                selected_clusters.append({
                    'cluster_id': cluster_id,
                    'theme': scored.get('theme', ''),
                    'overall_score': scored.get('overall_score', 0),
                    'recommendation': scored.get('recommendation', ''),
                    'scores': scored.get('scores', {}),
                    'ideas': cluster.get('ideas', [])
                })

        # Create synthesis
        synthesis = self._synthesize_selection(selected_clusters, scored_clusters)

        # Create selection step
        selection_step = WorkflowStep(
            step_number=len(convergent_result.steps) + 1,
            content=synthesis,
            metadata={
                'title': 'Interactive Selection',
                'num_selected': len(selected_clusters),
                'num_available': len(scored_clusters),
                'selected_cluster_ids': selected_cluster_ids,
                'selected_clusters': selected_clusters
            }
        )

        # Create result combining convergent steps + selection
        result = WorkflowResult(
            success=True,
            synthesis=synthesis,
            steps=convergent_result.steps + [selection_step],
            metadata={
                'workflow': 'ideate-selection',
                'selected_cluster_ids': selected_cluster_ids,
                'selected_clusters': selected_clusters,
                'convergent_metadata': convergent_result.metadata
            }
        )

        logger.info(f"Interactive selection completed. Selected {len(selected_clusters)} clusters")
        return result

    def _display_and_select_clusters(
        self,
        scored_clusters: List[Dict[str, Any]],
        max_selections: Optional[int] = None
    ) -> List[str]:
        """
        Display scored clusters and prompt user for selection.

        Args:
            scored_clusters: List of scored cluster dictionaries
            max_selections: Maximum selections allowed (None = unlimited)

        Returns:
            List of selected cluster IDs
        """
        # Sort by score (descending)
        sorted_clusters = sorted(
            scored_clusters,
            key=lambda x: x.get('overall_score', 0),
            reverse=True
        )

        print("\n" + "=" * 80)
        print("IDEATE: CLUSTER SELECTION")
        print("=" * 80 + "\n")

        print(f"Found {len(sorted_clusters)} idea clusters from brainstorming session.\n")

        # Display clusters with numbers
        for i, cluster in enumerate(sorted_clusters, 1):
            cluster_id = cluster.get('cluster_id', '')
            theme = cluster.get('theme', 'Untitled')
            score = cluster.get('overall_score', 0)
            rec = cluster.get('recommendation', 'Unknown')

            # Color-code by priority
            priority_marker = "🔴" if score >= 4.0 else "🟡" if score >= 3.0 else "🟢"

            print(f"{i}. {priority_marker} {theme}")
            print(f"   Score: {score:.1f}/5.0 ({rec})")

            # Show top scoring criteria
            scores = cluster.get('scores', {})
            top_scores = sorted(
                scores.items(),
                key=lambda x: x[1].get('score', 0),
                reverse=True
            )[:2]

            if top_scores:
                criteria_str = ", ".join([
                    f"{crit.title()}: {data.get('score', 0)}/5"
                    for crit, data in top_scores
                ])
                print(f"   Strengths: {criteria_str}")

            print()

        # Prompt for selection
        print("-" * 80)
        print("Select clusters to pursue (enter numbers separated by commas)")
        print("Examples: '1,3,5' or '1-3' or 'all' or 'none'")

        if max_selections:
            print(f"(Maximum {max_selections} selections allowed)")

        print("-" * 80 + "\n")

        # Get user input
        while True:
            try:
                user_input = input("Your selection: ").strip()

                if not user_input:
                    print("⚠️  Please enter a selection\n")
                    continue

                # Parse selection
                selected_indices = self._parse_selection_input(
                    user_input,
                    total_count=len(sorted_clusters),
                    max_selections=max_selections
                )

                if selected_indices is None:
                    continue  # Invalid input, prompt again

                # Convert indices to cluster IDs
                selected_ids = [
                    sorted_clusters[idx - 1].get('cluster_id', '')
                    for idx in selected_indices
                ]

                print(f"\n✅ Selected {len(selected_ids)} cluster(s)\n")
                return selected_ids

            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Selection cancelled by user")
                return []

    def _parse_selection_input(
        self,
        user_input: str,
        total_count: int,
        max_selections: Optional[int] = None
    ) -> Optional[List[int]]:
        """
        Parse user selection input.

        Supports formats: "1,3,5", "1-3", "all", "none"

        Args:
            user_input: Raw user input string
            total_count: Total number of items available
            max_selections: Maximum selections allowed

        Returns:
            List of selected indices (1-based), or None if invalid
        """
        user_input = user_input.strip().lower()

        # Handle special cases
        if user_input == 'none':
            return []

        if user_input == 'all':
            selections = list(range(1, total_count + 1))
            if max_selections and len(selections) > max_selections:
                print(f"⚠️  'all' exceeds maximum of {max_selections} selections\n")
                return None
            return selections

        # Parse comma-separated or range selections
        selections = set()

        try:
            parts = user_input.split(',')

            for part in parts:
                part = part.strip()

                # Range format: "1-3"
                if '-' in part:
                    range_parts = part.split('-')
                    if len(range_parts) != 2:
                        print(f"⚠️  Invalid range format: '{part}'\n")
                        return None

                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())

                    if start < 1 or end > total_count or start > end:
                        print(f"⚠️  Invalid range: {start}-{end} (valid: 1-{total_count})\n")
                        return None

                    selections.update(range(start, end + 1))

                # Single number
                else:
                    num = int(part)

                    if num < 1 or num > total_count:
                        print(f"⚠️  Invalid selection: {num} (valid: 1-{total_count})\n")
                        return None

                    selections.add(num)

        except ValueError as e:
            print(f"⚠️  Invalid input format: {user_input}\n")
            print("   Use numbers (1,2,3), ranges (1-3), 'all', or 'none'\n")
            return None

        # Check max selections
        if max_selections and len(selections) > max_selections:
            print(f"⚠️  Too many selections: {len(selections)} (maximum: {max_selections})\n")
            return None

        return sorted(list(selections))

    def _synthesize_selection(
        self,
        selected_clusters: List[Dict[str, Any]],
        all_clusters: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize selection results into summary.

        Args:
            selected_clusters: Clusters selected by user
            all_clusters: All available clusters

        Returns:
            Formatted synthesis text
        """
        if not selected_clusters:
            return "# Selection Summary\n\nNo clusters selected for implementation."

        synthesis_parts = [
            "# Selected Ideas for Implementation\n",
            f"\nSelected {len(selected_clusters)} of {len(all_clusters)} clusters:\n\n"
        ]

        # Sort by score
        sorted_selected = sorted(
            selected_clusters,
            key=lambda x: x.get('overall_score', 0),
            reverse=True
        )

        for cluster in sorted_selected:
            theme = cluster.get('theme', 'Untitled')
            score = cluster.get('overall_score', 0)
            rec = cluster.get('recommendation', '')

            synthesis_parts.append(f"## {theme}\n\n")
            synthesis_parts.append(f"**Score**: {score:.1f}/5.0 ({rec})\n\n")

            # Add scores
            scores = cluster.get('scores', {})
            if scores:
                synthesis_parts.append("**Evaluation**:\n")
                for criterion, data in scores.items():
                    val = data.get('score', 0)
                    explanation = data.get('explanation', '')
                    synthesis_parts.append(f"- {criterion.title()}: {val}/5 - {explanation}\n")
                synthesis_parts.append("\n")

            # Add ideas
            ideas = cluster.get('ideas', [])
            if ideas:
                synthesis_parts.append(f"**Ideas** ({len(ideas)}):\n")
                for idea_ref in ideas[:3]:  # Show top 3
                    idea_id = idea_ref.get('idea_id', '')
                    reason = idea_ref.get('reason', '')
                    synthesis_parts.append(f"- {idea_id}: {reason}\n")

                if len(ideas) > 3:
                    synthesis_parts.append(f"  ... and {len(ideas) - 3} more\n")

                synthesis_parts.append("\n")

        synthesis_parts.append("---\n\n## Next Steps\n\n")
        synthesis_parts.append("1. Develop detailed implementation plans for selected clusters\n")
        synthesis_parts.append("2. Prioritize ideas within each cluster\n")
        synthesis_parts.append("3. Estimate resources and timelines\n")
        synthesis_parts.append("4. Begin prototyping highest-priority ideas\n")

        return "".join(synthesis_parts)

    async def run_elaboration(
        self,
        selection_result: WorkflowResult,
        **kwargs
    ) -> WorkflowResult:
        """
        Elaborate selected clusters into detailed, actionable outlines.

        Takes selected idea clusters from run_interactive_selection and generates
        comprehensive outlines with implementation details, steps, considerations,
        and next actions for each selected cluster.

        This is the fourth and final step in the complete IDEATE workflow pattern.

        Args:
            selection_result: WorkflowResult from run_interactive_selection()
            **kwargs: Additional parameters for provider (temperature, max_tokens)

        Returns:
            WorkflowResult containing:
                - synthesis: Formatted collection of detailed outlines
                - steps: Elaboration steps for each selected cluster
                - metadata: Elaboration parameters and statistics

        Raises:
            ValueError: If selection_result is invalid or has no selected clusters
            Exception: If elaboration fails

        Example:
            >>> # After interactive selection
            >>> selection_result = workflow.run_interactive_selection(convergent_result)
            >>>
            >>> # Elaborate selected clusters into detailed outlines
            >>> elaboration_result = await workflow.run_elaboration(selection_result)
            >>>
            >>> # Access detailed outlines
            >>> print(elaboration_result.synthesis)
            >>> for step in elaboration_result.steps:
            ...     print(f"{step.metadata['theme']}: {len(step.metadata['outline_sections'])} sections")
        """
        if not selection_result or not selection_result.metadata:
            raise ValueError("Selection result must have metadata")

        selected_clusters = selection_result.metadata.get('selected_clusters', [])

        if not selected_clusters:
            raise ValueError("No selected clusters found in selection result")

        logger.info(f"Starting elaboration for {len(selected_clusters)} selected clusters")

        # Elaborate each selected cluster
        elaboration_steps = []
        for i, cluster in enumerate(selected_clusters, 1):
            logger.info(f"Elaborating cluster {i}/{len(selected_clusters)}: {cluster.get('theme', 'Untitled')}")

            elaboration_step = await self._elaborate_cluster(
                cluster=cluster,
                step_number=i,
                **kwargs
            )
            elaboration_steps.append(elaboration_step)

        # Create synthesis combining all elaborated outlines
        synthesis = self._synthesize_elaborations(
            elaboration_steps=elaboration_steps,
            selected_clusters=selected_clusters
        )

        # Create workflow result
        result = WorkflowResult(success=True, 
            synthesis=synthesis,
            steps=elaboration_steps,
            metadata={
                'workflow': 'ideate-elaboration',
                'num_elaborated': len(elaboration_steps),
                'selection_metadata': selection_result.metadata
            }
        )

        logger.info(f"Elaboration completed for {len(elaboration_steps)} clusters")
        return result

    async def _elaborate_cluster(
        self,
        cluster: Dict[str, Any],
        step_number: int,
        **kwargs
    ) -> WorkflowStep:
        """
        Elaborate a single cluster into a detailed outline.

        Args:
            cluster: Selected cluster dictionary with theme, scores, ideas
            step_number: Step number for this elaboration
            **kwargs: Additional parameters for provider

        Returns:
            WorkflowStep containing detailed outline with metadata
        """
        theme = cluster.get('theme', 'Untitled')
        logger.info(f"Elaborating cluster: {theme}")

        # Create elaboration prompt
        elaboration_prompt = self._create_elaboration_prompt(cluster)

        # Set moderate temperature for creative but structured elaboration
        temperature = kwargs.get('temperature', 0.6)

        # Create generation request
        request = GenerationRequest(
            prompt=elaboration_prompt,
            system_prompt=self._get_elaboration_system_prompt(),
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 3000)
        )

        try:
            # Generate elaborated outline with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed for cluster elaboration '{theme}': {', '.join(failed)}")

            # Parse outline sections from response
            outline_sections = self._parse_outline_sections(response.content)

            # Create workflow step
            step = WorkflowStep(
                step_number=step_number,
                content=response.content,
                model=self.provider.provider_name,
                metadata={
                    'title': f'Elaboration: {theme}',
                    'theme': theme,
                    'cluster_id': cluster.get('cluster_id', ''),
                    'overall_score': cluster.get('overall_score', 0),
                    'num_ideas': len(cluster.get('ideas', [])),
                    'outline_sections': outline_sections,
                    'temperature': temperature
                }
            )

            logger.info(f"Elaborated '{theme}' into {len(outline_sections)} sections")
            return step

        except Exception as e:
            logger.error(f"Cluster elaboration failed for '{theme}': {e}")
            raise

    def _create_elaboration_prompt(self, cluster: Dict[str, Any]) -> str:
        """
        Create prompt for elaborating a cluster into a detailed outline.

        Args:
            cluster: Cluster dictionary with theme, scores, ideas

        Returns:
            Formatted elaboration prompt
        """
        theme = cluster.get('theme', 'Untitled')
        description = cluster.get('description', '')
        scores = cluster.get('scores', {})
        ideas = cluster.get('ideas', [])

        prompt_parts = [
            f"You are creating a detailed, actionable outline for implementing the following idea cluster:\n\n",
            f"**Theme**: {theme}\n\n"
        ]

        if description:
            prompt_parts.append(f"**Description**: {description}\n\n")

        # Add evaluation scores
        if scores:
            prompt_parts.append("**Evaluation**:\n")
            for criterion, data in scores.items():
                score_val = data.get('score', 0)
                explanation = data.get('explanation', '')
                prompt_parts.append(f"- {criterion.title()}: {score_val}/5 - {explanation}\n")
            prompt_parts.append("\n")

        # Add related ideas
        if ideas:
            prompt_parts.append(f"**Related Ideas** ({len(ideas)}):\n")
            for idea_ref in ideas:
                idea_id = idea_ref.get('idea_id', '')
                reason = idea_ref.get('reason', '')
                prompt_parts.append(f"- {idea_id}: {reason}\n")
            prompt_parts.append("\n")

        prompt_parts.append(
            "---\n\n"
            "Create a comprehensive, actionable outline for implementing this idea cluster. "
            "Your outline should include:\n\n"
            "1. **Overview**: Brief summary of what this entails (2-3 sentences)\n"
            "2. **Goals & Objectives**: What this aims to achieve (3-5 bullet points)\n"
            "3. **Implementation Approach**: High-level strategy and methodology\n"
            "4. **Detailed Steps**: Concrete, sequential steps to implement (numbered list)\n"
            "5. **Key Considerations**: Important factors to keep in mind (technical, UX, business)\n"
            "6. **Success Metrics**: How to measure success\n"
            "7. **Potential Challenges**: Anticipated obstacles and mitigation strategies\n"
            "8. **Next Actions**: Immediate next steps to get started\n\n"
            "Make the outline specific, practical, and actionable. Include sufficient detail "
            "for someone to understand how to proceed with implementation."
        )

        return "".join(prompt_parts)

    def _get_elaboration_system_prompt(self) -> str:
        """
        Get system prompt for elaboration task.

        Returns:
            System prompt for detailed outline generation
        """
        return """You are an expert at taking creative ideas and developing them into detailed, actionable plans.

Your task is to create comprehensive outlines that transform high-level concepts into concrete implementation roadmaps.

Guidelines:
1. **Be specific**: Provide concrete details, not vague generalities
2. **Be practical**: Focus on actionable steps and realistic approaches
3. **Be thorough**: Cover all important aspects (technical, user, business)
4. **Be structured**: Organize information logically with clear sections
5. **Anticipate challenges**: Identify potential obstacles proactively
6. **Define success**: Include measurable outcomes and metrics

Your outline should give the reader a clear understanding of:
- What needs to be built/implemented
- How to approach the implementation
- What to watch out for
- How to know when it's successful

Write in a professional, clear style. Use markdown formatting for structure."""

    def _parse_outline_sections(self, elaboration_content: str) -> List[Dict[str, str]]:
        """
        Parse outline sections from elaboration response.

        Args:
            elaboration_content: LLM response with detailed outline

        Returns:
            List of section dictionaries with titles and content
        """
        sections = []
        lines = elaboration_content.split('\n')
        current_section = None

        for line in lines:
            line_stripped = line.strip()

            # Check for main section headers (## Header or **number. Header**)
            if line_stripped.startswith('##') and not line_stripped.startswith('###'):
                # Save previous section
                if current_section and current_section.get('content'):
                    sections.append(current_section)

                # Start new section
                section_title = line_stripped.lstrip('#').strip()
                current_section = {
                    'title': section_title,
                    'content': ''
                }

            elif line_stripped.startswith('**') and any(char.isdigit() for char in line_stripped[:10]):
                # Handle numbered sections like "**1. Overview**"
                if ']' not in line_stripped:  # Ignore IDEA-X references
                    # Save previous section
                    if current_section and current_section.get('content'):
                        sections.append(current_section)

                    # Extract title from **1. Title**
                    try:
                        title_part = line_stripped.split('**')[1]
                        # Remove leading number and period
                        if '.' in title_part:
                            title_part = title_part.split('.', 1)[1].strip()

                        current_section = {
                            'title': title_part,
                            'content': ''
                        }
                    except (IndexError, ValueError):
                        # If parsing fails, just accumulate content
                        if current_section is not None:
                            current_section['content'] += line + '\n'

            elif current_section is not None:
                # Accumulate content for current section
                current_section['content'] += line + '\n'

        # Don't forget the last section
        if current_section and current_section.get('content'):
            sections.append(current_section)

        logger.info(f"Parsed {len(sections)} outline sections")
        return sections

    def _synthesize_elaborations(
        self,
        elaboration_steps: List[WorkflowStep],
        selected_clusters: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize all elaborated outlines into formatted collection.

        Args:
            elaboration_steps: List of elaboration WorkflowSteps
            selected_clusters: Original selected clusters

        Returns:
            Formatted synthesis combining all outlines
        """
        synthesis_parts = [
            "# Detailed Implementation Outlines\n",
            f"\nElaborated {len(elaboration_steps)} selected idea cluster(s) into actionable outlines:\n\n",
            "---\n\n"
        ]

        for i, step in enumerate(elaboration_steps, 1):
            theme = step.metadata.get('theme', 'Untitled')
            score = step.metadata.get('overall_score', 0)
            num_sections = len(step.metadata.get('outline_sections', []))

            synthesis_parts.append(f"## Outline {i}: {theme}\n\n")
            synthesis_parts.append(f"**Score**: {score:.1f}/5.0 | **Sections**: {num_sections}\n\n")
            synthesis_parts.append(step.content)
            synthesis_parts.append("\n\n---\n\n")

        # Add summary and next steps
        synthesis_parts.append("## Summary\n\n")
        synthesis_parts.append(
            f"Created {len(elaboration_steps)} detailed implementation outline(s) from selected idea clusters. "
            f"Each outline includes:\n\n"
            "- Overview and objectives\n"
            "- Implementation approach and steps\n"
            "- Key considerations and challenges\n"
            "- Success metrics and next actions\n\n"
        )

        synthesis_parts.append("## Recommended Workflow\n\n")
        synthesis_parts.append(
            "1. **Review Outlines**: Read through all elaborated outlines above\n"
            "2. **Prioritize**: Choose which outline(s) to implement first based on scores and feasibility\n"
            "3. **Plan Resources**: Estimate time, team, and budget requirements\n"
            "4. **Create Tasks**: Break down detailed steps into trackable work items\n"
            "5. **Begin Implementation**: Start with first actionable steps from chosen outline\n"
        )

        return "".join(synthesis_parts)

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
