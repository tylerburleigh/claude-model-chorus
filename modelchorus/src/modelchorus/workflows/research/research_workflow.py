"""
Research workflow for systematic information gathering and analysis.

This module implements the ResearchWorkflow which provides systematic research
through structured question formulation, source gathering, and synthesis.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from ...core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from ...core.conversation import ConversationMemory
from ...core.registry import WorkflowRegistry
from ...providers import ModelProvider, GenerationRequest, GenerationResponse
from ...core.models import ConversationMessage

logger = logging.getLogger(__name__)


@WorkflowRegistry.register("research")
class ResearchWorkflow(BaseWorkflow):
    """
    Systematic research workflow for information gathering and analysis.

    This workflow implements structured research through question formulation,
    iterative source gathering, critical analysis, and comprehensive synthesis.

    Architecture:
    - Single-model systematic research with controlled creativity
    - Multi-phase research process (questions, sources, analysis, synthesis)
    - Conversation threading for iterative refinement
    - Focus on depth, accuracy, and comprehensive coverage

    Key Features:
    - Question formulation and refinement
    - Source identification and evaluation
    - Critical analysis and synthesis
    - Conversation threading via continuation_id
    - Inherits conversation support from BaseWorkflow
    - Structured research methodology

    The ResearchWorkflow is ideal for:
    - In-depth topic exploration
    - Literature review and source gathering
    - Competitive analysis
    - Technical research and investigation
    - Background research for decision-making

    Workflow Pattern:
    1. **Question Formulation**: Define research questions and scope
    2. **Source Gathering**: Identify relevant sources and information
    3. **Analysis**: Critically evaluate and analyze findings
    4. **Synthesis**: Combine insights into comprehensive understanding

    Example:
        >>> from modelchorus.providers import ClaudeProvider
        >>> from modelchorus.workflows import ResearchWorkflow
        >>> from modelchorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = ResearchWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # Conduct research
        >>> result = await workflow.run(
        ...     "Research emerging trends in AI orchestration"
        ... )
        >>> print(result.synthesis)
        >>>
        >>> # Deep dive on specific aspect
        >>> result2 = await workflow.run(
        ...     "Focus on multi-model consensus approaches",
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
        Initialize ResearchWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for research
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn conversations

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider is required for ResearchWorkflow")

        super().__init__(
            name="Research",
            description="Systematic research workflow for information gathering and analysis",
            config=config,
            conversation_memory=conversation_memory
        )

        self.provider = provider
        self.conversation_memory = conversation_memory

        # Initialize source registry for tracking research sources
        self.source_registry: List[Dict[str, Any]] = []

        # Research-specific default configuration
        self.default_config = {
            'temperature': 0.5,  # Balanced between creativity and accuracy
            'max_tokens': 4000,  # Allow for comprehensive responses
            'research_depth': 'thorough',  # shallow, moderate, thorough, comprehensive
            'source_validation': True,  # Validate source credibility
            'citation_style': 'informal',  # informal, academic, technical
        }

        # Merge with provided config
        if config:
            self.default_config.update(config)

        logger.info(f"Initialized ResearchWorkflow with provider: {provider.provider_name}")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate workflow-specific configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        valid_depths = ['shallow', 'moderate', 'thorough', 'comprehensive']
        valid_citation_styles = ['informal', 'academic', 'technical']

        if 'research_depth' in config:
            if config['research_depth'] not in valid_depths:
                logger.warning(f"Invalid research_depth: {config['research_depth']}")
                return False

        if 'citation_style' in config:
            if config['citation_style'] not in valid_citation_styles:
                logger.warning(f"Invalid citation_style: {config['citation_style']}")
                return False

        if 'temperature' in config:
            temp = config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 1:
                logger.warning(f"Invalid temperature: {temp}")
                return False

        return True

    def get_provider(self) -> ModelProvider:
        """
        Get the model provider for this workflow.

        Returns:
            ModelProvider instance
        """
        return self.provider

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute the research workflow.

        Args:
            prompt: Research topic or question
            continuation_id: Optional thread ID for continuing research
            **kwargs: Additional parameters for research customization

        Returns:
            WorkflowResult containing research findings and analysis

        Raises:
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("Research prompt cannot be empty")

        logger.info(f"Starting research workflow: {prompt[:100]}...")

        # Create or get conversation thread
        thread_id = continuation_id or str(uuid.uuid4())

        # Merge kwargs with default config
        config = {**self.default_config, **kwargs}

        # Validate merged configuration
        if not self.validate_config(config):
            raise ValueError("Invalid configuration for research workflow")

        # Create workflow steps list
        steps: List[WorkflowStep] = []

        # Step 1: Formulate research questions
        questions_step = await self._formulate_questions(prompt, thread_id, config)
        steps.append(questions_step)

        # Create workflow result
        result = WorkflowResult(
            steps=steps,
            synthesis=questions_step.content,  # Placeholder - will be updated
            metadata={
                'thread_id': thread_id,
                'research_topic': prompt,
                'research_depth': config.get('research_depth', 'thorough'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model': self.provider.provider_name,
            }
        )

        logger.info(f"Research workflow completed: {len(steps)} steps")
        return result

    async def _formulate_questions(
        self,
        prompt: str,
        thread_id: str,
        config: Dict[str, Any]
    ) -> WorkflowStep:
        """
        Formulate research questions based on the prompt.

        Args:
            prompt: Research topic or question
            thread_id: Conversation thread ID
            config: Configuration dictionary

        Returns:
            WorkflowStep containing formulated questions
        """
        logger.info("Formulating research questions")

        # Create system prompt for question formulation
        system_prompt = self._get_question_formulation_system_prompt()

        # Create research question prompt
        question_prompt = self._create_question_prompt(prompt, config)

        # Create generation request
        request = GenerationRequest(
            prompt=question_prompt,
            system_prompt=system_prompt,
            temperature=config.get('temperature', 0.5),
            max_tokens=config.get('max_tokens', 4000)
        )

        try:
            # Generate research questions
            response: GenerationResponse = await self.provider.generate(request)

            # Add to conversation memory if available
            if self.conversation_memory:
                self.conversation_memory.add_message(
                    thread_id=thread_id,
                    message=ConversationMessage(
                        role="user",
                        content=prompt,
                        timestamp=datetime.now(timezone.utc)
                    )
                )
                self.conversation_memory.add_message(
                    thread_id=thread_id,
                    message=ConversationMessage(
                        role="assistant",
                        content=response.content,
                        timestamp=datetime.now(timezone.utc),
                        model=self.provider.provider_name
                    )
                )

            # Create workflow step
            step = WorkflowStep(
                step_number=1,
                content=response.content,
                model=self.provider.provider_name,
                metadata={
                    'title': 'Research Question Formulation',
                    'research_depth': config.get('research_depth', 'thorough'),
                    'temperature': config.get('temperature', 0.5)
                }
            )

            logger.info("Research questions formulated successfully")
            return step

        except Exception as e:
            logger.error(f"Question formulation failed: {e}")
            raise

    def _get_question_formulation_system_prompt(self) -> str:
        """
        Get system prompt for research question formulation.

        Returns:
            System prompt for question formulation task
        """
        return """You are an expert research assistant specializing in formulating clear, focused research questions.

Your task is to:
1. Understand the research topic and its context
2. Identify key areas that need investigation
3. Formulate specific, answerable research questions
4. Organize questions by priority and theme

Guidelines:
- Create focused, specific questions rather than vague inquiries
- Cover multiple aspects of the topic comprehensively
- Prioritize questions by importance and feasibility
- Ensure questions are answerable through research
- Group related questions thematically
- Consider different perspectives and angles

Your output should provide a clear research roadmap."""

    def _create_question_prompt(self, prompt: str, config: Dict[str, Any]) -> str:
        """
        Create prompt for research question formulation.

        Args:
            prompt: Original research topic
            config: Configuration dictionary

        Returns:
            Formatted question formulation prompt
        """
        depth = config.get('research_depth', 'thorough')

        depth_guidance = {
            'shallow': 'Focus on 3-5 high-level questions covering main aspects',
            'moderate': 'Develop 5-8 questions covering key areas in moderate detail',
            'thorough': 'Create 8-12 detailed questions exploring topic comprehensively',
            'comprehensive': 'Generate 12-15+ questions covering all aspects and nuances'
        }

        return f"""Research Topic: {prompt}

Research Depth: {depth}
Guidance: {depth_guidance.get(depth, depth_guidance['thorough'])}

Please formulate research questions for this topic. Your questions should:
1. Cover the essential aspects of the topic
2. Be specific and answerable through research
3. Progress from foundational to advanced understanding
4. Consider multiple perspectives and dimensions

Organize your questions by theme or category, and indicate priority (High/Medium/Low) for each."""

    def ingest_source(
        self,
        title: str,
        url: Optional[str] = None,
        source_type: str = 'article',
        credibility: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a new research source into the registry.

        Args:
            title: Title or description of the source
            url: Optional URL or reference to the source
            source_type: Type of source (article, paper, book, website, etc.)
            credibility: Optional credibility assessment (high, medium, low)
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata

        Returns:
            Source dictionary with assigned ID
        """
        source_id = str(uuid.uuid4())[:8]

        source = {
            'source_id': source_id,
            'title': title,
            'url': url,
            'type': source_type,
            'credibility': credibility or 'unassessed',
            'tags': tags or [],
            'metadata': metadata or {},
            'ingested_at': datetime.now(timezone.utc).isoformat(),
            'validated': False
        }

        # Auto-validate if source_validation is enabled
        if self.default_config.get('source_validation', True):
            source = self._validate_source(source)

        self.source_registry.append(source)
        logger.info(f"Ingested source: {source_id} - {title}")

        return source

    def _validate_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate source credibility and metadata.

        Args:
            source: Source dictionary to validate

        Returns:
            Updated source dictionary with validation results
        """
        # Basic validation checks
        validation_score = 0
        validation_notes = []

        # Check if URL is provided
        if source.get('url'):
            validation_score += 2
            validation_notes.append("URL provided")
        else:
            validation_notes.append("No URL - manual source")

        # Check source type
        credible_types = ['paper', 'book', 'academic', 'official']
        if source.get('type') in credible_types:
            validation_score += 3
            validation_notes.append(f"Credible type: {source['type']}")

        # Determine credibility level
        if validation_score >= 4:
            source['credibility'] = 'high'
        elif validation_score >= 2:
            source['credibility'] = 'medium'
        else:
            source['credibility'] = 'low'

        source['validated'] = True
        source['validation_score'] = validation_score
        source['validation_notes'] = validation_notes

        logger.info(f"Validated source {source['source_id']}: {source['credibility']} credibility")

        return source

    def get_sources_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Retrieve sources by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of sources with the specified tag
        """
        return [
            source for source in self.source_registry
            if tag in source.get('tags', [])
        ]

    def get_sources_by_type(self, source_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve sources by type.

        Args:
            source_type: Source type to filter by

        Returns:
            List of sources of the specified type
        """
        return [
            source for source in self.source_registry
            if source.get('type') == source_type
        ]

    def get_sources_by_credibility(self, credibility: str) -> List[Dict[str, Any]]:
        """
        Retrieve sources by credibility level.

        Args:
            credibility: Credibility level (high, medium, low)

        Returns:
            List of sources with the specified credibility
        """
        return [
            source for source in self.source_registry
            if source.get('credibility') == credibility
        ]

    def get_source_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the source registry.

        Returns:
            Dictionary with registry statistics
        """
        total = len(self.source_registry)

        by_type = {}
        by_credibility = {}
        by_tags = {}

        for source in self.source_registry:
            # Count by type
            source_type = source.get('type', 'unknown')
            by_type[source_type] = by_type.get(source_type, 0) + 1

            # Count by credibility
            cred = source.get('credibility', 'unassessed')
            by_credibility[cred] = by_credibility.get(cred, 0) + 1

            # Count by tags
            for tag in source.get('tags', []):
                by_tags[tag] = by_tags.get(tag, 0) + 1

        return {
            'total_sources': total,
            'by_type': by_type,
            'by_credibility': by_credibility,
            'by_tags': by_tags,
            'validated_count': sum(1 for s in self.source_registry if s.get('validated', False))
        }

    def clear_source_registry(self) -> None:
        """Clear all sources from the registry."""
        self.source_registry = []
        logger.info("Source registry cleared")
