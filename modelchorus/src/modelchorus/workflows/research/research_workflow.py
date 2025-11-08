"""
Research workflow for systematic information gathering and analysis.

This module implements the ResearchWorkflow which provides systematic research
through structured question formulation, source gathering, and synthesis.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from ...core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from ...core.conversation import ConversationMemory
from ...core.registry import WorkflowRegistry
from ...providers import ModelProvider, GenerationRequest, GenerationResponse
from ...core.progress import emit_workflow_start, emit_workflow_complete

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """
    Represents a piece of evidence extracted from a source.

    Attributes:
        evidence_id: Unique identifier for this evidence
        source_id: ID of the source this evidence came from
        content: The extracted evidence content
        relevance_score: Relevance score (0.0-1.0)
        confidence: Confidence in the extraction (low, medium, high)
        category: Category or theme of the evidence
        supporting_quote: Optional direct quote from source
        metadata: Additional metadata
    """
    evidence_id: str
    source_id: str
    content: str
    relevance_score: float = 0.0
    confidence: str = 'medium'
    category: Optional[str] = None
    supporting_quote: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ExtractionResult:
    """
    Result from parallel evidence extraction.

    Attributes:
        source_id: ID of the source processed
        evidence: List of extracted evidence items
        success: Whether extraction succeeded
        error: Optional error message if extraction failed
        duration: Time taken for extraction in seconds
        metadata: Additional metadata
    """
    source_id: str
    evidence: List[Evidence] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Result from fact-checking and validation.

    Attributes:
        evidence_id: ID of the evidence being validated
        is_valid: Whether the evidence passed validation
        confidence_score: Validation confidence score (0.0-1.0)
        validation_notes: List of validation findings
        contradictions: List of contradictions found
        supporting_evidence: IDs of evidence that supports this claim
        refuting_evidence: IDs of evidence that refutes this claim
        metadata: Additional metadata
    """
    evidence_id: str
    is_valid: bool = True
    confidence_score: float = 0.5
    validation_notes: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    refuting_evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
        fallback_providers: Optional[List[ModelProvider]] = None,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize ResearchWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for research
            fallback_providers: Optional list of fallback providers to try if primary fails
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
        self.fallback_providers = fallback_providers or []
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
        skip_provider_check: bool = False,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute the research workflow.

        Args:
            prompt: Research topic or question
            continuation_id: Optional thread ID for continuing research
            skip_provider_check: Skip provider availability check (faster startup)
            **kwargs: Additional parameters for research customization

        Returns:
            WorkflowResult containing research findings and analysis

        Raises:
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("Research prompt cannot be empty")

        logger.info(f"Starting research workflow: {prompt[:100]}...")

        # Emit workflow start
        emit_workflow_start("research", "30-60s")

        # Check provider availability
        if not skip_provider_check:
            has_available, available, unavailable = await self.check_provider_availability(
                self.provider, self.fallback_providers
            )

            if not has_available:
                from ...providers.cli_provider import ProviderUnavailableError
                error_msg = "No providers available for research:\n"
                for name, error in unavailable:
                    error_msg += f"  - {name}: {error}\n"
                raise ProviderUnavailableError(
                    "all",
                    error_msg,
                    [
                        "Check installations: modelchorus list-providers --check",
                        "Install missing providers or update .modelchorusrc"
                    ]
                )

            if unavailable and logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Some providers unavailable: {[n for n, _ in unavailable]}")
                logger.info(f"Will use available providers: {available}")

        # Create or get conversation thread
        if continuation_id:
            thread_id = continuation_id
            # If continuing, check if thread exists, create if it doesn't
            if self.conversation_memory and self.conversation_memory.get_thread(thread_id) is None:
                # Thread doesn't exist, create it with the given ID
                # Since create_thread generates its own ID, we'll just use the continuation_id directly
                # and let add_message handle thread creation if needed
                pass
        else:
            # Create new thread via conversation memory
            if self.conversation_memory:
                thread_id = self.conversation_memory.create_thread(
                    workflow_name=self.name,
                    initial_context={'research_topic': prompt}
                )
            else:
                thread_id = str(uuid.uuid4())

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
            success=True,
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

        # Emit workflow complete
        emit_workflow_complete("research")

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
            # Generate research questions with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed before success: {', '.join(failed)}")

            # Add to conversation memory if available
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "user",
                    prompt,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name
                )
                self.add_message(
                    thread_id,
                    "assistant",
                    response.content,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name
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

    async def extract_evidence_from_sources(
        self,
        source_ids: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        max_concurrent: int = 5,
        timeout_per_source: float = 60.0
    ) -> List[ExtractionResult]:
        """
        Extract evidence from multiple sources in parallel.

        This method processes multiple sources concurrently to extract relevant
        evidence based on the research questions. It uses asyncio for parallel
        execution while respecting concurrency limits.

        Args:
            source_ids: Optional list of specific source IDs to process.
                       If None, processes all sources in registry.
            questions: Optional list of research questions to guide extraction.
                      If None, uses general extraction approach.
            max_concurrent: Maximum number of concurrent source extractions.
                           Default is 5 to avoid overwhelming the model.
            timeout_per_source: Timeout in seconds for each source extraction.
                               Default is 60.0 seconds.

        Returns:
            List of ExtractionResult objects, one per source processed.
            Results include both successful and failed extractions.

        Raises:
            ValueError: If source_ids contains invalid IDs not in registry

        Example:
            >>> # Extract from all sources
            >>> results = await workflow.extract_evidence_from_sources()
            >>>
            >>> # Extract from specific sources with questions
            >>> questions = ["What are the key benefits?", "What are the risks?"]
            >>> results = await workflow.extract_evidence_from_sources(
            ...     source_ids=['src1', 'src2'],
            ...     questions=questions
            ... )
        """
        # Determine which sources to process
        if source_ids is None:
            sources_to_process = self.source_registry
        else:
            # Validate source IDs
            valid_ids = {s['source_id'] for s in self.source_registry}
            invalid_ids = [sid for sid in source_ids if sid not in valid_ids]
            if invalid_ids:
                raise ValueError(f"Invalid source IDs: {invalid_ids}")

            sources_to_process = [
                s for s in self.source_registry
                if s['source_id'] in source_ids
            ]

        if not sources_to_process:
            logger.warning("No sources to process for evidence extraction")
            return []

        logger.info(f"Starting parallel evidence extraction from {len(sources_to_process)} sources "
                   f"(max concurrent: {max_concurrent})")

        # Create semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create extraction tasks
        tasks = [
            self._extract_evidence_from_source(
                source=source,
                questions=questions,
                semaphore=semaphore,
                timeout=timeout_per_source
            )
            for source in sources_to_process
        ]

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        extraction_results: List[ExtractionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_id = sources_to_process[i]['source_id']
                logger.error(f"Extraction failed for source {source_id}: {result}")
                extraction_results.append(ExtractionResult(
                    source_id=source_id,
                    success=False,
                    error=str(result)
                ))
            else:
                extraction_results.append(result)

        # Log summary statistics
        successful = sum(1 for r in extraction_results if r.success)
        failed = len(extraction_results) - successful
        total_evidence = sum(len(r.evidence) for r in extraction_results if r.success)

        logger.info(f"Evidence extraction complete: {successful} successful, {failed} failed, "
                   f"{total_evidence} evidence items extracted")

        return extraction_results

    async def _extract_evidence_from_source(
        self,
        source: Dict[str, Any],
        questions: Optional[List[str]],
        semaphore: asyncio.Semaphore,
        timeout: float
    ) -> ExtractionResult:
        """
        Extract evidence from a single source.

        This is an internal method called by extract_evidence_from_sources
        for each source. It uses a semaphore to control concurrency.

        Args:
            source: Source dictionary to extract evidence from
            questions: Optional research questions to guide extraction
            semaphore: Asyncio semaphore for controlling concurrency
            timeout: Timeout in seconds for this extraction

        Returns:
            ExtractionResult with extracted evidence or error information
        """
        source_id = source['source_id']
        start_time = asyncio.get_event_loop().time()

        async with semaphore:
            try:
                # Create extraction prompt
                extraction_prompt = self._create_extraction_prompt(source, questions)

                # Create generation request
                request = GenerationRequest(
                    prompt=extraction_prompt,
                    system_prompt=self._get_evidence_extraction_system_prompt(),
                    temperature=0.3,  # Lower temperature for more focused extraction
                    max_tokens=2000
                )

                # Execute with timeout and fallback
                response, used_provider, failed = await asyncio.wait_for(
                    self._execute_with_fallback(request, self.provider, self.fallback_providers),
                    timeout=timeout
                )
                if failed:
                    logger.warning(f"Providers failed for source {source_id}: {', '.join(failed)}")

                # Parse evidence from response
                evidence_items = self._parse_evidence_from_response(
                    response.content,
                    source_id
                )

                duration = asyncio.get_event_loop().time() - start_time

                logger.info(f"Extracted {len(evidence_items)} evidence items from source {source_id} "
                           f"in {duration:.2f}s")

                return ExtractionResult(
                    source_id=source_id,
                    evidence=evidence_items,
                    success=True,
                    duration=duration,
                    metadata={
                        'source_title': source.get('title'),
                        'source_type': source.get('type'),
                        'credibility': source.get('credibility')
                    }
                )

            except asyncio.TimeoutError:
                duration = asyncio.get_event_loop().time() - start_time
                logger.warning(f"Evidence extraction timed out for source {source_id} "
                             f"after {timeout}s")
                return ExtractionResult(
                    source_id=source_id,
                    success=False,
                    error=f"Extraction timed out after {timeout}s",
                    duration=duration
                )

            except Exception as e:
                duration = asyncio.get_event_loop().time() - start_time
                logger.error(f"Evidence extraction failed for source {source_id}: {e}")
                return ExtractionResult(
                    source_id=source_id,
                    success=False,
                    error=str(e),
                    duration=duration
                )

    def _create_extraction_prompt(
        self,
        source: Dict[str, Any],
        questions: Optional[List[str]]
    ) -> str:
        """
        Create prompt for evidence extraction from a source.

        Args:
            source: Source dictionary
            questions: Optional research questions

        Returns:
            Formatted extraction prompt
        """
        prompt_parts = [
            f"Source: {source.get('title')}",
            f"Type: {source.get('type')}",
        ]

        if source.get('url'):
            prompt_parts.append(f"URL: {source['url']}")

        if source.get('metadata'):
            prompt_parts.append(f"Metadata: {source['metadata']}")

        prompt_parts.append("\n")

        if questions:
            prompt_parts.append("Research Questions:")
            for i, q in enumerate(questions, 1):
                prompt_parts.append(f"{i}. {q}")
            prompt_parts.append("\n")
            prompt_parts.append(
                "Extract key evidence from this source that addresses these research questions."
            )
        else:
            prompt_parts.append(
                "Extract key evidence and insights from this source."
            )

        prompt_parts.append("\n")
        prompt_parts.append(
            "For each piece of evidence, provide:\n"
            "1. The evidence content\n"
            "2. Relevance score (0.0-1.0)\n"
            "3. Confidence level (low/medium/high)\n"
            "4. Category or theme\n"
            "5. Supporting quote (if applicable)\n\n"
            "Format each piece of evidence clearly with these components."
        )

        return "\n".join(prompt_parts)

    def _get_evidence_extraction_system_prompt(self) -> str:
        """
        Get system prompt for evidence extraction.

        Returns:
            System prompt for evidence extraction task
        """
        return """You are an expert research analyst specializing in evidence extraction and synthesis.

Your task is to extract relevant, high-quality evidence from research sources.

Guidelines:
- Focus on factual, verifiable evidence
- Assign accurate relevance scores based on importance to research questions
- Be conservative with confidence levels - only use 'high' for well-supported claims
- Categorize evidence by theme or topic for easier synthesis
- Include direct quotes when they strengthen the evidence
- Distinguish between primary evidence and interpretations
- Note any potential biases or limitations in the source

Your extractions should be precise, well-organized, and ready for synthesis."""

    def _parse_evidence_from_response(
        self,
        response_content: str,
        source_id: str
    ) -> List[Evidence]:
        """
        Parse evidence items from model response.

        This method attempts to extract structured evidence from the model's
        response text. It looks for evidence markers and extracts components.

        Args:
            response_content: The model's response text
            source_id: ID of the source this evidence came from

        Returns:
            List of Evidence objects parsed from the response

        Note:
            This is a simplified parser. In production, you might want to use
            more sophisticated parsing or request structured JSON output.
        """
        evidence_items: List[Evidence] = []

        # Simple parsing approach: split by numbered items or blank lines
        # This is a basic implementation - could be enhanced with JSON output
        lines = response_content.strip().split('\n')

        current_evidence = {}
        evidence_content = []

        for line in lines:
            line = line.strip()
            if not line:
                # Blank line might indicate end of evidence item
                if current_evidence and evidence_content:
                    evidence_items.append(self._create_evidence_from_parsed(
                        evidence_content,
                        source_id,
                        current_evidence
                    ))
                    current_evidence = {}
                    evidence_content = []
                continue

            # Look for component markers
            if line.lower().startswith(('relevance:', 'score:')):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    current_evidence['relevance_score'] = float(score_str)
                except (ValueError, IndexError):
                    pass
            elif line.lower().startswith('confidence:'):
                confidence = line.split(':', 1)[1].strip().lower()
                if confidence in ['low', 'medium', 'high']:
                    current_evidence['confidence'] = confidence
            elif line.lower().startswith('category:'):
                current_evidence['category'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith(('quote:', 'supporting quote:')):
                current_evidence['supporting_quote'] = line.split(':', 1)[1].strip()
            else:
                # Assume it's content
                evidence_content.append(line)

        # Don't forget the last evidence item
        if current_evidence and evidence_content:
            evidence_items.append(self._create_evidence_from_parsed(
                evidence_content,
                source_id,
                current_evidence
            ))

        # If no evidence parsed (might be unstructured response), create single evidence
        if not evidence_items and response_content.strip():
            logger.warning(f"Could not parse structured evidence from source {source_id}, "
                         f"using full response as single evidence item")
            evidence_items.append(Evidence(
                evidence_id=str(uuid.uuid4())[:8],
                source_id=source_id,
                content=response_content.strip(),
                relevance_score=0.5,
                confidence='medium'
            ))

        return evidence_items

    def _create_evidence_from_parsed(
        self,
        content_lines: List[str],
        source_id: str,
        metadata: Dict[str, Any]
    ) -> Evidence:
        """
        Create an Evidence object from parsed components.

        Args:
            content_lines: Lines of evidence content
            source_id: Source ID
            metadata: Parsed metadata (relevance, confidence, etc.)

        Returns:
            Evidence object
        """
        return Evidence(
            evidence_id=str(uuid.uuid4())[:8],
            source_id=source_id,
            content=' '.join(content_lines).strip(),
            relevance_score=metadata.get('relevance_score', 0.5),
            confidence=metadata.get('confidence', 'medium'),
            category=metadata.get('category'),
            supporting_quote=metadata.get('supporting_quote'),
            metadata=metadata
        )

    async def validate_evidence(
        self,
        evidence_items: List[Evidence],
        cross_reference: bool = True,
        detect_contradictions: bool = True
    ) -> List[ValidationResult]:
        """
        Validate evidence items with fact-checking and confidence scoring.

        This method performs systematic validation of evidence including:
        - Internal consistency checking
        - Cross-referencing between evidence items
        - Contradiction detection
        - Confidence score calculation

        Args:
            evidence_items: List of Evidence objects to validate
            cross_reference: Whether to cross-reference evidence items (default: True)
            detect_contradictions: Whether to detect contradictions (default: True)

        Returns:
            List of ValidationResult objects with validation outcomes

        Example:
            >>> # Validate extracted evidence
            >>> evidence_list = [evidence1, evidence2, evidence3]
            >>> validation_results = await workflow.validate_evidence(evidence_list)
            >>>
            >>> # Check for contradictions
            >>> for result in validation_results:
            ...     if result.contradictions:
            ...         print(f"Evidence {result.evidence_id} has contradictions")
        """
        if not evidence_items:
            logger.warning("No evidence items provided for validation")
            return []

        logger.info(f"Validating {len(evidence_items)} evidence items "
                   f"(cross_reference: {cross_reference}, detect_contradictions: {detect_contradictions})")

        validation_results: List[ValidationResult] = []

        for evidence in evidence_items:
            # Validate each piece of evidence
            result = await self._validate_single_evidence(
                evidence=evidence,
                all_evidence=evidence_items if cross_reference else [evidence],
                detect_contradictions=detect_contradictions
            )
            validation_results.append(result)

        # Log summary
        valid_count = sum(1 for r in validation_results if r.is_valid)
        avg_confidence = sum(r.confidence_score for r in validation_results) / len(validation_results)
        contradictions_found = sum(len(r.contradictions) for r in validation_results)

        logger.info(f"Validation complete: {valid_count}/{len(validation_results)} valid, "
                   f"avg confidence: {avg_confidence:.2f}, contradictions: {contradictions_found}")

        return validation_results

    async def _validate_single_evidence(
        self,
        evidence: Evidence,
        all_evidence: List[Evidence],
        detect_contradictions: bool
    ) -> ValidationResult:
        """
        Validate a single piece of evidence.

        Args:
            evidence: Evidence to validate
            all_evidence: All evidence items for cross-referencing
            detect_contradictions: Whether to detect contradictions

        Returns:
            ValidationResult with validation outcome
        """
        # Create validation prompt
        validation_prompt = self._create_validation_prompt(evidence, all_evidence, detect_contradictions)

        try:
            # Create generation request
            request = GenerationRequest(
                prompt=validation_prompt,
                system_prompt=self._get_validation_system_prompt(),
                temperature=0.2,  # Low temperature for consistent validation
                max_tokens=1500
            )

            # Get validation response with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed for validation of {evidence.evidence_id}: {', '.join(failed)}")

            # Parse validation result
            validation_result = self._parse_validation_response(
                response.content,
                evidence.evidence_id
            )

            logger.info(f"Validated evidence {evidence.evidence_id}: "
                       f"valid={validation_result.is_valid}, "
                       f"confidence={validation_result.confidence_score:.2f}")

            return validation_result

        except Exception as e:
            logger.error(f"Validation failed for evidence {evidence.evidence_id}: {e}")
            # Return default validation result on error
            return ValidationResult(
                evidence_id=evidence.evidence_id,
                is_valid=True,  # Assume valid if validation fails
                confidence_score=0.3,  # Low confidence
                validation_notes=[f"Validation error: {str(e)}"]
            )

    def _create_validation_prompt(
        self,
        evidence: Evidence,
        all_evidence: List[Evidence],
        detect_contradictions: bool
    ) -> str:
        """
        Create prompt for evidence validation.

        Args:
            evidence: Evidence to validate
            all_evidence: All evidence for cross-referencing
            detect_contradictions: Whether to check for contradictions

        Returns:
            Formatted validation prompt
        """
        prompt_parts = [
            "Evidence to Validate:",
            f"ID: {evidence.evidence_id}",
            f"Content: {evidence.content}",
            f"Source: {evidence.source_id}",
            f"Initial Confidence: {evidence.confidence}",
            ""
        ]

        if evidence.supporting_quote:
            prompt_parts.append(f"Supporting Quote: {evidence.supporting_quote}")
            prompt_parts.append("")

        prompt_parts.append("Validation Tasks:")
        prompt_parts.append("1. Assess the factual accuracy and credibility of this evidence")
        prompt_parts.append("2. Assign a confidence score (0.0-1.0) based on:")
        prompt_parts.append("   - Evidence quality and specificity")
        prompt_parts.append("   - Source credibility")
        prompt_parts.append("   - Presence of supporting quotes or citations")
        prompt_parts.append("   - Internal consistency")

        if detect_contradictions and len(all_evidence) > 1:
            prompt_parts.append("3. Check for contradictions with other evidence:")
            prompt_parts.append("")
            # Include snippets of other evidence for comparison
            other_evidence = [e for e in all_evidence if e.evidence_id != evidence.evidence_id]
            for i, other in enumerate(other_evidence[:5], 1):  # Limit to 5 for context
                prompt_parts.append(f"   Evidence {i} ({other.evidence_id}): {other.content[:150]}...")
            prompt_parts.append("")

        prompt_parts.append("Provide:")
        prompt_parts.append("- Is Valid: true/false")
        prompt_parts.append("- Confidence Score: 0.0-1.0")
        prompt_parts.append("- Validation Notes: Key findings")
        prompt_parts.append("- Contradictions: Any contradictions found (if applicable)")
        prompt_parts.append("- Supporting Evidence IDs: Evidence that supports this claim")
        prompt_parts.append("- Refuting Evidence IDs: Evidence that refutes this claim")

        return "\n".join(prompt_parts)

    def _get_validation_system_prompt(self) -> str:
        """
        Get system prompt for evidence validation.

        Returns:
            System prompt for validation task
        """
        return """You are an expert fact-checker and research validator specializing in evidence assessment.

Your task is to rigorously validate evidence for accuracy, consistency, and reliability.

Validation Guidelines:
- Be critical but fair in assessing evidence quality
- High confidence (0.8-1.0): Multiple sources, verifiable facts, strong citations
- Medium confidence (0.5-0.7): Reasonable claims, some support, plausible
- Low confidence (0.0-0.4): Unverified claims, contradictions, weak sources

Contradiction Detection:
- Identify direct contradictions in facts or claims
- Note when evidence presents different perspectives vs. actual contradictions
- Be precise about what contradicts and why

Confidence Scoring Factors:
- Source credibility and authority
- Specificity and detail level
- Presence of citations or supporting quotes
- Consistency with other evidence
- Verifiability of claims

Be thorough, objective, and provide clear reasoning for your assessments."""

    def _parse_validation_response(
        self,
        response_content: str,
        evidence_id: str
    ) -> ValidationResult:
        """
        Parse validation result from model response.

        Args:
            response_content: Model's validation response
            evidence_id: ID of evidence being validated

        Returns:
            ValidationResult object
        """
        result = ValidationResult(evidence_id=evidence_id)

        lines = response_content.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse structured components
            if line.lower().startswith('is valid:'):
                valid_str = line.split(':', 1)[1].strip().lower()
                result.is_valid = valid_str in ['true', 'yes', 'valid']

            elif line.lower().startswith('confidence score:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    # Extract just the number (handle "0.85" or "0.85 (high)" formats)
                    score = float(score_str.split()[0])
                    result.confidence_score = max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass

            elif line.lower().startswith('validation notes:'):
                current_section = 'notes'
                note = line.split(':', 1)[1].strip()
                if note:
                    result.validation_notes.append(note)

            elif line.lower().startswith('contradictions:'):
                current_section = 'contradictions'
                contradiction = line.split(':', 1)[1].strip()
                if contradiction and contradiction.lower() not in ['none', 'n/a']:
                    result.contradictions.append(contradiction)

            elif line.lower().startswith('supporting evidence'):
                current_section = 'supporting'
                # Extract IDs from the line
                ids_str = line.split(':', 1)[1].strip() if ':' in line else ''
                if ids_str and ids_str.lower() not in ['none', 'n/a']:
                    result.supporting_evidence.extend(ids_str.split(','))

            elif line.lower().startswith('refuting evidence'):
                current_section = 'refuting'
                ids_str = line.split(':', 1)[1].strip() if ':' in line else ''
                if ids_str and ids_str.lower() not in ['none', 'n/a']:
                    result.refuting_evidence.extend(ids_str.split(','))

            elif current_section == 'notes' and line.startswith(('-', '•', '*')):
                result.validation_notes.append(line.lstrip('-•* '))

            elif current_section == 'contradictions' and line.startswith(('-', '•', '*')):
                result.contradictions.append(line.lstrip('-•* '))

        # Ensure we have at least one validation note
        if not result.validation_notes:
            result.validation_notes.append("Validation completed")

        return result

    def generate_dossier(
        self,
        evidence_items: List[Evidence],
        validation_results: Optional[List[ValidationResult]] = None,
        include_citations: bool = True,
        include_evidence_table: bool = True,
        include_contradictions: bool = True,
        citation_style: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive research dossier with citations and evidence tables.

        This method compiles all research findings into a formatted dossier including:
        - Executive summary of findings
        - Source citations
        - Evidence tables organized by category
        - Contradiction analysis
        - Validation metrics

        Args:
            evidence_items: List of Evidence objects to include
            validation_results: Optional validation results for evidence
            include_citations: Whether to include source citations (default: True)
            include_evidence_table: Whether to include evidence tables (default: True)
            include_contradictions: Whether to include contradiction analysis (default: True)
            citation_style: Citation style to use (informal/academic/technical)
                          If None, uses config default

        Returns:
            Formatted research dossier as markdown string

        Example:
            >>> # Generate full dossier
            >>> evidence = await workflow.extract_evidence_from_sources()
            >>> validation = await workflow.validate_evidence(evidence)
            >>> dossier = workflow.generate_dossier(
            ...     evidence_items=evidence,
            ...     validation_results=validation
            ... )
            >>> print(dossier)
        """
        if not evidence_items:
            logger.warning("No evidence provided for dossier generation")
            return "# Research Dossier\n\nNo evidence available."

        # Determine citation style
        style = citation_style or self.default_config.get('citation_style', 'informal')

        logger.info(f"Generating research dossier with {len(evidence_items)} evidence items "
                   f"(style: {style})")

        # Build dossier sections
        sections = []

        # Header
        sections.append("# Research Dossier")
        sections.append("")
        sections.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        sections.append("")

        # Executive Summary
        sections.append(self._create_executive_summary(evidence_items, validation_results))
        sections.append("")

        # Source Citations
        if include_citations and self.source_registry:
            sections.append(self._format_citations(style))
            sections.append("")

        # Evidence Tables
        if include_evidence_table:
            sections.append(self._create_evidence_table(evidence_items, validation_results))
            sections.append("")

        # Contradiction Analysis
        if include_contradictions and validation_results:
            contradictions_section = self._format_contradictions(validation_results, evidence_items)
            if contradictions_section:
                sections.append(contradictions_section)
                sections.append("")

        # Validation Metrics
        if validation_results:
            sections.append(self._create_validation_metrics(validation_results))
            sections.append("")

        # Footer
        sections.append("---")
        sections.append(f"*Dossier generated by ModelChorus ResearchWorkflow*")

        dossier = "\n".join(sections)
        logger.info(f"Research dossier generated: {len(dossier)} characters")

        return dossier

    def _create_executive_summary(
        self,
        evidence_items: List[Evidence],
        validation_results: Optional[List[ValidationResult]]
    ) -> str:
        """
        Create executive summary section of dossier.

        Args:
            evidence_items: List of evidence
            validation_results: Optional validation results

        Returns:
            Formatted executive summary section
        """
        lines = ["## Executive Summary", ""]

        # Evidence count
        lines.append(f"**Total Evidence Items:** {len(evidence_items)}")

        # Source count
        unique_sources = len(set(e.source_id for e in evidence_items))
        lines.append(f"**Unique Sources:** {unique_sources}")

        # Categories
        categories = set(e.category for e in evidence_items if e.category)
        if categories:
            lines.append(f"**Categories Covered:** {len(categories)}")

        # Validation summary
        if validation_results:
            valid_count = sum(1 for r in validation_results if r.is_valid)
            avg_confidence = sum(r.confidence_score for r in validation_results) / len(validation_results)
            lines.append(f"**Validated Evidence:** {valid_count}/{len(validation_results)} "
                        f"({valid_count/len(validation_results)*100:.1f}%)")
            lines.append(f"**Average Confidence Score:** {avg_confidence:.2f}")

            # Contradictions
            contradictions = sum(len(r.contradictions) for r in validation_results)
            if contradictions > 0:
                lines.append(f"**Contradictions Found:** {contradictions}")

        return "\n".join(lines)

    def _format_citations(self, style: str = 'informal') -> str:
        """
        Format source citations in specified style.

        Args:
            style: Citation style (informal/academic/technical)

        Returns:
            Formatted citations section
        """
        lines = ["## Source Citations", ""]

        if not self.source_registry:
            lines.append("*No sources registered*")
            return "\n".join(lines)

        # Sort sources by type and credibility
        sorted_sources = sorted(
            self.source_registry,
            key=lambda s: (s.get('credibility', 'low'), s.get('type', 'unknown'))
        )

        for i, source in enumerate(sorted_sources, 1):
            citation = self._format_single_citation(source, style, i)
            lines.append(citation)
            lines.append("")

        return "\n".join(lines)

    def _format_single_citation(
        self,
        source: Dict[str, Any],
        style: str,
        number: int
    ) -> str:
        """
        Format a single source citation.

        Args:
            source: Source dictionary
            style: Citation style
            number: Citation number

        Returns:
            Formatted citation string
        """
        source_id = source.get('source_id', 'unknown')
        title = source.get('title', 'Untitled')
        url = source.get('url')
        source_type = source.get('type', 'unknown')
        credibility = source.get('credibility', 'unassessed')

        if style == 'academic':
            # Academic style: [Number] Title. Type. URL (Credibility: level)
            citation = f"[{number}] **{title}**. *{source_type.capitalize()}*."
            if url:
                citation += f" Available at: {url}."
            citation += f" (Credibility: {credibility})"

        elif style == 'technical':
            # Technical style: [source_id] Title - Type (credibility) <URL>
            citation = f"[{source_id}] **{title}** - {source_type}"
            if url:
                citation += f" <{url}>"
            citation += f" (credibility: {credibility})"

        else:  # informal
            # Informal style: • Title (type) [credibility]
            citation = f"• **{title}** ({source_type})"
            if url:
                citation += f" - {url}"
            citation += f" [*{credibility} credibility*]"

        return citation

    def _create_evidence_table(
        self,
        evidence_items: List[Evidence],
        validation_results: Optional[List[ValidationResult]]
    ) -> str:
        """
        Create formatted evidence tables organized by category.

        Args:
            evidence_items: List of evidence
            validation_results: Optional validation results

        Returns:
            Formatted evidence tables section
        """
        lines = ["## Evidence Analysis", ""]

        # Create validation lookup
        validation_map = {}
        if validation_results:
            validation_map = {v.evidence_id: v for v in validation_results}

        # Group evidence by category
        categorized: Dict[str, List[Evidence]] = {}
        uncategorized: List[Evidence] = []

        for evidence in evidence_items:
            category = evidence.category
            if category:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(evidence)
            else:
                uncategorized.append(evidence)

        # Create table for each category
        for category, items in sorted(categorized.items()):
            lines.append(f"### {category}")
            lines.append("")
            lines.append(self._create_evidence_category_table(items, validation_map))
            lines.append("")

        # Add uncategorized evidence if any
        if uncategorized:
            lines.append("### Uncategorized Evidence")
            lines.append("")
            lines.append(self._create_evidence_category_table(uncategorized, validation_map))

        return "\n".join(lines)

    def _create_evidence_category_table(
        self,
        evidence_items: List[Evidence],
        validation_map: Dict[str, ValidationResult]
    ) -> str:
        """
        Create markdown table for a category of evidence.

        Args:
            evidence_items: Evidence items in this category
            validation_map: Map of evidence ID to validation result

        Returns:
            Formatted markdown table
        """
        lines = []

        # Table header
        lines.append("| Evidence | Source | Confidence | Relevance | Validation |")
        lines.append("|----------|--------|-----------|-----------|------------|")

        # Table rows
        for evidence in evidence_items:
            # Truncate content for table display
            content = evidence.content[:80] + "..." if len(evidence.content) > 80 else evidence.content
            content = content.replace("|", "\\|")  # Escape pipe characters

            source_id = evidence.source_id[:8]
            confidence = evidence.confidence
            relevance = f"{evidence.relevance_score:.2f}"

            # Add validation info if available
            validation_info = "—"
            if evidence.evidence_id in validation_map:
                val_result = validation_map[evidence.evidence_id]
                if val_result.is_valid:
                    validation_info = f"✓ {val_result.confidence_score:.2f}"
                else:
                    validation_info = f"✗ {val_result.confidence_score:.2f}"

            lines.append(f"| {content} | {source_id} | {confidence} | {relevance} | {validation_info} |")

        return "\n".join(lines)

    def _format_contradictions(
        self,
        validation_results: List[ValidationResult],
        evidence_items: List[Evidence]
    ) -> str:
        """
        Format contradiction analysis section.

        Args:
            validation_results: Validation results with contradictions
            evidence_items: Evidence items for reference

        Returns:
            Formatted contradictions section, or empty string if none
        """
        # Find all validation results with contradictions
        with_contradictions = [
            v for v in validation_results
            if v.contradictions
        ]

        if not with_contradictions:
            return ""

        lines = ["## Contradiction Analysis", ""]
        lines.append(f"**Contradictions Found:** {len(with_contradictions)} evidence items with conflicts")
        lines.append("")

        # Create evidence lookup
        evidence_map = {e.evidence_id: e for e in evidence_items}

        for val_result in with_contradictions:
            evidence = evidence_map.get(val_result.evidence_id)
            if not evidence:
                continue

            lines.append(f"### Evidence {val_result.evidence_id}")
            lines.append("")
            lines.append(f"**Content:** {evidence.content}")
            lines.append(f"**Source:** {evidence.source_id}")
            lines.append(f"**Confidence:** {val_result.confidence_score:.2f}")
            lines.append("")
            lines.append("**Contradictions:**")
            for contradiction in val_result.contradictions:
                lines.append(f"- {contradiction}")

            # Add supporting/refuting evidence if available
            if val_result.supporting_evidence:
                lines.append("")
                lines.append("**Supporting Evidence:**")
                for supp_id in val_result.supporting_evidence:
                    lines.append(f"- {supp_id}")

            if val_result.refuting_evidence:
                lines.append("")
                lines.append("**Refuting Evidence:**")
                for ref_id in val_result.refuting_evidence:
                    lines.append(f"- {ref_id}")

            lines.append("")

        return "\n".join(lines)

    def _create_validation_metrics(
        self,
        validation_results: List[ValidationResult]
    ) -> str:
        """
        Create validation metrics summary section.

        Args:
            validation_results: List of validation results

        Returns:
            Formatted validation metrics section
        """
        lines = ["## Validation Metrics", ""]

        # Overall statistics
        total = len(validation_results)
        valid = sum(1 for v in validation_results if v.is_valid)
        invalid = total - valid

        lines.append(f"**Total Validated:** {total}")
        lines.append(f"**Valid:** {valid} ({valid/total*100:.1f}%)")
        lines.append(f"**Invalid:** {invalid} ({invalid/total*100:.1f}%)")
        lines.append("")

        # Confidence distribution
        high_conf = sum(1 for v in validation_results if v.confidence_score >= 0.7)
        medium_conf = sum(1 for v in validation_results if 0.4 <= v.confidence_score < 0.7)
        low_conf = sum(1 for v in validation_results if v.confidence_score < 0.4)

        lines.append("**Confidence Distribution:**")
        lines.append(f"- High (≥0.7): {high_conf} ({high_conf/total*100:.1f}%)")
        lines.append(f"- Medium (0.4-0.7): {medium_conf} ({medium_conf/total*100:.1f}%)")
        lines.append(f"- Low (<0.4): {low_conf} ({low_conf/total*100:.1f}%)")
        lines.append("")

        # Average confidence
        avg_confidence = sum(v.confidence_score for v in validation_results) / total
        lines.append(f"**Average Confidence Score:** {avg_confidence:.3f}")

        return "\n".join(lines)
