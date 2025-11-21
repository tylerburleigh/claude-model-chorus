"""
Role-based orchestration framework for sequential and parallel model execution.

Provides infrastructure for coordinating multiple AI models in different roles,
enabling workflows like ARGUMENT (multi-perspective debate), IDEATE (creative
brainstorming with diverse viewpoints), and RESEARCH (multi-source investigation).

This module defines the core abstractions for role assignment, stance configuration,
and orchestration patterns that enable sophisticated multi-model workflows.

Public API:
    - ModelRole: Data class defining a model's role, stance, and prompt customization
    - RoleOrchestrator: Coordinator for executing models in assigned roles
    - OrchestrationPattern: Enum for execution patterns (sequential/parallel/hybrid)
    - OrchestrationResult: Result data structure from orchestrated execution
    - SynthesisStrategy: Enum for strategies to combine role outputs
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class OrchestrationPattern(str, Enum):
    """
    Execution patterns for multi-model orchestration.

    Defines how multiple models with assigned roles are coordinated
    during workflow execution. Different patterns enable different
    collaboration strategies.

    Values:
        SEQUENTIAL: Execute models one at a time in defined order
                   (e.g., analyst → critic → synthesizer)
        PARALLEL: Execute all models concurrently, then aggregate
                 (e.g., multiple experts providing simultaneous input)
        HYBRID: Mix of sequential and parallel phases
               (e.g., parallel research → sequential debate → parallel voting)
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class SynthesisStrategy(str, Enum):
    """
    Strategies for combining multiple role outputs into a unified result.

    After roles execute (sequentially or in parallel), their outputs can be
    synthesized using different strategies depending on workflow needs.

    Values:
        NONE: No synthesis - return raw responses as-is
             Use when: You want to process each response individually

        CONCATENATE: Simple concatenation with role labels
                    Use when: You want a readable combined text with clear attribution

        AI_SYNTHESIZE: Use AI model to intelligently combine responses
                      Use when: You want a coherent synthesis that resolves conflicts
                               and integrates multiple perspectives

        STRUCTURED: Combine into structured format (dict with role keys)
                   Use when: You need programmatic access to individual responses
                            while keeping them organized

    Example:
        >>> # Get raw responses
        >>> result = await orchestrator.execute(prompt, synthesis=SynthesisStrategy.NONE)
        >>>
        >>> # Get simple concatenation
        >>> result = await orchestrator.execute(prompt, synthesis=SynthesisStrategy.CONCATENATE)
        >>>
        >>> # Get AI-synthesized output
        >>> result = await orchestrator.execute(
        ...     prompt,
        ...     synthesis=SynthesisStrategy.AI_SYNTHESIZE,
        ...     synthesis_provider=synthesis_model
        ... )
    """

    NONE = "none"
    CONCATENATE = "concatenate"
    AI_SYNTHESIZE = "ai_synthesize"
    STRUCTURED = "structured"


class ModelRole(BaseModel):
    """
    Data class defining a model's role, stance, and prompt customization.

    Represents a specific role assignment for an AI model in a multi-model
    workflow. Includes the role name, optional stance (for/against/neutral),
    and prompt customization to guide the model's behavior.

    Used in workflows like ARGUMENT (models with different stances),
    IDEATE (models with different creative perspectives), and RESEARCH
    (models focusing on different aspects of investigation).

    Attributes:
        role: Descriptive name for this role (e.g., "proponent", "critic", "synthesizer")
        model: Model identifier to assign to this role (e.g., "gpt-5", "gemini-2.5-pro")
        stance: Optional stance for debate-style workflows ("for", "against", "neutral")
        stance_prompt: Optional additional prompt text to reinforce the stance
        system_prompt: Optional system-level prompt for this role
        temperature: Optional temperature override for this role (0.0-1.0)
        max_tokens: Optional max tokens override for this role
        metadata: Additional metadata for this role (tags, priority, etc.)

    Example:
        >>> proponent = ModelRole(
        ...     role="proponent",
        ...     model="gpt-5",
        ...     stance="for",
        ...     stance_prompt="You are advocating FOR the proposal. Present strong supporting arguments."
        ... )
        >>> critic = ModelRole(
        ...     role="critic",
        ...     model="gemini-2.5-pro",
        ...     stance="against",
        ...     stance_prompt="You are critically analyzing AGAINST the proposal. Identify weaknesses and risks."
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "proponent",
                "model": "gpt-5",
                "stance": "for",
                "stance_prompt": "You are advocating FOR the proposal. Present strong supporting arguments.",
                "system_prompt": "You are an expert debater focused on building compelling cases.",
                "temperature": 0.8,
                "max_tokens": 4000,
                "metadata": {
                    "priority": 1,
                    "tags": ["debate", "advocacy"],
                },
            }
        }
    )

    role: str = Field(
        ...,
        description="Descriptive name for this role (e.g., 'proponent', 'critic', 'synthesizer')",
        min_length=1,
        max_length=100,
    )

    model: str = Field(
        ...,
        description="Model identifier to assign to this role (e.g., 'gpt-5', 'gemini-2.5-pro')",
        min_length=1,
    )

    stance: str | None = Field(
        default=None,
        description="Optional stance for debate-style workflows ('for', 'against', 'neutral')",
    )

    stance_prompt: str | None = Field(
        default=None,
        description="Optional additional prompt text to reinforce the stance",
    )

    system_prompt: str | None = Field(
        default=None,
        description="Optional system-level prompt for this role",
    )

    temperature: float | None = Field(
        default=None,
        description="Optional temperature override for this role (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    max_tokens: int | None = Field(
        default=None,
        description="Optional max tokens override for this role",
        gt=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this role (tags, priority, constraints, etc.)",
    )

    @field_validator("stance")
    @classmethod
    def validate_stance(cls, v: str | None) -> str | None:
        """
        Validate stance is one of the allowed values.

        Ensures stance, if provided, is one of the standard values
        for consistency across workflows.

        Args:
            v: Stance value to validate

        Returns:
            Validated stance value (lowercase)

        Raises:
            ValueError: If stance is not one of the allowed values
        """
        if v is None:
            return v

        allowed_stances = {"for", "against", "neutral"}
        stance_lower = v.lower()

        if stance_lower not in allowed_stances:
            raise ValueError(f"Stance must be one of {allowed_stances}, got '{v}'")

        return stance_lower

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float | None) -> float | None:
        """
        Validate temperature is in valid range.

        Ensures temperature, if provided, is between 0.0 and 1.0 inclusive.
        Pydantic's ge/le constraints handle this, but explicit validator
        provides clearer error messages.

        Args:
            v: Temperature value to validate

        Returns:
            Validated temperature value

        Raises:
            ValueError: If temperature is outside [0.0, 1.0] range
        """
        if v is None:
            return v

        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {v}")

        return v

    def get_full_prompt(self, base_prompt: str) -> str:
        """
        Construct full prompt by combining base prompt with role customizations.

        Merges the workflow's base prompt with this role's stance prompt
        and system prompt to create the complete prompt sent to the model.

        Args:
            base_prompt: The base prompt from the workflow

        Returns:
            Complete prompt string with role customizations applied

        Example:
            >>> role = ModelRole(
            ...     role="critic",
            ...     model="gpt-5",
            ...     stance="against",
            ...     stance_prompt="Identify weaknesses and risks.",
            ...     system_prompt="You are a critical analyst."
            ... )
            >>> full_prompt = role.get_full_prompt("Analyze this proposal: ...")
            >>> print(full_prompt)
            You are a critical analyst.

            Identify weaknesses and risks.

            Analyze this proposal: ...
        """
        prompt_parts = []

        # Add system prompt if provided
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)

        # Add stance prompt if provided
        if self.stance_prompt:
            prompt_parts.append(self.stance_prompt)

        # Add base prompt
        prompt_parts.append(base_prompt)

        return "\n\n".join(prompt_parts)


@dataclass
class OrchestrationResult:
    """
    Result from orchestrating multiple models with assigned roles.

    Contains responses from all models, execution metadata, synthesis output,
    and error information. Used by RoleOrchestrator to return structured results
    after executing a multi-model workflow.

    Attributes:
        role_responses: List of (role_name, response) tuples in execution order
        all_responses: List of all GenerationResponse objects
        failed_roles: List of role names that failed to execute
        pattern_used: Orchestration pattern that was executed
        execution_order: List of role names in the order they were executed
        synthesized_output: Optional synthesized/combined output (if synthesis enabled)
        synthesis_strategy: Strategy used for synthesis (if any)
        metadata: Additional execution metadata (timing, context, synthesis_metadata, etc.)

    Example:
        >>> result = OrchestrationResult(
        ...     role_responses=[
        ...         ("proponent", GenerationResponse(content="Argument FOR...", model="gpt-5")),
        ...         ("critic", GenerationResponse(content="Argument AGAINST...", model="gemini-2.5-pro")),
        ...     ],
        ...     pattern_used=OrchestrationPattern.SEQUENTIAL,
        ...     execution_order=["proponent", "critic"],
        ...     synthesized_output="After considering both perspectives...",
        ...     synthesis_strategy=SynthesisStrategy.AI_SYNTHESIZE
        ... )
    """

    role_responses: list[tuple[str, Any]] = field(default_factory=list)
    all_responses: list[Any] = field(default_factory=list)
    failed_roles: list[str] = field(default_factory=list)
    pattern_used: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL
    execution_order: list[str] = field(default_factory=list)
    synthesized_output: Any | None = None
    synthesis_strategy: SynthesisStrategy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RoleOrchestrator:
    """
    Coordinator for executing multiple models with assigned roles.

    Manages the execution of multi-model workflows where each model has a specific
    role (e.g., proponent, critic, synthesizer). Supports both sequential and parallel
    execution patterns. Handles provider resolution, prompt customization, and result
    aggregation.

    Execution Patterns:
        SEQUENTIAL: Roles execute one at a time in defined order
                   (e.g., analyst → critic → synthesizer)
        PARALLEL: All roles execute concurrently for independent perspectives
                 (e.g., multiple experts providing simultaneous input)
        HYBRID: Not yet implemented (future work)

    This orchestrator enables workflows like:
    - ARGUMENT: Sequential debate (roles take turns) or parallel perspectives
    - IDEATE: Multiple creative perspectives (parallel for diversity)
    - RESEARCH: Multi-angle investigation (parallel for breadth)

    Attributes:
        roles: List of ModelRole instances defining the workflow
        provider_map: Mapping from model identifiers to provider instances
        pattern: Orchestration pattern (SEQUENTIAL or PARALLEL)
        default_timeout: Default timeout for each model execution (seconds)

    Example:
        >>> from model_chorus.core.role_orchestration import ModelRole, RoleOrchestrator
        >>> from model_chorus.providers import ClaudeProvider, GeminiProvider
        >>>
        >>> # Define roles
        >>> roles = [
        ...     ModelRole(role="proponent", model="claude", stance="for"),
        ...     ModelRole(role="critic", model="gemini", stance="against"),
        ... ]
        >>>
        >>> # Create provider map
        >>> providers = {
        ...     "claude": ClaudeProvider(),
        ...     "gemini": GeminiProvider(),
        ... }
        >>>
        >>> # Create orchestrator
        >>> orchestrator = RoleOrchestrator(roles, providers)
        >>>
        >>> # Execute workflow
        >>> result = await orchestrator.execute("Should we adopt this proposal?")
        >>> for role_name, response in result.role_responses:
        ...     print(f"{role_name}: {response.content}")
    """

    def __init__(
        self,
        roles: list[ModelRole],
        provider_map: dict[str, Any],
        pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL,
        default_timeout: float = 120.0,
    ):
        """
        Initialize the role orchestrator.

        Args:
            roles: List of ModelRole instances defining the workflow
            provider_map: Mapping from model identifiers to provider instances
                         (e.g., {"gpt-5": openai_provider, "gemini-2.5-pro": gemini_provider})
            pattern: Orchestration pattern to use (default: SEQUENTIAL)
            default_timeout: Default timeout for each model execution in seconds (default: 120.0)

        Raises:
            ValueError: If roles list is empty
            ValueError: If pattern is not SEQUENTIAL (only sequential supported currently)
        """
        if not roles:
            raise ValueError("At least one role is required")

        if pattern not in (
            OrchestrationPattern.SEQUENTIAL,
            OrchestrationPattern.PARALLEL,
        ):
            raise ValueError(
                f"Only SEQUENTIAL and PARALLEL patterns are supported, got {pattern}. "
                f"HYBRID pattern is not yet implemented."
            )

        self.roles = roles
        self.provider_map = provider_map
        self.pattern = pattern
        self.default_timeout = default_timeout

        logger.info(
            f"Initialized RoleOrchestrator with {len(roles)} roles, pattern={pattern}"
        )

    def _resolve_provider(self, model_id: str) -> Any:
        """
        Resolve a model identifier to a provider instance.

        Looks up the provider for the given model identifier in the provider map.
        Handles common variations (e.g., "gpt5" vs "gpt-5") by checking multiple
        formats.

        Args:
            model_id: Model identifier from ModelRole.model

        Returns:
            Provider instance for the model

        Raises:
            ValueError: If no provider is found for the model identifier
        """
        # Try exact match first
        if model_id in self.provider_map:
            return self.provider_map[model_id]

        # Try common variations
        variations = [
            model_id.lower(),
            model_id.replace("-", ""),
            model_id.replace("_", ""),
        ]

        for variation in variations:
            if variation in self.provider_map:
                logger.debug(
                    f"Resolved model '{model_id}' to provider via variation '{variation}'"
                )
                return self.provider_map[variation]

        # No provider found
        available = ", ".join(self.provider_map.keys())
        raise ValueError(
            f"No provider found for model '{model_id}'. Available providers: {available}"
        )

    async def execute(
        self,
        base_prompt: str,
        context: str | None = None,
    ) -> OrchestrationResult:
        """
        Execute the orchestrated workflow with all roles.

        Executes each role according to the configured pattern (sequential or parallel),
        using the base prompt customized for each role's stance and configuration.
        Collects all responses and returns a structured result.

        Sequential pattern: Executes roles one at a time in order
        Parallel pattern: Executes all roles concurrently

        Args:
            base_prompt: The base prompt to send to all models
            context: Optional additional context to include (e.g., from previous execution)

        Returns:
            OrchestrationResult containing all responses and execution metadata

        Example (Sequential):
            >>> orchestrator = RoleOrchestrator(roles, providers, pattern=OrchestrationPattern.SEQUENTIAL)
            >>> result = await orchestrator.execute("Should we adopt TypeScript?")
            >>> # Roles execute one after another

        Example (Parallel):
            >>> orchestrator = RoleOrchestrator(roles, providers, pattern=OrchestrationPattern.PARALLEL)
            >>> result = await orchestrator.execute("Should we adopt TypeScript?")
            >>> # All roles execute simultaneously
        """
        # Route to appropriate execution method based on pattern
        if self.pattern == OrchestrationPattern.SEQUENTIAL:
            return await self._execute_sequential(base_prompt, context)
        elif self.pattern == OrchestrationPattern.PARALLEL:
            return await self._execute_parallel(base_prompt, context)
        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")

    async def _execute_sequential(
        self,
        base_prompt: str,
        context: str | None = None,
    ) -> OrchestrationResult:
        """
        Execute all roles sequentially in order.

        Internal method for sequential execution pattern. Roles are executed
        one at a time in the order they appear in self.roles.

        Args:
            base_prompt: The base prompt to send to all models
            context: Optional additional context to include

        Returns:
            OrchestrationResult with sequential execution metadata
        """
        # Import here to avoid circular dependency
        from ..providers import GenerationRequest

        logger.info(f"Starting sequential execution of {len(self.roles)} roles")

        role_responses = []
        all_responses = []
        failed_roles = []
        execution_order = []

        # Build context-enhanced prompt if context provided
        full_base_prompt = base_prompt
        if context:
            full_base_prompt = f"{context}\n\n{base_prompt}"

        # Execute each role sequentially
        for idx, role in enumerate(self.roles):
            execution_order.append(role.role)

            try:
                logger.info(
                    f"Executing role {idx + 1}/{len(self.roles)}: {role.role} (model: {role.model})"
                )

                # Resolve provider for this role
                provider = self._resolve_provider(role.model)

                # Construct full prompt with role customizations
                full_prompt = role.get_full_prompt(full_base_prompt)

                # Create generation request with role-specific overrides
                request = GenerationRequest(
                    prompt=full_prompt,
                    system_prompt=role.system_prompt,
                    temperature=role.temperature,
                    max_tokens=role.max_tokens,
                )

                # Execute via provider
                response = await provider.generate(request)

                # Store results
                role_responses.append((role.role, response))
                all_responses.append(response)

                logger.info(
                    f"Role '{role.role}' completed successfully ({len(response.content)} chars)"
                )

            except Exception as e:
                logger.error(f"Role '{role.role}' failed: {e}", exc_info=True)
                failed_roles.append(role.role)
                # Continue with remaining roles even if one fails

        # Build result
        result = OrchestrationResult(
            role_responses=role_responses,
            all_responses=all_responses,
            failed_roles=failed_roles,
            pattern_used=self.pattern,
            execution_order=execution_order,
            metadata={
                "total_roles": len(self.roles),
                "successful_roles": len(role_responses),
                "failed_roles": len(failed_roles),
            },
        )

        logger.info(
            f"Sequential orchestration complete: {len(role_responses)}/{len(self.roles)} roles succeeded"
        )

        return result

    async def _execute_parallel(
        self,
        base_prompt: str,
        context: str | None = None,
    ) -> OrchestrationResult:
        """
        Execute all roles in parallel (concurrently).

        Internal method for parallel execution pattern. All roles are executed
        simultaneously using asyncio.gather, which improves performance when roles
        are independent.

        Args:
            base_prompt: The base prompt to send to all models
            context: Optional additional context to include

        Returns:
            OrchestrationResult with parallel execution metadata
        """
        # Import here to avoid circular dependency
        from ..providers import GenerationRequest

        logger.info(f"Starting parallel execution of {len(self.roles)} roles")

        # Build context-enhanced prompt if context provided
        full_base_prompt = base_prompt
        if context:
            full_base_prompt = f"{context}\n\n{base_prompt}"

        # Create tasks for all roles
        async def execute_role(
            role: ModelRole, index: int
        ) -> tuple[int, str, Any | None, str | None]:
            """
            Execute a single role and return (index, role_name, response, error).

            Args:
                role: The ModelRole to execute
                index: Position in execution order

            Returns:
                Tuple of (index, role_name, response or None, error_message or None)
            """
            try:
                logger.info(
                    f"Executing role {index + 1}/{len(self.roles)}: {role.role} (model: {role.model})"
                )

                # Resolve provider for this role
                provider = self._resolve_provider(role.model)

                # Construct full prompt with role customizations
                full_prompt = role.get_full_prompt(full_base_prompt)

                # Create generation request with role-specific overrides
                request = GenerationRequest(
                    prompt=full_prompt,
                    system_prompt=role.system_prompt,
                    temperature=role.temperature,
                    max_tokens=role.max_tokens,
                )

                # Execute via provider
                response = await provider.generate(request)

                logger.info(
                    f"Role '{role.role}' completed successfully ({len(response.content)} chars)"
                )

                return (index, role.role, response, None)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Role '{role.role}' failed: {error_msg}", exc_info=True)
                return (index, role.role, None, error_msg)

        # Execute all roles in parallel
        tasks = [execute_role(role, idx) for idx, role in enumerate(self.roles)]
        results = await asyncio.gather(*tasks)

        # Process results and maintain execution order
        role_responses = []
        all_responses = []
        failed_roles = []
        execution_order = []

        # Sort by index to maintain execution order
        sorted_results = sorted(results, key=lambda x: x[0])

        for index, role_name, response, error in sorted_results:
            execution_order.append(role_name)

            if error is None:
                # Success
                role_responses.append((role_name, response))
                all_responses.append(response)
            else:
                # Failure
                failed_roles.append(role_name)

        # Build result
        result = OrchestrationResult(
            role_responses=role_responses,
            all_responses=all_responses,
            failed_roles=failed_roles,
            pattern_used=self.pattern,
            execution_order=execution_order,
            metadata={
                "total_roles": len(self.roles),
                "successful_roles": len(role_responses),
                "failed_roles": len(failed_roles),
            },
        )

        logger.info(
            f"Parallel orchestration complete: {len(role_responses)}/{len(self.roles)} roles succeeded"
        )

        return result

    async def synthesize(
        self,
        result: OrchestrationResult,
        strategy: SynthesisStrategy = SynthesisStrategy.CONCATENATE,
        synthesis_provider: Any | None = None,
        synthesis_prompt: str | None = None,
    ) -> OrchestrationResult:
        """
        Synthesize multiple role outputs into a unified result.

        Takes the raw role responses and combines them according to the specified
        strategy. Can be called manually after execute() or automatically via
        execute(synthesis_strategy=...).

        Args:
            result: OrchestrationResult from execute()
            strategy: How to combine the outputs (CONCATENATE, AI_SYNTHESIZE, STRUCTURED, NONE)
            synthesis_provider: Optional provider for AI synthesis (uses first role's provider if None)
            synthesis_prompt: Optional custom prompt for AI synthesis

        Returns:
            Updated OrchestrationResult with synthesized_output field populated

        Example:
            >>> result = await orchestrator.execute("Should we adopt TypeScript?")
            >>> synthesized = await orchestrator.synthesize(result, SynthesisStrategy.AI_SYNTHESIZE)
            >>> print(synthesized.synthesized_output)
        """
        if strategy == SynthesisStrategy.NONE:
            # No synthesis - return original result
            result.synthesis_strategy = strategy
            return result

        elif strategy == SynthesisStrategy.CONCATENATE:
            # Simple concatenation with role labels
            logger.info("Synthesizing with CONCATENATE strategy")

            parts = []
            for role_name, response in result.role_responses:
                parts.append(f"## {role_name.upper()}\n\n{response.content}")

            synthesized = "\n\n---\n\n".join(parts)
            result.synthesized_output = synthesized
            result.synthesis_strategy = strategy
            result.metadata["synthesis_method"] = "concatenate"

            logger.info(
                f"Concatenated {len(result.role_responses)} responses ({len(synthesized)} chars)"
            )

            return result

        elif strategy == SynthesisStrategy.STRUCTURED:
            # Build structured dict with role keys
            logger.info("Synthesizing with STRUCTURED strategy")

            structured = {}
            for role_name, response in result.role_responses:
                structured[role_name] = {
                    "content": response.content,
                    "model": response.model,
                    "usage": response.usage,
                }

            result.synthesized_output = structured
            result.synthesis_strategy = strategy
            result.metadata["synthesis_method"] = "structured"

            logger.info(f"Structured synthesis created with {len(structured)} roles")

            return result

        elif strategy == SynthesisStrategy.AI_SYNTHESIZE:
            # Use AI to synthesize responses
            logger.info("Synthesizing with AI_SYNTHESIZE strategy")

            # Determine which provider to use
            if synthesis_provider is None:
                if not result.role_responses:
                    raise ValueError("No role responses to synthesize")
                # Use first role's provider as default
                first_role = self.roles[0]
                synthesis_provider = self._resolve_provider(first_role.model)
                logger.debug(f"Using {first_role.model} provider for synthesis")

            # Build synthesis prompt
            if synthesis_prompt is None:
                synthesis_prompt = self._build_synthesis_prompt(result)

            try:
                # Import here to avoid circular dependency
                from ..providers import GenerationRequest

                # Create request for synthesis
                request = GenerationRequest(
                    prompt=synthesis_prompt,
                    system_prompt="You are an expert at synthesizing multiple perspectives into coherent, balanced analysis.",
                    temperature=0.7,
                )

                # Execute synthesis
                synthesis_response = await synthesis_provider.generate(request)

                result.synthesized_output = synthesis_response.content
                result.synthesis_strategy = strategy
                result.metadata["synthesis_method"] = "ai"
                result.metadata["synthesis_model"] = synthesis_response.model
                result.metadata["synthesis_usage"] = synthesis_response.usage

                logger.info(
                    f"AI synthesis completed ({len(synthesis_response.content)} chars)"
                )

                return result

            except Exception as e:
                logger.error(f"AI synthesis failed: {e}", exc_info=True)
                logger.warning("Falling back to CONCATENATE strategy")

                # Fallback to concatenation
                return await self.synthesize(result, SynthesisStrategy.CONCATENATE)

        else:
            raise ValueError(f"Unknown synthesis strategy: {strategy}")

    def _build_synthesis_prompt(self, result: OrchestrationResult) -> str:
        """
        Build a prompt for AI-powered synthesis of role responses.

        Args:
            result: OrchestrationResult containing role responses

        Returns:
            Synthesis prompt string
        """
        prompt_parts = [
            "You have received multiple perspectives on a question from different roles.",
            "Please synthesize these perspectives into a coherent, balanced response that:",
            "1. Integrates key insights from each perspective",
            "2. Identifies areas of agreement and disagreement",
            "3. Provides a nuanced final recommendation or conclusion",
            "",
            "Here are the perspectives:",
            "",
        ]

        for role_name, response in result.role_responses:
            prompt_parts.append(f"**{role_name.upper()}:**")
            prompt_parts.append(response.content)
            prompt_parts.append("")

        prompt_parts.append("Please synthesize these perspectives:")

        return "\n".join(prompt_parts)
