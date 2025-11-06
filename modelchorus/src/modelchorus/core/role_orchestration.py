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
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, field_validator

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

    stance: Optional[str] = Field(
        default=None,
        description="Optional stance for debate-style workflows ('for', 'against', 'neutral')",
    )

    stance_prompt: Optional[str] = Field(
        default=None,
        description="Optional additional prompt text to reinforce the stance",
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system-level prompt for this role",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Optional temperature override for this role (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Optional max tokens override for this role",
        gt=0,
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this role (tags, priority, constraints, etc.)",
    )

    @field_validator("stance")
    @classmethod
    def validate_stance(cls, v: Optional[str]) -> Optional[str]:
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
            raise ValueError(
                f"Stance must be one of {allowed_stances}, got '{v}'"
            )

        return stance_lower

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
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

    Contains responses from all models, execution metadata, and error information.
    Used by RoleOrchestrator to return structured results after executing
    a multi-model workflow.

    Attributes:
        role_responses: List of (role_name, response) tuples in execution order
        all_responses: List of all GenerationResponse objects
        failed_roles: List of role names that failed to execute
        pattern_used: Orchestration pattern that was executed
        execution_order: List of role names in the order they were executed
        metadata: Additional execution metadata (timing, context, etc.)

    Example:
        >>> result = OrchestrationResult(
        ...     role_responses=[
        ...         ("proponent", GenerationResponse(content="Argument FOR...", model="gpt-5")),
        ...         ("critic", GenerationResponse(content="Argument AGAINST...", model="gemini-2.5-pro")),
        ...     ],
        ...     pattern_used=OrchestrationPattern.SEQUENTIAL,
        ...     execution_order=["proponent", "critic"]
        ... )
    """

    role_responses: List[tuple[str, Any]] = field(default_factory=list)
    all_responses: List[Any] = field(default_factory=list)
    failed_roles: List[str] = field(default_factory=list)
    pattern_used: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL
    execution_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        >>> from modelchorus.core.role_orchestration import ModelRole, RoleOrchestrator
        >>> from modelchorus.providers import ClaudeProvider, GeminiProvider
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
        roles: List[ModelRole],
        provider_map: Dict[str, Any],
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

        if pattern not in (OrchestrationPattern.SEQUENTIAL, OrchestrationPattern.PARALLEL):
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
        context: Optional[str] = None,
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
        context: Optional[str] = None,
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

        logger.info(
            f"Starting sequential execution of {len(self.roles)} roles"
        )

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
                logger.error(
                    f"Role '{role.role}' failed: {e}",
                    exc_info=True
                )
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
        context: Optional[str] = None,
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

        logger.info(
            f"Starting parallel execution of {len(self.roles)} roles"
        )

        # Build context-enhanced prompt if context provided
        full_base_prompt = base_prompt
        if context:
            full_base_prompt = f"{context}\n\n{base_prompt}"

        # Create tasks for all roles
        async def execute_role(role: ModelRole, index: int) -> tuple[int, Optional[str], Optional[Any], Optional[str]]:
            """
            Execute a single role and return (index, role_name, response, error).

            Args:
                role: The ModelRole to execute
                index: Position in execution order

            Returns:
                Tuple of (index, role_name, response or None, error_message or None)
            """
            try:
                logger.info(f"Executing role {index + 1}/{len(self.roles)}: {role.role} (model: {role.model})")

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

                logger.info(f"Role '{role.role}' completed successfully ({len(response.content)} chars)")

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
