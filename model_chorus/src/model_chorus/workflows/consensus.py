"""
Consensus workflow for multi-model coordination.

This module implements the ConsensusWorkflow which enables querying multiple
AI models simultaneously and synthesizing their responses for improved accuracy
and reliability.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, replace
from enum import Enum

from ..providers import (
    ModelProvider,
    GenerationRequest,
    GenerationResponse,
)
from ..core.progress import emit_workflow_start, emit_provider_start, emit_provider_complete, emit_workflow_complete

logger = logging.getLogger(__name__)


class ConsensusStrategy(Enum):
    """Strategy for reaching consensus among multiple model responses."""

    MAJORITY = "majority"  # Use most common response
    WEIGHTED = "weighted"  # Weight by model confidence/quality
    SYNTHESIZE = "synthesize"  # Combine all responses into synthesis
    FIRST_VALID = "first_valid"  # Use first successful response
    ALL_RESPONSES = "all_responses"  # Return all responses without synthesis


@dataclass
class ProviderConfig:
    """Configuration for a provider in the consensus workflow."""

    provider: ModelProvider
    weight: float = 1.0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result from a consensus workflow execution."""

    consensus_response: Optional[str] = None
    all_responses: List[GenerationResponse] = field(default_factory=list)
    provider_results: Dict[str, GenerationResponse] = field(default_factory=dict)
    failed_providers: List[str] = field(default_factory=list)
    strategy_used: ConsensusStrategy = ConsensusStrategy.ALL_RESPONSES
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusWorkflow:
    """
    Workflow for coordinating multiple AI models to reach consensus.

    This workflow allows querying multiple models simultaneously and applying
    various strategies to synthesize or select from their responses.

    Key features:
    - Parallel execution across multiple providers
    - Configurable consensus strategies
    - Error handling and fallback mechanisms
    - Response weighting and prioritization
    - Timeout management per provider

    Example:
        >>> from model_chorus.providers import ClaudeProvider, GeminiProvider
        >>> from model_chorus.workflows import ConsensusWorkflow
        >>>
        >>> # Create providers
        >>> claude = ClaudeProvider()
        >>> gemini = GeminiProvider()
        >>>
        >>> # Create workflow with providers
        >>> workflow = ConsensusWorkflow([claude, gemini])
        >>>
        >>> # Execute consensus query
        >>> request = GenerationRequest(prompt="Explain quantum computing")
        >>> result = await workflow.execute(request)
        >>>
        >>> # Access consensus or individual responses
        >>> print(result.consensus_response)
        >>> for provider, response in result.provider_results.items():
        ...     print(f"{provider}: {response.content}")
    """

    def __init__(
        self,
        providers: List[ModelProvider],
        strategy: ConsensusStrategy = ConsensusStrategy.ALL_RESPONSES,
        default_timeout: float = 120.0,
        num_to_consult: Optional[int] = None,
    ):
        """
        Initialize the consensus workflow.

        Args:
            providers: List of ModelProvider instances in priority order
            strategy: Strategy for reaching consensus (default: ALL_RESPONSES)
            default_timeout: Default timeout per provider in seconds (default: 120)
            num_to_consult: Number of successful responses required (default: use all providers)
        """
        if not providers:
            raise ValueError("At least one provider must be specified")

        self.provider_configs = [
            ProviderConfig(provider=p, timeout=default_timeout)
            for p in providers
        ]
        self.strategy = strategy
        self.default_timeout = default_timeout
        self.num_to_consult = num_to_consult if num_to_consult is not None else len(providers)

    def add_provider(
        self,
        provider: ModelProvider,
        weight: float = 1.0,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Add a provider to the consensus workflow.

        Args:
            provider: ModelProvider instance to add
            weight: Weight for this provider in consensus (default: 1.0)
            timeout: Custom timeout for this provider (default: use workflow default)
        """
        config = ProviderConfig(
            provider=provider,
            weight=weight,
            timeout=timeout or self.default_timeout,
        )
        self.provider_configs.append(config)

    async def _execute_provider(
        self,
        config: ProviderConfig,
        request: GenerationRequest,
    ) -> tuple[str, Optional[GenerationResponse], Optional[Exception]]:
        """
        Execute a single provider with timeout and error handling.

        Args:
            config: Provider configuration
            request: Generation request

        Returns:
            Tuple of (provider_name, response, error)
        """
        provider_name = config.provider.provider_name

        try:
            logger.info(f"Executing provider: {provider_name}")
            emit_provider_start(provider_name)

            base_metadata = dict(request.metadata) if request.metadata else {}
            provider_metadata = config.metadata or {}
            merged_metadata = (
                {**provider_metadata, **base_metadata}
                if provider_metadata
                else base_metadata
            )
            # Clone the request per provider so model overrides and future metadata
            # customizations apply without mutating the shared request object.
            provider_request = replace(request, metadata=merged_metadata)

            # Execute with timeout
            response = await asyncio.wait_for(
                config.provider.generate(provider_request),
                timeout=config.timeout,
            )

            logger.info(
                f"Provider {provider_name} completed successfully: "
                f"{len(response.content)} chars"
            )
            emit_provider_complete(provider_name)
            return provider_name, response, None

        except asyncio.TimeoutError:
            error = TimeoutError(
                f"Provider {provider_name} timed out after {config.timeout}s"
            )
            logger.warning(str(error))
            return provider_name, None, error

        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            return provider_name, None, e

    async def execute(
        self,
        request: GenerationRequest,
        strategy: Optional[ConsensusStrategy] = None,
    ) -> ConsensusResult:
        """
        Execute the consensus workflow with dynamic fallback.

        Tries providers in priority order until num_to_consult successful responses
        are obtained. If a provider fails, automatically tries the next provider
        in the priority list.

        Args:
            request: GenerationRequest to send to providers
            strategy: Override the default consensus strategy (optional)

        Returns:
            ConsensusResult with consensus response and individual results

        Raises:
            RuntimeError: If unable to obtain num_to_consult successful responses
        """
        strategy = strategy or self.strategy

        logger.info(
            f"Starting consensus workflow: need {self.num_to_consult} successful responses "
            f"from {len(self.provider_configs)} available providers, strategy: {strategy.value}"
        )

        # Emit workflow start
        emit_workflow_start("consensus", "10-30s")

        # Track successful and failed providers
        provider_results = {}
        all_responses = []
        failed_providers = []
        provider_index = 0

        # Try providers until we have enough successful responses or run out
        while len(all_responses) < self.num_to_consult and provider_index < len(self.provider_configs):
            # Calculate how many more we need
            needed = self.num_to_consult - len(all_responses)
            # Calculate how many providers we can try in this batch
            available = len(self.provider_configs) - provider_index
            batch_size = min(needed, available)

            # Get the next batch of providers to try
            batch_configs = self.provider_configs[provider_index:provider_index + batch_size]

            logger.info(
                f"Trying batch of {batch_size} providers (already have {len(all_responses)}/{self.num_to_consult})"
            )

            # Execute batch in parallel
            tasks = [
                self._execute_provider(config, request)
                for config in batch_configs
            ]

            results = await asyncio.gather(*tasks)

            # Process results
            for provider_name, response, error in results:
                if response is not None:
                    provider_results[provider_name] = response
                    all_responses.append(response)
                    logger.info(f"Provider {provider_name} succeeded ({len(all_responses)}/{self.num_to_consult})")
                else:
                    failed_providers.append(provider_name)
                    logger.warning(f"Provider {provider_name} failed, will try fallback if available")

            provider_index += batch_size

        # Check if we got enough successful responses
        if len(all_responses) < self.num_to_consult:
            error_msg = (
                f"Consensus workflow failed: only {len(all_responses)}/{self.num_to_consult} "
                f"providers succeeded. Tried {len(self.provider_configs)} providers total."
            )
            logger.error(error_msg)
            emit_workflow_complete("consensus")
            raise RuntimeError(error_msg)

        # Apply consensus strategy
        consensus_response = self._apply_strategy(
            strategy, all_responses, provider_results
        )

        # Build metadata
        metadata = {
            "total_providers": len(self.provider_configs),
            "providers_tried": provider_index,
            "successful_providers": len(all_responses),
            "failed_providers": len(failed_providers),
            "num_to_consult": self.num_to_consult,
            "strategy": strategy.value,
        }

        result = ConsensusResult(
            consensus_response=consensus_response,
            all_responses=all_responses,
            provider_results=provider_results,
            failed_providers=failed_providers,
            strategy_used=strategy,
            metadata=metadata,
        )

        logger.info(
            f"Consensus workflow completed: {len(all_responses)}/{self.num_to_consult} "
            f"providers succeeded (tried {provider_index} total)"
        )

        # Emit workflow complete
        emit_workflow_complete("consensus")

        return result

    def _apply_strategy(
        self,
        strategy: ConsensusStrategy,
        responses: List[GenerationResponse],
        provider_results: Dict[str, GenerationResponse],
    ) -> Optional[str]:
        """
        Apply consensus strategy to synthesize responses.

        Args:
            strategy: Consensus strategy to use
            responses: List of successful responses
            provider_results: Dict of provider name to response

        Returns:
            Consensus response string, or None if strategy doesn't produce one
        """
        if not responses:
            logger.warning("No successful responses to apply consensus strategy")
            return None

        if strategy == ConsensusStrategy.ALL_RESPONSES:
            # Return all responses concatenated with provider labels
            parts = []
            for provider_name, response in provider_results.items():
                parts.append(f"## {provider_name.upper()}\n\n{response.content}")
            return "\n\n---\n\n".join(parts)

        elif strategy == ConsensusStrategy.FIRST_VALID:
            # Return first successful response
            return responses[0].content if responses else None

        elif strategy == ConsensusStrategy.MAJORITY:
            # Find most common response (simple implementation)
            # For real use, would need more sophisticated comparison
            response_counts: Dict[str, int] = {}
            for response in responses:
                content = response.content.strip()
                response_counts[content] = response_counts.get(content, 0) + 1

            if response_counts:
                most_common = max(response_counts.items(), key=lambda x: x[1])
                return most_common[0]
            return None

        elif strategy == ConsensusStrategy.WEIGHTED:
            # For weighted, return longest response as simple heuristic
            # Real implementation would use provider weights
            longest = max(responses, key=lambda r: len(r.content))
            return longest.content

        elif strategy == ConsensusStrategy.SYNTHESIZE:
            # For synthesis, concatenate with clear attribution
            synthesis_parts = [
                "# Synthesized Response from Multiple Models\n"
            ]

            for i, (provider_name, response) in enumerate(
                provider_results.items(), 1
            ):
                synthesis_parts.append(
                    f"\n## Perspective {i}: {provider_name}\n\n{response.content}"
                )

            synthesis_parts.append(
                "\n\n---\n\n*Note: This is a synthesis of responses from "
                f"{len(responses)} different models. Consider cross-referencing "
                "for accuracy.*"
            )

            return "\n".join(synthesis_parts)

        else:
            logger.warning(f"Unknown consensus strategy: {strategy}")
            return None

    def get_provider_count(self) -> int:
        """Get the number of configured providers."""
        return len(self.provider_configs)

    def get_providers(self) -> List[ModelProvider]:
        """Get list of all configured providers."""
        return [config.provider for config in self.provider_configs]

    def set_strategy(self, strategy: ConsensusStrategy) -> None:
        """
        Set the default consensus strategy.

        Args:
            strategy: New consensus strategy to use
        """
        self.strategy = strategy
        logger.info(f"Consensus strategy updated to: {strategy.value}")

    def __repr__(self) -> str:
        """String representation of the workflow."""
        provider_names = [c.provider.provider_name for c in self.provider_configs]
        return (
            f"ConsensusWorkflow("
            f"providers={provider_names}, "
            f"strategy={self.strategy.value}"
            f")"
        )
