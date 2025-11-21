"""
Unit tests for role-based orchestration framework.

Tests verify that:
- ModelRole correctly validates input and constructs prompts
- OrchestrationResult properly stores execution metadata
- RoleOrchestrator handles initialization, provider resolution
- Sequential execution pattern works correctly
- Parallel execution pattern works correctly
- Synthesis strategies combine outputs appropriately
- Error handling and edge cases are managed properly
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from model_chorus.core.role_orchestration import (
    ModelRole,
    OrchestrationPattern,
    OrchestrationResult,
    RoleOrchestrator,
    SynthesisStrategy,
)


# Mock GenerationRequest and GenerationResponse for testing
@dataclass
class MockGenerationRequest:
    """Mock GenerationRequest for testing."""

    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class MockGenerationResponse:
    """Mock GenerationResponse for testing."""

    content: str
    model: str
    usage: dict[str, int] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = {"input_tokens": 100, "output_tokens": 50}


class MockProvider:
    """Mock provider for testing orchestration."""

    def __init__(
        self,
        model_name: str = "mock-model",
        fail: bool = False,
        response_content: str | None = None,
    ):
        """
        Initialize mock provider.

        Args:
            model_name: Name of the model this provider represents
            fail: If True, generate() will raise an exception
            response_content: Custom content for responses (default uses model_name)
        """
        self.model_name = model_name
        self.fail = fail
        self.response_content = response_content or f"Response from {model_name}"
        self.generate_called = False
        self.last_request = None

    async def generate(self, request: MockGenerationRequest) -> MockGenerationResponse:
        """
        Mock generate method.

        Args:
            request: Generation request

        Returns:
            Mock generation response

        Raises:
            Exception: If fail=True
        """
        self.generate_called = True
        self.last_request = request

        if self.fail:
            raise Exception(f"Mock failure from {self.model_name}")

        return MockGenerationResponse(
            content=self.response_content,
            model=self.model_name,
        )


class TestModelRole:
    """Test suite for ModelRole class."""

    def test_model_role_creation_minimal(self):
        """Test creating ModelRole with minimal required fields."""
        role = ModelRole(role="analyst", model="gpt-5")

        assert role.role == "analyst"
        assert role.model == "gpt-5"
        assert role.stance is None
        assert role.stance_prompt is None
        assert role.system_prompt is None
        assert role.temperature is None
        assert role.max_tokens is None
        assert role.metadata == {}

    def test_model_role_creation_full(self):
        """Test creating ModelRole with all fields."""
        metadata = {"priority": 1, "tags": ["debate"]}
        role = ModelRole(
            role="proponent",
            model="gpt-5",
            stance="for",
            stance_prompt="Argue in favor",
            system_prompt="You are a debater",
            temperature=0.8,
            max_tokens=4000,
            metadata=metadata,
        )

        assert role.role == "proponent"
        assert role.model == "gpt-5"
        assert role.stance == "for"
        assert role.stance_prompt == "Argue in favor"
        assert role.system_prompt == "You are a debater"
        assert role.temperature == 0.8
        assert role.max_tokens == 4000
        assert role.metadata == metadata

    def test_stance_validation_valid(self):
        """Test stance validation with valid values."""
        for stance in ["for", "against", "neutral", "FOR", "AGAINST", "NEUTRAL"]:
            role = ModelRole(role="test", model="gpt-5", stance=stance)
            assert role.stance == stance.lower()

    def test_stance_validation_invalid(self):
        """Test stance validation with invalid values."""
        with pytest.raises(ValueError, match="Stance must be one of"):
            ModelRole(role="test", model="gpt-5", stance="invalid")

    def test_temperature_validation_valid(self):
        """Test temperature validation with valid values."""
        for temp in [0.0, 0.5, 1.0]:
            role = ModelRole(role="test", model="gpt-5", temperature=temp)
            assert role.temperature == temp

    def test_temperature_validation_invalid(self):
        """Test temperature validation with invalid values."""
        with pytest.raises(ValueError):
            ModelRole(role="test", model="gpt-5", temperature=-0.1)

        with pytest.raises(ValueError):
            ModelRole(role="test", model="gpt-5", temperature=1.5)

    def test_max_tokens_validation_valid(self):
        """Test max_tokens validation with valid values."""
        role = ModelRole(role="test", model="gpt-5", max_tokens=1000)
        assert role.max_tokens == 1000

    def test_max_tokens_validation_invalid(self):
        """Test max_tokens validation with invalid values."""
        with pytest.raises(ValueError):
            ModelRole(role="test", model="gpt-5", max_tokens=0)

        with pytest.raises(ValueError):
            ModelRole(role="test", model="gpt-5", max_tokens=-100)

    def test_get_full_prompt_base_only(self):
        """Test prompt construction with only base prompt."""
        role = ModelRole(role="test", model="gpt-5")
        full_prompt = role.get_full_prompt("Base prompt")

        assert full_prompt == "Base prompt"

    def test_get_full_prompt_with_system(self):
        """Test prompt construction with system prompt."""
        role = ModelRole(
            role="test",
            model="gpt-5",
            system_prompt="You are an expert",
        )
        full_prompt = role.get_full_prompt("Base prompt")

        assert full_prompt == "You are an expert\n\nBase prompt"

    def test_get_full_prompt_with_stance(self):
        """Test prompt construction with stance prompt."""
        role = ModelRole(
            role="test",
            model="gpt-5",
            stance="for",
            stance_prompt="Argue in favor",
        )
        full_prompt = role.get_full_prompt("Base prompt")

        assert full_prompt == "Argue in favor\n\nBase prompt"

    def test_get_full_prompt_with_all(self):
        """Test prompt construction with all prompt types."""
        role = ModelRole(
            role="test",
            model="gpt-5",
            system_prompt="You are an expert",
            stance="for",
            stance_prompt="Argue in favor",
        )
        full_prompt = role.get_full_prompt("Base prompt")

        expected = "You are an expert\n\nArgue in favor\n\nBase prompt"
        assert full_prompt == expected

    def test_role_length_validation(self):
        """Test role name length validation."""
        # Valid: 1-100 characters
        ModelRole(role="a", model="gpt-5")
        ModelRole(role="a" * 100, model="gpt-5")

        # Invalid: empty string
        with pytest.raises(ValueError):
            ModelRole(role="", model="gpt-5")

        # Invalid: > 100 characters
        with pytest.raises(ValueError):
            ModelRole(role="a" * 101, model="gpt-5")

    def test_model_name_validation(self):
        """Test model name validation."""
        # Valid: non-empty
        ModelRole(role="test", model="gpt-5")

        # Invalid: empty string
        with pytest.raises(ValueError):
            ModelRole(role="test", model="")


class TestOrchestrationResult:
    """Test suite for OrchestrationResult dataclass."""

    def test_orchestration_result_creation_minimal(self):
        """Test creating OrchestrationResult with minimal fields."""
        result = OrchestrationResult()

        assert result.role_responses == []
        assert result.all_responses == []
        assert result.failed_roles == []
        assert result.pattern_used == OrchestrationPattern.SEQUENTIAL
        assert result.execution_order == []
        assert result.synthesized_output is None
        assert result.synthesis_strategy is None
        assert result.metadata == {}

    def test_orchestration_result_creation_full(self):
        """Test creating OrchestrationResult with all fields."""
        response = MockGenerationResponse(content="Test", model="gpt-5")
        result = OrchestrationResult(
            role_responses=[("analyst", response)],
            all_responses=[response],
            failed_roles=["critic"],
            pattern_used=OrchestrationPattern.PARALLEL,
            execution_order=["analyst", "critic"],
            synthesized_output="Synthesized content",
            synthesis_strategy=SynthesisStrategy.CONCATENATE,
            metadata={"total_roles": 2},
        )

        assert len(result.role_responses) == 1
        assert result.role_responses[0][0] == "analyst"
        assert len(result.all_responses) == 1
        assert result.failed_roles == ["critic"]
        assert result.pattern_used == OrchestrationPattern.PARALLEL
        assert result.execution_order == ["analyst", "critic"]
        assert result.synthesized_output == "Synthesized content"
        assert result.synthesis_strategy == SynthesisStrategy.CONCATENATE
        assert result.metadata["total_roles"] == 2


class TestRoleOrchestrator:
    """Test suite for RoleOrchestrator class."""

    @pytest.fixture
    def simple_roles(self):
        """Create a simple set of roles for testing."""
        return [
            ModelRole(role="analyst", model="gpt-5"),
            ModelRole(role="critic", model="gemini"),
        ]

    @pytest.fixture
    def simple_providers(self):
        """Create a simple provider map for testing."""
        return {
            "gpt-5": MockProvider("gpt-5", response_content="GPT-5 analysis"),
            "gemini": MockProvider("gemini", response_content="Gemini critique"),
        }

    def test_orchestrator_initialization_minimal(self, simple_roles, simple_providers):
        """Test orchestrator initialization with minimal parameters."""
        orchestrator = RoleOrchestrator(simple_roles, simple_providers)

        assert orchestrator.roles == simple_roles
        assert orchestrator.provider_map == simple_providers
        assert orchestrator.pattern == OrchestrationPattern.SEQUENTIAL
        assert orchestrator.default_timeout == 120.0

    def test_orchestrator_initialization_custom(self, simple_roles, simple_providers):
        """Test orchestrator initialization with custom parameters."""
        orchestrator = RoleOrchestrator(
            simple_roles,
            simple_providers,
            pattern=OrchestrationPattern.PARALLEL,
            default_timeout=60.0,
        )

        assert orchestrator.pattern == OrchestrationPattern.PARALLEL
        assert orchestrator.default_timeout == 60.0

    def test_orchestrator_initialization_empty_roles(self, simple_providers):
        """Test orchestrator initialization with empty roles list."""
        with pytest.raises(ValueError, match="At least one role is required"):
            RoleOrchestrator([], simple_providers)

    def test_orchestrator_initialization_unsupported_pattern(
        self, simple_roles, simple_providers
    ):
        """Test orchestrator initialization with unsupported pattern."""
        with pytest.raises(
            ValueError, match="Only SEQUENTIAL and PARALLEL patterns are supported"
        ):
            RoleOrchestrator(
                simple_roles,
                simple_providers,
                pattern=OrchestrationPattern.HYBRID,
            )

    def test_resolve_provider_exact_match(self, simple_roles, simple_providers):
        """Test provider resolution with exact match."""
        orchestrator = RoleOrchestrator(simple_roles, simple_providers)

        provider = orchestrator._resolve_provider("gpt-5")
        assert provider.model_name == "gpt-5"

    def test_resolve_provider_case_variation(self, simple_roles):
        """Test provider resolution with case variation."""
        providers = {"gpt5": MockProvider("gpt5")}
        orchestrator = RoleOrchestrator(simple_roles, providers)

        # Should match via lowercase variation
        provider = orchestrator._resolve_provider("GPT5")
        assert provider.model_name == "gpt5"

    def test_resolve_provider_hyphen_variation(self, simple_roles):
        """Test provider resolution with hyphen variation."""
        providers = {"gpt5": MockProvider("gpt5")}
        orchestrator = RoleOrchestrator(simple_roles, providers)

        # Should match by removing hyphens
        provider = orchestrator._resolve_provider("gpt-5")
        assert provider.model_name == "gpt5"

    def test_resolve_provider_not_found(self, simple_roles, simple_providers):
        """Test provider resolution with unknown model."""
        orchestrator = RoleOrchestrator(simple_roles, simple_providers)

        with pytest.raises(ValueError, match="No provider found for model"):
            orchestrator._resolve_provider("unknown-model")

    @pytest.mark.asyncio
    async def test_execute_sequential_success(self, simple_roles, simple_providers):
        """Test successful sequential execution."""
        # Patch GenerationRequest import
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(
                simple_roles,
                simple_providers,
                pattern=OrchestrationPattern.SEQUENTIAL,
            )

            result = await orchestrator.execute("Test prompt")

            # Verify result structure
            assert len(result.role_responses) == 2
            assert result.role_responses[0][0] == "analyst"
            assert result.role_responses[1][0] == "critic"
            assert result.role_responses[0][1].content == "GPT-5 analysis"
            assert result.role_responses[1][1].content == "Gemini critique"

            assert len(result.all_responses) == 2
            assert result.failed_roles == []
            assert result.pattern_used == OrchestrationPattern.SEQUENTIAL
            assert result.execution_order == ["analyst", "critic"]

            assert result.metadata["total_roles"] == 2
            assert result.metadata["successful_roles"] == 2
            assert result.metadata["failed_roles"] == 0

    @pytest.mark.asyncio
    async def test_execute_sequential_with_context(
        self, simple_roles, simple_providers
    ):
        """Test sequential execution with context."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute(
                "Test prompt",
                context="Previous context",
            )

            # Check that first provider received context-enhanced prompt
            first_provider = simple_providers["gpt-5"]
            assert "Previous context" in first_provider.last_request.prompt
            assert "Test prompt" in first_provider.last_request.prompt

    @pytest.mark.asyncio
    async def test_execute_sequential_partial_failure(self, simple_roles):
        """Test sequential execution with partial failure."""
        providers = {
            "gpt-5": MockProvider("gpt-5"),
            "gemini": MockProvider("gemini", fail=True),
        }

        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, providers)

            result = await orchestrator.execute("Test prompt")

            # First role should succeed, second should fail
            assert len(result.role_responses) == 1
            assert result.role_responses[0][0] == "analyst"
            assert len(result.failed_roles) == 1
            assert "critic" in result.failed_roles

            assert result.metadata["successful_roles"] == 1
            assert result.metadata["failed_roles"] == 1

    @pytest.mark.asyncio
    async def test_execute_parallel_success(self, simple_roles, simple_providers):
        """Test successful parallel execution."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(
                simple_roles,
                simple_providers,
                pattern=OrchestrationPattern.PARALLEL,
            )

            result = await orchestrator.execute("Test prompt")

            # Verify result structure (order should be maintained)
            assert len(result.role_responses) == 2
            assert result.role_responses[0][0] == "analyst"
            assert result.role_responses[1][0] == "critic"

            assert len(result.all_responses) == 2
            assert result.failed_roles == []
            assert result.pattern_used == OrchestrationPattern.PARALLEL
            assert result.execution_order == ["analyst", "critic"]

            # Both providers should have been called
            assert simple_providers["gpt-5"].generate_called
            assert simple_providers["gemini"].generate_called

    @pytest.mark.asyncio
    async def test_execute_parallel_partial_failure(self, simple_roles):
        """Test parallel execution with partial failure."""
        providers = {
            "gpt-5": MockProvider("gpt-5"),
            "gemini": MockProvider("gemini", fail=True),
        }

        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(
                simple_roles,
                providers,
                pattern=OrchestrationPattern.PARALLEL,
            )

            result = await orchestrator.execute("Test prompt")

            # First role should succeed, second should fail
            assert len(result.role_responses) == 1
            assert result.role_responses[0][0] == "analyst"
            assert len(result.failed_roles) == 1
            assert "critic" in result.failed_roles

    @pytest.mark.asyncio
    async def test_execute_parallel_maintains_order(self, simple_providers):
        """Test that parallel execution maintains role order in results."""
        # Create roles with specific order
        roles = [
            ModelRole(role="role1", model="gpt-5"),
            ModelRole(role="role2", model="gemini"),
            ModelRole(role="role3", model="gpt-5"),
        ]

        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(
                roles,
                simple_providers,
                pattern=OrchestrationPattern.PARALLEL,
            )

            result = await orchestrator.execute("Test prompt")

            # Verify order is maintained despite parallel execution
            assert result.execution_order == ["role1", "role2", "role3"]
            assert result.role_responses[0][0] == "role1"
            assert result.role_responses[1][0] == "role2"
            assert result.role_responses[2][0] == "role3"

    @pytest.mark.asyncio
    async def test_synthesize_none_strategy(self, simple_roles, simple_providers):
        """Test synthesis with NONE strategy."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")
            synthesized = await orchestrator.synthesize(result, SynthesisStrategy.NONE)

            assert synthesized.synthesis_strategy == SynthesisStrategy.NONE
            assert synthesized.synthesized_output is None

    @pytest.mark.asyncio
    async def test_synthesize_concatenate_strategy(
        self, simple_roles, simple_providers
    ):
        """Test synthesis with CONCATENATE strategy."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")
            synthesized = await orchestrator.synthesize(
                result, SynthesisStrategy.CONCATENATE
            )

            assert synthesized.synthesis_strategy == SynthesisStrategy.CONCATENATE
            assert "## ANALYST" in synthesized.synthesized_output
            assert "## CRITIC" in synthesized.synthesized_output
            assert "GPT-5 analysis" in synthesized.synthesized_output
            assert "Gemini critique" in synthesized.synthesized_output
            assert synthesized.metadata["synthesis_method"] == "concatenate"

    @pytest.mark.asyncio
    async def test_synthesize_structured_strategy(self, simple_roles, simple_providers):
        """Test synthesis with STRUCTURED strategy."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")
            synthesized = await orchestrator.synthesize(
                result, SynthesisStrategy.STRUCTURED
            )

            assert synthesized.synthesis_strategy == SynthesisStrategy.STRUCTURED
            assert isinstance(synthesized.synthesized_output, dict)
            assert "analyst" in synthesized.synthesized_output
            assert "critic" in synthesized.synthesized_output
            assert (
                synthesized.synthesized_output["analyst"]["content"] == "GPT-5 analysis"
            )
            assert (
                synthesized.synthesized_output["critic"]["content"] == "Gemini critique"
            )
            assert synthesized.metadata["synthesis_method"] == "structured"

    @pytest.mark.asyncio
    async def test_synthesize_ai_strategy_default_provider(
        self, simple_roles, simple_providers
    ):
        """Test synthesis with AI_SYNTHESIZE strategy using default provider."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")
            synthesized = await orchestrator.synthesize(
                result, SynthesisStrategy.AI_SYNTHESIZE
            )

            assert synthesized.synthesis_strategy == SynthesisStrategy.AI_SYNTHESIZE
            assert synthesized.synthesized_output is not None
            assert synthesized.metadata["synthesis_method"] == "ai"
            assert "synthesis_model" in synthesized.metadata

    @pytest.mark.asyncio
    async def test_synthesize_ai_strategy_custom_provider(
        self, simple_roles, simple_providers
    ):
        """Test synthesis with AI_SYNTHESIZE strategy using custom provider."""
        synthesis_provider = MockProvider(
            "synthesis-model", response_content="Synthesized result"
        )

        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")
            synthesized = await orchestrator.synthesize(
                result,
                SynthesisStrategy.AI_SYNTHESIZE,
                synthesis_provider=synthesis_provider,
            )

            assert synthesized.synthesized_output == "Synthesized result"
            assert synthesized.metadata["synthesis_model"] == "synthesis-model"

    @pytest.mark.asyncio
    async def test_synthesize_ai_strategy_custom_prompt(
        self, simple_roles, simple_providers
    ):
        """Test synthesis with AI_SYNTHESIZE strategy using custom prompt."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")

            custom_prompt = "Custom synthesis prompt"
            synthesized = await orchestrator.synthesize(
                result,
                SynthesisStrategy.AI_SYNTHESIZE,
                synthesis_prompt=custom_prompt,
            )

            # Verify custom prompt was used
            first_provider = simple_providers["gpt-5"]
            assert custom_prompt in first_provider.last_request.prompt

    @pytest.mark.asyncio
    async def test_synthesize_ai_strategy_fallback_on_failure(self, simple_roles):
        """Test synthesis with AI_SYNTHESIZE falls back to CONCATENATE on failure."""
        providers = {
            "gpt-5": MockProvider("gpt-5", fail=True),  # Will fail during synthesis
            "gemini": MockProvider("gemini"),
        }

        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            # First execute successfully
            orchestrator = RoleOrchestrator(simple_roles, providers)
            providers["gpt-5"].fail = False  # Allow initial execution to succeed

            result = await orchestrator.execute("Test prompt")

            # Now fail during synthesis
            providers["gpt-5"].fail = True
            synthesized = await orchestrator.synthesize(
                result, SynthesisStrategy.AI_SYNTHESIZE
            )

            # Should fall back to CONCATENATE
            assert synthesized.synthesis_strategy == SynthesisStrategy.CONCATENATE
            assert "## ANALYST" in synthesized.synthesized_output

    @pytest.mark.asyncio
    async def test_synthesize_ai_strategy_no_responses(
        self, simple_roles, simple_providers
    ):
        """Test synthesis with AI_SYNTHESIZE fails when no responses available."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            # Create empty result
            result = OrchestrationResult()

            with pytest.raises(ValueError, match="No role responses to synthesize"):
                await orchestrator.synthesize(result, SynthesisStrategy.AI_SYNTHESIZE)

    @pytest.mark.asyncio
    async def test_synthesize_unknown_strategy(self, simple_roles, simple_providers):
        """Test synthesis with unknown strategy raises error."""
        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(simple_roles, simple_providers)

            result = await orchestrator.execute("Test prompt")

            with pytest.raises(ValueError, match="Unknown synthesis strategy"):
                await orchestrator.synthesize(result, "invalid_strategy")

    @pytest.mark.asyncio
    async def test_role_prompt_customization(self, simple_providers):
        """Test that role-specific prompts are properly customized."""
        roles = [
            ModelRole(
                role="proponent",
                model="gpt-5",
                stance="for",
                stance_prompt="Argue FOR the proposal",
                system_prompt="You are a proponent",
                temperature=0.8,
                max_tokens=2000,
            ),
            ModelRole(
                role="opponent",
                model="gemini",
                stance="against",
                stance_prompt="Argue AGAINST the proposal",
                system_prompt="You are an opponent",
                temperature=0.3,
                max_tokens=1500,
            ),
        ]

        with patch("model_chorus.providers.GenerationRequest", MockGenerationRequest):
            orchestrator = RoleOrchestrator(roles, simple_providers)

            result = await orchestrator.execute("Should we adopt this?")

            # Check first provider received customized prompt
            gpt5_request = simple_providers["gpt-5"].last_request
            assert "You are a proponent" in gpt5_request.prompt
            assert "Argue FOR the proposal" in gpt5_request.prompt
            assert "Should we adopt this?" in gpt5_request.prompt
            assert gpt5_request.temperature == 0.8
            assert gpt5_request.max_tokens == 2000

            # Check second provider received customized prompt
            gemini_request = simple_providers["gemini"].last_request
            assert "You are an opponent" in gemini_request.prompt
            assert "Argue AGAINST the proposal" in gemini_request.prompt
            assert gemini_request.temperature == 0.3
            assert gemini_request.max_tokens == 1500

    def test_build_synthesis_prompt(self, simple_roles, simple_providers):
        """Test synthesis prompt construction."""
        orchestrator = RoleOrchestrator(simple_roles, simple_providers)

        # Create mock result
        result = OrchestrationResult(
            role_responses=[
                ("analyst", MockGenerationResponse("Analysis content", "gpt-5")),
                ("critic", MockGenerationResponse("Critique content", "gemini")),
            ]
        )

        prompt = orchestrator._build_synthesis_prompt(result)

        # Verify prompt structure
        assert "multiple perspectives" in prompt.lower()
        assert "**ANALYST:**" in prompt
        assert "Analysis content" in prompt
        assert "**CRITIC:**" in prompt
        assert "Critique content" in prompt
        assert "synthesize these perspectives" in prompt.lower()


class TestEnums:
    """Test suite for enum classes."""

    def test_orchestration_pattern_values(self):
        """Test OrchestrationPattern enum values."""
        assert OrchestrationPattern.SEQUENTIAL.value == "sequential"
        assert OrchestrationPattern.PARALLEL.value == "parallel"
        assert OrchestrationPattern.HYBRID.value == "hybrid"

    def test_synthesis_strategy_values(self):
        """Test SynthesisStrategy enum values."""
        assert SynthesisStrategy.NONE.value == "none"
        assert SynthesisStrategy.CONCATENATE.value == "concatenate"
        assert SynthesisStrategy.AI_SYNTHESIZE.value == "ai_synthesize"
        assert SynthesisStrategy.STRUCTURED.value == "structured"
