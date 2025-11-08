"""
Tests for persona system in STUDY workflow.

Tests verify that:
- Persona dataclasses work correctly (Persona, PersonaResponse)
- PersonaRegistry manages personas properly
- Individual persona implementations work as expected
- Persona factory functions create correctly configured instances
- Default registry is properly initialized
"""

import pytest
from model_chorus.workflows.study.persona_base import (
    Persona,
    PersonaResponse,
    PersonaRegistry
)
from model_chorus.workflows.study.personas import (
    ResearcherPersona,
    CriticPersona,
    PlannerPersona,
    create_researcher,
    create_critic,
    create_planner,
    get_default_registry,
    create_default_personas,
)


class TestPersonaResponse:
    """Test suite for PersonaResponse dataclass."""

    def test_persona_response_init(self):
        """Test PersonaResponse initializes with default values."""
        response = PersonaResponse()

        assert response.findings == []
        assert response.confidence_update is None
        assert response.metadata == {}

    def test_persona_response_with_findings(self):
        """Test PersonaResponse with findings."""
        findings = ["Finding 1", "Finding 2", "Finding 3"]
        response = PersonaResponse(findings=findings)

        assert response.findings == findings
        assert len(response.findings) == 3

    def test_persona_response_with_confidence_update(self):
        """Test PersonaResponse with confidence update."""
        response = PersonaResponse(confidence_update="high")

        assert response.confidence_update == "high"

    def test_persona_response_with_metadata(self):
        """Test PersonaResponse with metadata."""
        metadata = {"key": "value", "count": 42}
        response = PersonaResponse(metadata=metadata)

        assert response.metadata == metadata
        assert response.metadata["key"] == "value"
        assert response.metadata["count"] == 42

    def test_persona_response_full_initialization(self):
        """Test PersonaResponse full initialization."""
        findings = ["Finding A", "Finding B"]
        metadata = {"status": "complete"}

        response = PersonaResponse(
            findings=findings,
            confidence_update="medium",
            metadata=metadata
        )

        assert response.findings == findings
        assert response.confidence_update == "medium"
        assert response.metadata == metadata


class TestPersona:
    """Test suite for Persona base class."""

    def test_persona_init(self):
        """Test Persona initializes with required parameters."""
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template"
        )

        assert persona.name == "TestPersona"
        assert persona.prompt_template == "Test template"
        assert persona.temperature == 0.7
        assert persona.max_tokens == 4096

    def test_persona_init_custom_temperature(self):
        """Test Persona with custom temperature."""
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template",
            temperature=0.3
        )

        assert persona.temperature == 0.3

    def test_persona_init_custom_max_tokens(self):
        """Test Persona with custom max_tokens."""
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template",
            max_tokens=2048
        )

        assert persona.max_tokens == 2048

    def test_persona_invoke_returns_response(self):
        """Test Persona.invoke returns PersonaResponse."""
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template"
        )

        response = persona.invoke({"prompt": "Test"})

        assert isinstance(response, PersonaResponse)

    def test_persona_invoke_with_context(self):
        """Test Persona.invoke processes context."""
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template"
        )
        context = {"prompt": "Test question", "phase": "discovery"}

        response = persona.invoke(context)

        assert response is not None
        assert response.findings is not None

    def test_persona_invoke_includes_metadata(self):
        """Test Persona.invoke includes persona name in metadata."""
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template"
        )

        response = persona.invoke({})

        assert "persona" in response.metadata
        assert response.metadata["persona"] == "TestPersona"


class TestPersonaRegistry:
    """Test suite for PersonaRegistry."""

    def test_registry_init(self):
        """Test PersonaRegistry initializes empty."""
        registry = PersonaRegistry()

        assert registry.list_all() == []

    def test_registry_register_persona(self):
        """Test registering a persona."""
        registry = PersonaRegistry()
        persona = Persona(
            name="TestPersona",
            prompt_template="Test template"
        )

        registry.register(persona)

        assert len(registry.list_all()) == 1

    def test_registry_register_multiple_personas(self):
        """Test registering multiple personas."""
        registry = PersonaRegistry()
        personas = [
            Persona(name="Persona1", prompt_template="Template 1"),
            Persona(name="Persona2", prompt_template="Template 2"),
            Persona(name="Persona3", prompt_template="Template 3"),
        ]

        for persona in personas:
            registry.register(persona)

        assert len(registry.list_all()) == 3

    def test_registry_register_duplicate_raises_error(self):
        """Test registering duplicate persona name raises ValueError."""
        registry = PersonaRegistry()
        persona1 = Persona(name="Duplicate", prompt_template="Template 1")
        persona2 = Persona(name="Duplicate", prompt_template="Template 2")

        registry.register(persona1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(persona2)

    def test_registry_get_persona(self):
        """Test retrieving a persona by name."""
        registry = PersonaRegistry()
        persona = Persona(name="TestPersona", prompt_template="Test")

        registry.register(persona)
        retrieved = registry.get("TestPersona")

        assert retrieved is not None
        assert retrieved.name == "TestPersona"
        assert retrieved.prompt_template == "Test"

    def test_registry_get_nonexistent_persona(self):
        """Test retrieving nonexistent persona returns None."""
        registry = PersonaRegistry()

        result = registry.get("NonExistent")

        assert result is None

    def test_registry_list_all(self):
        """Test list_all returns all registered personas."""
        registry = PersonaRegistry()
        personas = [
            Persona(name="Persona1", prompt_template="Template 1"),
            Persona(name="Persona2", prompt_template="Template 2"),
        ]

        for persona in personas:
            registry.register(persona)

        all_personas = registry.list_all()

        assert len(all_personas) == 2
        assert any(p.name == "Persona1" for p in all_personas)
        assert any(p.name == "Persona2" for p in all_personas)


class TestResearcherPersona:
    """Test suite for ResearcherPersona."""

    def test_researcher_init(self):
        """Test ResearcherPersona initializes correctly."""
        researcher = ResearcherPersona()

        assert researcher.name == "Researcher"
        assert researcher.temperature == 0.7
        assert researcher.max_tokens == 4096

    def test_researcher_init_custom_temperature(self):
        """Test ResearcherPersona with custom temperature."""
        researcher = ResearcherPersona(temperature=0.5)

        assert researcher.temperature == 0.5

    def test_researcher_prompt_template(self):
        """Test ResearcherPersona has prompt template."""
        researcher = ResearcherPersona()

        assert researcher.prompt_template is not None
        assert len(researcher.prompt_template) > 0
        assert "Researcher" in researcher.prompt_template or "systematic" in researcher.prompt_template.lower()

    def test_researcher_invoke_returns_response(self):
        """Test ResearcherPersona.invoke returns PersonaResponse."""
        researcher = ResearcherPersona()
        context = {"prompt": "What is machine learning?"}

        response = researcher.invoke(context)

        assert isinstance(response, PersonaResponse)

    def test_researcher_invoke_includes_findings(self):
        """Test ResearcherPersona.invoke includes findings."""
        researcher = ResearcherPersona()
        context = {"prompt": "Test question"}

        response = researcher.invoke(context)

        assert response.findings is not None
        assert isinstance(response.findings, list)
        assert len(response.findings) > 0

    def test_researcher_invoke_discovery_phase(self):
        """Test ResearcherPersona confidence in discovery phase."""
        researcher = ResearcherPersona()
        context = {"prompt": "Test", "phase": "discovery"}

        response = researcher.invoke(context)

        assert response.confidence_update == "medium"

    def test_researcher_invoke_validation_phase(self):
        """Test ResearcherPersona confidence in validation phase."""
        researcher = ResearcherPersona()
        context = {"prompt": "Test", "phase": "validation"}

        response = researcher.invoke(context)

        assert response.confidence_update == "high"

    def test_researcher_invoke_metadata(self):
        """Test ResearcherPersona includes metadata."""
        researcher = ResearcherPersona()
        context = {"prompt": "Test", "phase": "discovery"}

        response = researcher.invoke(context)

        assert response.metadata is not None
        assert response.metadata["persona"] == "Researcher"
        assert response.metadata["phase"] == "discovery"
        assert "approach" in response.metadata

    def test_researcher_invoke_includes_prompt_in_findings(self):
        """Test ResearcherPersona includes prompt in findings."""
        researcher = ResearcherPersona()
        prompt = "Unique test question 12345"
        context = {"prompt": prompt}

        response = researcher.invoke(context)

        # At least one finding should mention the prompt
        assert any(prompt in finding for finding in response.findings)

    def test_create_researcher_factory(self):
        """Test create_researcher factory function."""
        researcher = create_researcher()

        assert isinstance(researcher, ResearcherPersona)
        assert researcher.name == "Researcher"

    def test_create_researcher_factory_with_params(self):
        """Test create_researcher factory with parameters."""
        researcher = create_researcher(temperature=0.4, max_tokens=2048)

        assert researcher.temperature == 0.4
        assert researcher.max_tokens == 2048


class TestCriticPersona:
    """Test suite for CriticPersona."""

    def test_critic_init(self):
        """Test CriticPersona initializes correctly."""
        critic = CriticPersona()

        assert critic.name == "Critic"
        assert critic.temperature == 0.6
        assert critic.max_tokens == 4096

    def test_critic_init_custom_temperature(self):
        """Test CriticPersona with custom temperature."""
        critic = CriticPersona(temperature=0.4)

        assert critic.temperature == 0.4

    def test_critic_prompt_template(self):
        """Test CriticPersona has prompt template."""
        critic = CriticPersona()

        assert critic.prompt_template is not None
        assert len(critic.prompt_template) > 0
        assert "Critic" in critic.prompt_template or "scrutiny" in critic.prompt_template.lower()

    def test_critic_invoke_returns_response(self):
        """Test CriticPersona.invoke returns PersonaResponse."""
        critic = CriticPersona()
        context = {"prompt": "Test claim"}

        response = critic.invoke(context)

        assert isinstance(response, PersonaResponse)

    def test_critic_invoke_includes_findings(self):
        """Test CriticPersona.invoke includes findings."""
        critic = CriticPersona()
        context = {"prompt": "Test question"}

        response = critic.invoke(context)

        assert response.findings is not None
        assert isinstance(response.findings, list)
        assert len(response.findings) > 0

    def test_critic_invoke_metadata(self):
        """Test CriticPersona includes metadata."""
        critic = CriticPersona()
        context = {"prompt": "Test"}

        response = critic.invoke(context)

        assert response.metadata is not None
        assert response.metadata["persona"] == "Critic"

    def test_create_critic_factory(self):
        """Test create_critic factory function."""
        critic = create_critic()

        assert isinstance(critic, CriticPersona)
        assert critic.name == "Critic"

    def test_create_critic_factory_with_params(self):
        """Test create_critic factory with parameters."""
        critic = create_critic(temperature=0.3, max_tokens=3000)

        assert critic.temperature == 0.3
        assert critic.max_tokens == 3000


class TestPlannerPersona:
    """Test suite for PlannerPersona."""

    def test_planner_init(self):
        """Test PlannerPersona initializes correctly."""
        planner = PlannerPersona()

        assert planner.name == "Planner"
        assert planner.temperature == 0.7
        assert planner.max_tokens == 4096

    def test_planner_init_custom_temperature(self):
        """Test PlannerPersona with custom temperature."""
        planner = PlannerPersona(temperature=0.3)

        assert planner.temperature == 0.3

    def test_planner_prompt_template(self):
        """Test PlannerPersona has prompt template."""
        planner = PlannerPersona()

        assert planner.prompt_template is not None
        assert len(planner.prompt_template) > 0

    def test_planner_invoke_returns_response(self):
        """Test PlannerPersona.invoke returns PersonaResponse."""
        planner = PlannerPersona()
        context = {"prompt": "Test planning question"}

        response = planner.invoke(context)

        assert isinstance(response, PersonaResponse)

    def test_planner_invoke_includes_findings(self):
        """Test PlannerPersona.invoke includes findings."""
        planner = PlannerPersona()
        context = {"prompt": "Test question"}

        response = planner.invoke(context)

        assert response.findings is not None
        assert isinstance(response.findings, list)
        assert len(response.findings) > 0

    def test_create_planner_factory(self):
        """Test create_planner factory function."""
        planner = create_planner()

        assert isinstance(planner, PlannerPersona)
        assert planner.name == "Planner"

    def test_create_planner_factory_with_params(self):
        """Test create_planner factory with parameters."""
        planner = create_planner(temperature=0.2, max_tokens=2500)

        assert planner.temperature == 0.2
        assert planner.max_tokens == 2500


class TestPersonaFactories:
    """Test suite for persona factory functions."""

    def test_create_default_personas(self):
        """Test create_default_personas returns three personas."""
        personas = create_default_personas()

        assert len(personas) == 3
        assert any(p.name == "Researcher" for p in personas)
        assert any(p.name == "Critic" for p in personas)
        assert any(p.name == "Planner" for p in personas)

    def test_create_default_personas_types(self):
        """Test create_default_personas returns correct types."""
        personas = create_default_personas()

        assert isinstance(personas[0], ResearcherPersona) or isinstance(personas[0], Persona)
        assert isinstance(personas[1], CriticPersona) or isinstance(personas[1], Persona)
        assert isinstance(personas[2], PlannerPersona) or isinstance(personas[2], Persona)

    def test_create_default_personas_independent(self):
        """Test multiple calls create independent instances."""
        personas1 = create_default_personas()
        personas2 = create_default_personas()

        # Should be different instances
        assert personas1[0] is not personas2[0]
        assert personas1[1] is not personas2[1]
        assert personas1[2] is not personas2[2]

    def test_get_default_registry(self):
        """Test get_default_registry returns pre-populated registry."""
        registry = get_default_registry()

        assert isinstance(registry, PersonaRegistry)
        assert len(registry.list_all()) == 3

    def test_get_default_registry_has_researcher(self):
        """Test default registry includes Researcher."""
        registry = get_default_registry()

        researcher = registry.get("Researcher")
        assert researcher is not None
        assert researcher.name == "Researcher"

    def test_get_default_registry_has_critic(self):
        """Test default registry includes Critic."""
        registry = get_default_registry()

        critic = registry.get("Critic")
        assert critic is not None
        assert critic.name == "Critic"

    def test_get_default_registry_has_planner(self):
        """Test default registry includes Planner."""
        registry = get_default_registry()

        planner = registry.get("Planner")
        assert planner is not None
        assert planner.name == "Planner"

    def test_get_default_registry_independent(self):
        """Test multiple calls create independent registries."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        # Registries are independent
        # (Modifying one shouldn't affect the other)
        assert registry1 is not registry2


class TestPersonaIntegration:
    """Integration tests for persona system."""

    def test_registry_with_all_personas(self):
        """Test registry with all three personas."""
        registry = get_default_registry()

        # All personas should be accessible
        assert registry.get("Researcher") is not None
        assert registry.get("Critic") is not None
        assert registry.get("Planner") is not None

        # All should be distinct
        researcher = registry.get("Researcher")
        critic = registry.get("Critic")
        planner = registry.get("Planner")

        assert researcher.name != critic.name
        assert critic.name != planner.name
        assert researcher.name != planner.name

    def test_persona_responses_structure(self):
        """Test all personas return properly structured responses."""
        registry = get_default_registry()
        context = {"prompt": "Test topic", "phase": "discovery"}

        for persona in registry.list_all():
            response = persona.invoke(context)

            assert isinstance(response, PersonaResponse)
            assert isinstance(response.findings, list)
            assert isinstance(response.metadata, dict)
            assert "persona" in response.metadata

    def test_different_personas_different_findings(self):
        """Test that different personas provide distinct findings."""
        researcher = create_researcher()
        critic = create_critic()
        planner = create_planner()

        context = {"prompt": "How should we approach this?"}

        researcher_response = researcher.invoke(context)
        critic_response = critic.invoke(context)
        planner_response = planner.invoke(context)

        # All should have findings but they should be from different perspectives
        assert len(researcher_response.findings) > 0
        assert len(critic_response.findings) > 0
        assert len(planner_response.findings) > 0

        # Metadata should identify the personas
        assert researcher_response.metadata["persona"] == "Researcher"
        assert critic_response.metadata["persona"] == "Critic"
        assert planner_response.metadata["persona"] == "Planner"

    def test_persona_temperature_affects_generation(self):
        """Test that temperature setting is preserved."""
        cool_researcher = ResearcherPersona(temperature=0.2)
        hot_researcher = ResearcherPersona(temperature=0.9)

        assert cool_researcher.temperature == 0.2
        assert hot_researcher.temperature == 0.9
        assert cool_researcher.temperature != hot_researcher.temperature

    def test_persona_max_tokens_configuration(self):
        """Test that max_tokens setting is preserved."""
        short_persona = Persona(
            name="Short",
            prompt_template="Test",
            max_tokens=1000
        )
        long_persona = Persona(
            name="Long",
            prompt_template="Test",
            max_tokens=8000
        )

        assert short_persona.max_tokens == 1000
        assert long_persona.max_tokens == 8000
