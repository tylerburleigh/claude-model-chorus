# Persona-Driven Orchestration: Implementation Examples

## 1. Persona Registry Definition

### Researcher Persona Configuration

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Persona:
    """Configuration for a reasoning persona."""
    name: str
    role_description: str
    system_prompt: str
    reasoning_steps: List[str]
    output_format: str
    temperature: float
    max_tokens: int
    validation_rules: List[str]
    model_preference: str = "claude-3-5-sonnet-20241022"

# Researcher Persona
RESEARCHER_PERSONA = Persona(
    name="researcher",
    role_description="A meticulous researcher analyzing this problem step-by-step.",
    system_prompt="""You are a research analyst who investigates problems methodically.

Your approach:
1. Ask clarifying questions about the problem
2. Gather multiple data sources and evidence
3. Identify patterns and correlations
4. Build comprehensive understanding
5. Document evidence chains clearly

Your goal: Uncover root causes and establish evidence-based understanding.

Constraints:
- Be thorough and avoid premature conclusions
- Support all claims with specific evidence
- Consider multiple explanations
- Highlight gaps in understanding
- Suggest areas for further investigation

Output format:
EVIDENCE:
[List specific findings with evidence]

PATTERNS:
[Identified patterns and correlations]

HYPOTHESIS:
[Current working theory with confidence level]

GAPS:
[Unresolved questions or missing evidence]""",
    reasoning_steps=[
        "Gather all available evidence and data points",
        "Organize evidence by source and reliability",
        "Identify patterns and correlations",
        "Form initial hypothesis based on evidence",
        "List gaps and missing information",
        "Assess confidence level based on evidence quality"
    ],
    output_format="""
    Structured JSON with:
    - evidence: list of findings with source
    - patterns: identified correlations
    - hypothesis: current theory
    - confidence: low/medium/high
    - gaps: unresolved questions
    """,
    temperature=0.4,  # Lower temperature for consistency
    max_tokens=2000,
    validation_rules=[
        "All claims must be supported by evidence",
        "Distinguish between observation and inference",
        "Flag assumptions explicitly",
        "Quantify where possible"
    ]
)

# Critic Persona
CRITIC_PERSONA = Persona(
    name="critic",
    role_description="A critical analyst who challenges assumptions and identifies weaknesses.",
    system_prompt="""You are a critical analyst and skeptic.

Your approach:
1. Challenge every major claim and assumption
2. Look for logical gaps and inconsistencies
3. Identify edge cases and exceptions
4. Stress-test recommendations
5. Propose alternative explanations

Your goal: Ensure robust analysis by stress-testing hypotheses.

Constraints:
- Be rigorous and skeptical
- Question assumptions explicitly
- Look for what could go wrong
- Propose concrete alternatives
- Validate evidence quality

Output format:
CHALLENGES:
[List assumptions being challenged]

VULNERABILITIES:
[Logical gaps and weaknesses found]

ALTERNATIVE_HYPOTHESES:
[Other possible explanations]

RISKS:
[What could invalidate the hypothesis]

REMAINING_CONFIDENCE:
[Assessment of hypothesis soundness]""",
    reasoning_steps=[
        "Identify all major assumptions in current hypothesis",
        "Challenge each assumption with evidence or lack thereof",
        "Look for logical gaps and inconsistencies",
        "Identify edge cases and boundary conditions",
        "Propose at least 2 alternative explanations",
        "Assess what evidence would refute hypothesis"
    ],
    output_format="""
    Structured JSON with:
    - challenges: assumption challenges
    - vulnerabilities: logical gaps
    - alternatives: other explanations
    - risks: invalidation scenarios
    - confidence_assessment: sound/needs_work
    """,
    temperature=0.3,  # Very low for consistency
    max_tokens=1500,
    validation_rules=[
        "Each challenge must be specific and concrete",
        "Distinguish between valid concern and mere doubt",
        "Provide evidence for challenges where possible",
        "Suggest how to address each vulnerability"
    ]
)

# Planner Persona
PLANNER_PERSONA = Persona(
    name="planner",
    role_description="A strategic planner who structures complex information into actionable steps.",
    system_prompt="""You are a strategic planner and project designer.

Your approach:
1. Break down into clear phases
2. Identify dependencies and sequencing
3. Assign timelines and success criteria
4. Consider resource constraints
5. Create rollout strategy

Your goal: Transform analysis into executable action plans.

Constraints:
- Be concrete and practical
- Every step must be actionable
- Include success criteria
- Consider implementation risks
- Plan for monitoring and validation

Output format:
PHASES:
[Numbered phases with tasks]

TIMELINE:
[Duration estimates]

SUCCESS_CRITERIA:
[Measurable outcomes]

DEPENDENCIES:
[Task sequencing]

RISKS:
[Implementation challenges]""",
    reasoning_steps=[
        "Review all findings and validate root cause",
        "Break down solution into phases",
        "Identify dependencies between phases",
        "Estimate timeline for each phase",
        "Define success criteria for validation",
        "Identify implementation risks"
    ],
    output_format="""
    Structured JSON with:
    - phases: list of execution phases
    - timeline: duration estimates
    - success_criteria: measurable outcomes
    - dependencies: task sequencing
    - risks: implementation challenges
    - metrics: how to monitor
    """,
    temperature=0.3,  # Consistent planning
    max_tokens=2000,
    validation_rules=[
        "All phases must have clear success criteria",
        "Timeline estimates must be reasonable",
        "Dependencies must be explicitly stated",
        "Risks must have mitigation strategies"
    ]
)

# Persona Registry
PERSONA_REGISTRY = {
    "researcher": RESEARCHER_PERSONA,
    "critic": CRITIC_PERSONA,
    "planner": PLANNER_PERSONA
}
```

## 2. Context Router Implementation

```python
from enum import Enum
from typing import Dict, Optional

class InvestigationPhase(Enum):
    """Phases of an investigation."""
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    PLANNING = "planning"
    COMPLETE = "complete"

@dataclass
class InvestigationContext:
    """Context for routing decisions."""
    investigation_id: str
    phase: InvestigationPhase
    hypothesis_confidence: str  # exploring, low, medium, high
    unresolved_questions: int
    identified_risks: int
    prior_persona: Optional[str]
    iterations_count: int
    investigation_type: str  # debugging, architecture, security, etc.

class ContextRouter:
    """Routes investigations to appropriate personas."""

    def __init__(self, registry: Dict[str, Persona]):
        self.registry = registry

    def route(self, context: InvestigationContext) -> str:
        """Determine which persona to invoke next."""

        scoring = {
            "researcher": 0.0,
            "critic": 0.0,
            "planner": 0.0,
        }

        # Factor 1: Investigation phase
        if context.phase == InvestigationPhase.DISCOVERY:
            scoring["researcher"] += 0.4
        elif context.phase == InvestigationPhase.VALIDATION:
            scoring["critic"] += 0.4
        elif context.phase == InvestigationPhase.PLANNING:
            scoring["planner"] += 0.4

        # Factor 2: Hypothesis confidence
        confidence_routing = {
            "exploring": {"researcher": 0.3},
            "low": {"researcher": 0.3},
            "medium": {"critic": 0.3},
            "high": {"planner": 0.3},
        }

        if context.hypothesis_confidence in confidence_routing:
            for persona, score in confidence_routing[context.hypothesis_confidence].items():
                scoring[persona] += score

        # Factor 3: Unresolved items
        if context.unresolved_questions > 0:
            scoring["researcher"] += 0.2
        if context.identified_risks > 0:
            scoring["critic"] += 0.2

        # Factor 4: Avoid repeating same persona
        if context.prior_persona:
            scoring[context.prior_persona] -= 0.1

        # Factor 5: Investigation type hints
        type_hints = {
            "debugging": {"researcher": 0.15, "critic": 0.1},
            "architecture": {"researcher": 0.1, "critic": 0.15, "planner": 0.15},
            "security": {"researcher": 0.1, "critic": 0.2, "planner": 0.1},
        }

        if context.investigation_type in type_hints:
            for persona, score in type_hints[context.investigation_type].items():
                scoring[persona] += score

        # Escalation: too many iterations
        if context.iterations_count > 5 and context.hypothesis_confidence < "high":
            scoring["critic"] -= 0.3  # Avoid infinite loops
            scoring["researcher"] -= 0.3

        # Return highest scoring persona
        selected_persona = max(scoring, key=scoring.get)

        print(f"Routing Decision: {selected_persona}")
        print(f"  Scores: {scoring}")
        print(f"  Reasons: phase={context.phase.value}, confidence={context.hypothesis_confidence}")

        return selected_persona
```

## 3. Memory Controller Implementation

```python
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import sqlite3
from pathlib import Path

class MemoryEntry:
    """Single memory entry."""

    def __init__(
        self,
        investigation_id: str,
        persona: str,
        findings: Dict[str, Any],
        timestamp: datetime,
        confidence: str,
        context_tags: List[str],
        references: List[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.investigation_id = investigation_id
        self.persona = persona
        self.findings = findings
        self.timestamp = timestamp
        self.confidence = confidence
        self.context_tags = context_tags
        self.references = references or []
        self.version = 1
        self.sync_status = "pending"  # pending, synced

    def to_dict(self):
        return {
            "id": self.id,
            "investigation_id": self.investigation_id,
            "persona": self.persona,
            "findings": self.findings,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "context_tags": self.context_tags,
            "references": self.references,
            "version": self.version
        }

class MemoryController:
    """Manages short-term and long-term memory."""

    def __init__(self, db_path: str = "/tmp/investigation_memory.db"):
        self.db_path = db_path
        self.short_term: Dict[str, MemoryEntry] = {}  # In-memory cache
        self.sync_queue: List[MemoryEntry] = []
        self._init_database()

    def _init_database(self):
        """Initialize long-term storage database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_logs (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                persona TEXT NOT NULL,
                findings TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence TEXT NOT NULL,
                context_tags TEXT NOT NULL,
                references TEXT NOT NULL,
                version INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_investigation
            ON memory_logs(investigation_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_persona_time
            ON memory_logs(persona, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence
            ON memory_logs(confidence)
        """)

        conn.commit()
        conn.close()

    def store_finding(
        self,
        investigation_id: str,
        persona: str,
        findings: Dict[str, Any],
        confidence: str,
        context_tags: List[str],
        references: List[str] = None
    ) -> str:
        """Store a finding in memory."""

        entry = MemoryEntry(
            investigation_id=investigation_id,
            persona=persona,
            findings=findings,
            timestamp=datetime.now(),
            confidence=confidence,
            context_tags=context_tags,
            references=references
        )

        # Store in short-term cache
        self.short_term[entry.id] = entry

        # Queue for long-term storage
        self.sync_queue.append(entry)

        # Check if we need to evict from short-term
        self._manage_cache_pressure()

        # Trigger async sync
        # (In real implementation, this would be async)
        self._sync_to_longterm()

        return entry.id

    def _manage_cache_pressure(self):
        """Manage short-term cache size."""
        MAX_CACHE_SIZE = 1_000_000  # 1MB
        EVICTION_THRESHOLD = 0.8

        cache_size = sum(
            len(json.dumps(entry.to_dict()).encode())
            for entry in self.short_term.values()
        )

        if cache_size > MAX_CACHE_SIZE * EVICTION_THRESHOLD:
            # Evict oldest non-critical entries
            sorted_entries = sorted(
                self.short_term.items(),
                key=lambda x: x[1].timestamp
            )

            for entry_id, entry in sorted_entries[:len(sorted_entries) // 2]:
                # Keep high-confidence entries
                if entry.confidence not in ["high", "very_high"]:
                    del self.short_term[entry_id]

    def _sync_to_longterm(self):
        """Sync pending entries to long-term storage."""
        if not self.sync_queue:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for entry in self.sync_queue:
            cursor.execute("""
                INSERT INTO memory_logs
                (id, investigation_id, persona, findings, timestamp,
                 confidence, context_tags, references, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.investigation_id,
                entry.persona,
                json.dumps(entry.findings),
                entry.timestamp.isoformat(),
                entry.confidence,
                json.dumps(entry.context_tags),
                json.dumps(entry.references),
                entry.version
            ))
            entry.sync_status = "synced"

        conn.commit()
        conn.close()

        # Clear sync queue
        self.sync_queue = []

    def get_investigation_context(self, investigation_id: str) -> Dict[str, Any]:
        """Retrieve all findings for an investigation."""

        # Check short-term first
        short_term_entries = [
            entry for entry in self.short_term.values()
            if entry.investigation_id == investigation_id
        ]

        # Get from long-term for full history
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM memory_logs
            WHERE investigation_id = ?
            ORDER BY timestamp DESC
        """, (investigation_id,))

        long_term_entries = cursor.fetchall()
        conn.close()

        return {
            "short_term": [e.to_dict() for e in short_term_entries],
            "long_term": long_term_entries
        }

    def query_by_persona(
        self,
        investigation_id: str,
        persona: str
    ) -> List[Dict[str, Any]]:
        """Get all findings from specific persona."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM memory_logs
            WHERE investigation_id = ? AND persona = ?
            ORDER BY timestamp DESC
        """, (investigation_id, persona))

        results = cursor.fetchall()
        conn.close()

        return results

    def query_by_confidence(
        self,
        investigation_id: str,
        min_confidence: str
    ) -> List[Dict[str, Any]]:
        """Get findings with minimum confidence level."""

        confidence_levels = ["exploring", "low", "medium", "high", "very_high"]
        min_index = confidence_levels.index(min_confidence)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ",".join(
            "?" * len(confidence_levels[min_index:])
        )

        cursor.execute(f"""
            SELECT * FROM memory_logs
            WHERE investigation_id = ? AND confidence IN ({placeholders})
            ORDER BY timestamp DESC
        """, [investigation_id] + confidence_levels[min_index:])

        results = cursor.fetchall()
        conn.close()

        return results
```

## 4. Orchestration Workflow Example

```python
class OrchestrationEngine:
    """Orchestrates investigation workflow."""

    def __init__(
        self,
        persona_registry: Dict[str, Persona],
        memory_controller: MemoryController,
        router: ContextRouter
    ):
        self.personas = persona_registry
        self.memory = memory_controller
        self.router = router

    async def run_investigation(
        self,
        investigation_type: str,
        initial_context: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run a complete investigation."""

        investigation_id = str(uuid.uuid4())
        iteration = 0
        phase = InvestigationPhase.DISCOVERY
        hypothesis = None
        confidence = "exploring"

        print(f"Starting investigation {investigation_id}")
        print(f"Type: {investigation_type}")
        print(f"Context: {initial_context}\n")

        while iteration < max_iterations and phase != InvestigationPhase.COMPLETE:
            iteration += 1

            # Build routing context
            context = InvestigationContext(
                investigation_id=investigation_id,
                phase=phase,
                hypothesis_confidence=confidence,
                unresolved_questions=self._count_questions(hypothesis),
                identified_risks=self._count_risks(self.memory.get_investigation_context(investigation_id)),
                prior_persona=self._get_prior_persona(investigation_id),
                iterations_count=iteration,
                investigation_type=investigation_type
            )

            # Route to next persona
            next_persona_name = self.router.route(context)
            next_persona = self.personas[next_persona_name]

            print(f"\n{'='*60}")
            print(f"Iteration {iteration}: Invoking {next_persona_name.upper()}")
            print(f"Phase: {phase.value}, Confidence: {confidence}")
            print(f"{'='*60}\n")

            # Invoke persona
            result = await self._invoke_persona(
                persona=next_persona,
                investigation_id=investigation_id,
                context=initial_context,
                memory=self.memory.get_investigation_context(investigation_id)
            )

            # Store findings
            self.memory.store_finding(
                investigation_id=investigation_id,
                persona=next_persona_name,
                findings=result["findings"],
                confidence=result.get("confidence", "medium"),
                context_tags=[investigation_type, phase.value],
                references=result.get("references", [])
            )

            # Update investigation state
            if next_persona_name == "researcher":
                hypothesis = result.get("hypothesis", hypothesis)
                confidence = result.get("confidence", confidence)
                if phase == InvestigationPhase.DISCOVERY and confidence in ["medium", "high"]:
                    phase = InvestigationPhase.VALIDATION

            elif next_persona_name == "critic":
                challenges = result.get("challenges_count", 0)
                if challenges == 0:
                    phase = InvestigationPhase.PLANNING
                confidence = result.get("confidence", confidence)

            elif next_persona_name == "planner":
                phase = InvestigationPhase.COMPLETE

            # Display findings
            print(f"\nFindings from {next_persona_name}:")
            print(json.dumps(result["findings"], indent=2))
            print(f"\nConfidence: {result.get('confidence', 'medium')}")

        return {
            "investigation_id": investigation_id,
            "iterations": iteration,
            "final_phase": phase.value,
            "final_confidence": confidence,
            "hypothesis": hypothesis,
            "memory": self.memory.get_investigation_context(investigation_id)
        }

    async def _invoke_persona(
        self,
        persona: Persona,
        investigation_id: str,
        context: str,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke a persona to process context."""

        # Format the prompt with context and prior findings
        prompt = self._build_prompt(persona, context, memory)

        # Call the model (using Claude API)
        response = await self._call_model(
            model=persona.model_preference,
            system_prompt=persona.system_prompt,
            user_prompt=prompt,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens
        )

        # Parse and validate response
        findings = self._parse_response(response, persona.output_format)

        # Validate against persona rules
        self._validate_findings(findings, persona.validation_rules)

        return findings

    def _build_prompt(
        self,
        persona: Persona,
        context: str,
        memory: Dict[str, Any]
    ) -> str:
        """Build prompt with context and prior findings."""

        prompt = f"""
INVESTIGATION CONTEXT:
{context}

PRIOR FINDINGS:
"""

        # Add relevant prior findings
        for entry in memory.get("long_term", [])[:5]:  # Last 5 entries
            prompt += f"\n- [{entry[3]}] {entry[4][:200]}..."

        prompt += f"""

YOUR TASK:
{persona.role_description}

REASONING STEPS:
{chr(10).join(persona.reasoning_steps)}

REQUIRED OUTPUT FORMAT:
{persona.output_format}
"""
        return prompt

    async def _call_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Claude API."""

        # (In real implementation, use anthropic library)
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return message.content[0].text

    def _parse_response(
        self,
        response: str,
        output_format: str
    ) -> Dict[str, Any]:
        """Parse model response into structured findings."""

        # Extract JSON from response
        try:
            # Look for JSON block
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: parse key sections
        return {
            "findings": response,
            "confidence": "medium"
        }

    def _validate_findings(
        self,
        findings: Dict[str, Any],
        validation_rules: List[str]
    ) -> bool:
        """Validate findings against persona rules."""

        # In real implementation, validate against rules
        return True

    def _count_questions(self, hypothesis: str) -> int:
        """Count unresolved questions."""
        if not hypothesis:
            return 5  # Start with high count
        return hypothesis.count("?")

    def _count_risks(self, memory: Dict[str, Any]) -> int:
        """Count identified risks from memory."""
        return 0  # Simplified

    def _get_prior_persona(self, investigation_id: str) -> Optional[str]:
        """Get the previous persona invoked."""
        return None  # Simplified
```

## 5. Usage Example

```python
async def main():
    """Example investigation session."""

    # Initialize components
    memory = MemoryController()
    router = ContextRouter(PERSONA_REGISTRY)
    engine = OrchestrationEngine(PERSONA_REGISTRY, memory, router)

    # Run investigation
    result = await engine.run_investigation(
        investigation_type="debugging",
        initial_context="""
        Users report intermittent 401 errors on checkout flow.
        Error rate: ~5% of requests
        Pattern: Random, no clear timing
        Logs: Show valid tokens, auth middleware trace unclear
        """,
        max_iterations=10
    )

    # Display results
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)
    print(f"Investigation ID: {result['investigation_id']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Hypothesis: {result['hypothesis']}")
    print(f"Final Confidence: {result['final_confidence']}")

    # Inspect memory
    print("\nMemory Contents:")
    for entry in result['memory']['long_term']:
        print(f"  - {entry[3]} ({entry[4]}): {entry[5]}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

These code examples show how the persona-driven orchestration system would be implemented in practice, with concrete data structures, routing logic, and memory management.

