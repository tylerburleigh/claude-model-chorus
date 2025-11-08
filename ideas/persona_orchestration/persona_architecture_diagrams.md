# Persona-Driven Orchestration: Architecture Diagrams

## 1. System Overview Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION LAYER                         │
│                                                                         │
│  CLI Interface │ Web Dashboard │ API │ IDE Integration                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
        ┌───────▼────────────┐        ┌──────────▼──────────┐
        │ ORCHESTRATION      │        │  INVESTIGATION      │
        │ LAYER              │        │  STATE MANAGER      │
        │                    │        │                     │
        │ ┌────────────────┐ │        │ ┌─────────────────┐ │
        │ │ Context Router │ │        │ │ Phase Tracker   │ │
        │ │                │ │        │ │ Hypothesis Mgmt │ │
        │ │ Analyzes:      │ │        │ │ Confidence Prog │ │
        │ │ - Phase        │ │        │ └─────────────────┘ │
        │ │ - Confidence   │ │        │                     │
        │ │ - Unresolved   │ │        │ ┌─────────────────┐ │
        │ │ - Memory state │ │        │ │ Task Queue      │ │
        │ │ - User prefs   │ │        │ │ (Async sched)   │ │
        │ └────────┬───────┘ │        │ └─────────────────┘ │
        │          │         │        └─────────────────────┘
        │ ┌────────▼───────┐ │
        │ │ Persona        │ │
        │ │ Registry       │ │
        │ │                │ │
        │ │ Stores:        │ │
        │ │ - Templates    │ │
        │ │ - Constraints  │ │
        │ │ - Parameters   │ │
        │ └────────────────┘ │
        │                    │
        │ ┌────────────────┐ │
        │ │ Orchestration  │ │
        │ │ Rules Engine   │ │
        │ │ (DSL-based)    │ │
        │ └────────────────┘ │
        └────────┬───────────┘
                 │
    ┌────────────┼─────────────────────────────────────┐
    │            │                                      │
    ▼            ▼                                      ▼
┌──────────────────────────┐              ┌───────────────────────────┐
│   PERSONA ENGINES        │              │   TEMPORAL MEMORY LAYER   │
│                          │              │                           │
│ ┌──────────────────────┐ │              │ ┌─────────────────────┐  │
│ │  RESEARCHER          │ │              │ │ SHORT-TERM CACHE    │  │
│ │                      │ │              │ │ (In-Memory)         │  │
│ │ Prompt: Investigate  │ │              │ │                     │  │
│ │ Style: Analysis      │ │              │ │ - Current thread    │  │
│ │ Constraints: Thorough│ │              │ │ - Active findings   │  │
│ │ Model: Claude        │ │              │ │ - Working hypotheses│  │
│ └──────────────────────┘ │              │ │ - Context window    │  │
│                          │              │ │ (LRU eviction)      │  │
│ ┌──────────────────────┐ │              │ └────────────┬────────┘  │
│ │  CRITIC              │ │              │              │ Flush on  │
│ │                      │ │              │              │ cache full│
│ │ Prompt: Challenge    │ │              │              │           │
│ │ Style: Skeptical     │ │              │              ▼           │
│ │ Constraints: Rigorous│ │              │ ┌─────────────────────┐  │
│ │ Model: Claude        │ │              │ │ LONG-TERM LOGS      │  │
│ └──────────────────────┘ │              │ │ (Indexed Database)  │  │
│                          │              │ │                     │  │
│ ┌──────────────────────┐ │              │ │ Indexed by:         │  │
│ │  PLANNER             │ │              │ │ - Investigation ID  │  │
│ │                      │ │              │ │ - Timestamp         │  │
│ │ Prompt: Design       │ │              │ │ - Persona           │  │
│ │ Style: Structured    │ │              │ │ - Context tags      │  │
│ │ Constraints: Practical│ │             │ │ - Confidence        │  │
│ │ Model: Claude        │ │              │ │ - Full-text search  │  │
│ └──────────────────────┘ │              │ │                     │  │
│                          │              │ │ Retention: Unlimited│  │
└───────────┬──────────────┘              │ │ Queryable: Yes      │  │
            │                             │ └─────────────────────┘  │
            │                             └───────────────────────────┘
            │
    ┌───────▼────────────┐
    │ MODEL PROVIDERS    │
    │                    │
    │ Claude             │
    │ Gemini (fallback)  │
    │ Codex (fallback)   │
    └────────────────────┘
```

## 2. Investigation Flow Sequence

```
User Initiates Investigation
        │
        ▼
┌─────────────────────────────────────┐
│ Investigation Created               │
│ - UUID generated                    │
│ - Phase: DISCOVERY                  │
│ - Hypothesis: None                  │
│ - Confidence: exploring             │
└─────────────────┬───────────────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │ Route to Persona?     │
      └───────┬───────────────┘
              │
              ├─ Phase: DISCOVERY → RESEARCHER
              │
              ▼
    ┌─────────────────────────────┐
    │ Invoke RESEARCHER           │
    │ - Load context from memory  │
    │ - Generate analysis         │
    │ - Update hypothesis         │
    │ - Store findings            │
    └──────────┬──────────────────┘
               │
               ├─ SHORT-TERM: Store in cache
               │
               ├─ LONG-TERM: Index in logs
               │
               ▼
    ┌─────────────────────────────┐
    │ Update Investigation State  │
    │ - Findings added            │
    │ - Confidence: low           │
    │ - Phase: VALIDATION         │
    └──────────┬──────────────────┘
               │
               ▼
      ┌───────────────────────┐
      │ Route to Persona?     │
      │                       │
      │ Phase: VALIDATION     │
      │ Confidence: low       │
      │ → CRITIC              │
      └───────┬───────────────┘
              │
              ▼
    ┌─────────────────────────────┐
    │ Invoke CRITIC               │
    │ - Load Researcher findings  │
    │ - Challenge assumptions     │
    │ - Generate alternative hyp. │
    │ - Store challenges          │
    └──────────┬──────────────────┘
               │
               ├─ SHORT-TERM: Store in cache
               │
               ├─ LONG-TERM: Index in logs
               │
               ▼
    ┌─────────────────────────────┐
    │ Update Investigation State  │
    │ - Challenges documented     │
    │ - Confidence: medium        │
    │ - Decision point            │
    └──────────┬──────────────────┘
               │
        ┌──────┴──────────┐
        │                 │
   Challenges     No Challenges
   Found              Found
        │                 │
        ▼                 ▼
    RESEARCHER         PLANNER
    Re-analyze         Design Plan
        │                 │
        ▼                 ▼
    ┌─────────────────────────────┐
    │ Route to Next Persona?      │
    │ Or Complete Investigation   │
    └─────────────────────────────┘
```

## 3. Memory Layout During Investigation

```
INVESTIGATION STATE: debug-latency-001

PHASE 1: RESEARCHER ANALYSIS
┌─────────────────────────────────────────────────────────┐
│ RESEARCHER PROCESSING (13:42)                           │
│ Status: Complete                                        │
│                                                         │
│ Input Context:                                          │
│ - API request logs (13:30-13:45)                       │
│ - Database metrics                                      │
│ - Service dependencies                                 │
│                                                         │
│ Findings:                                               │
│ R1. P99 latency spike from 50ms to 2000ms at 13:30    │
│ R2. Coincides with connection pool exhaustion          │
│ R3. Database query patterns unchanged                   │
│ R4. No deployment changes at 13:30                     │
│                                                         │
│ Initial Hypothesis: "DB connection exhaustion"         │
│ Confidence: LOW                                        │
│                                                         │
│ Memory Stored:                                          │
│ ├─ SHORT-TERM: [R1, R2, R3, R4, hypothesis]           │
│ └─ LONG-TERM: [indexed entry with full details]       │
└─────────────────────────────────────────────────────────┘

PHASE 2: CRITIC CHALLENGE
┌─────────────────────────────────────────────────────────┐
│ CRITIC PROCESSING (13:48)                               │
│ Status: Complete                                        │
│                                                         │
│ Input Context (from memory):                            │
│ ├─ Researcher findings: R1, R2, R3, R4                │
│ ├─ Current hypothesis: "DB connection"                │
│ └─ Investigation phase: VALIDATION                     │
│                                                         │
│ Challenges:                                             │
│ C1. Connection exhaustion could be symptom, not cause  │
│ C2. No deployment changes = external dependency issue  │
│ C3. Query patterns haven't changed = not query load    │
│                                                         │
│ Alternative Hypothesis: "External service timeout"     │
│ Confidence: MEDIUM (hypothesis refined)                │
│                                                         │
│ Cross-References:                                       │
│ ├─ Directly references: R1, R2, R3, R4                │
│ ├─ Contradicts: R2 as root cause                      │
│ └─ Proposes: Check payment processor API               │
│                                                         │
│ Memory Stored:                                          │
│ ├─ SHORT-TERM: [C1, C2, C3, hypothesis, references]  │
│ └─ LONG-TERM: [indexed entry with cross-refs]         │
└─────────────────────────────────────────────────────────┘

PHASE 3: RESEARCHER RE-ANALYSIS
┌─────────────────────────────────────────────────────────┐
│ RESEARCHER RE-PROCESSING (13:55)                        │
│ Status: Complete                                        │
│                                                         │
│ Input Context (from memory):                            │
│ ├─ Original findings: R1, R2, R3, R4                  │
│ ├─ Critic challenges: C1, C2, C3                      │
│ ├─ Refined hypothesis: "External service"             │
│ └─ Action items: Check payment processor               │
│                                                         │
│ New Analysis:                                           │
│ R5. Payment processor timeout logs: response 50ms→2s   │
│ R6. Timeout at exact 13:30 timestamp                   │
│ R7. Incident window: 13:30-13:45 (matches)            │
│ R8. Payment API status page showed incident            │
│                                                         │
│ Refined Hypothesis: "Payment processor outage"         │
│ Confidence: HIGH (root cause confirmed)                │
│                                                         │
│ Validation of Critic feedback:                          │
│ ├─ C1 addressed: Not symptom, confirmed root cause     │
│ ├─ C2 validated: External dependency confirmed         │
│ └─ C3 correct: Not query load issue                    │
│                                                         │
│ Memory Stored:                                          │
│ ├─ SHORT-TERM: [R5, R6, R7, R8, refined hypothesis]  │
│ └─ LONG-TERM: [indexed with confidence HIGH]          │
└─────────────────────────────────────────────────────────┘

PHASE 4: PLANNER DESIGN
┌─────────────────────────────────────────────────────────┐
│ PLANNER PROCESSING (14:02)                              │
│ Status: Complete                                        │
│                                                         │
│ Input Context (from memory):                            │
│ ├─ All researcher findings: R1-R8                      │
│ ├─ All critic challenges: C1-C3 (validated)            │
│ ├─ Confirmed root cause: "Payment processor outage"    │
│ └─ Confidence: HIGH                                     │
│                                                         │
│ Remediation Design:                                     │
│ P1. Phase 1 (1 hour): Implement timeout fallback       │
│ P2. Phase 2 (2 hours): Add circuit breaker pattern     │
│ P3. Phase 3 (3 hours): Implement health checks         │
│ P4. Phase 4 (ongoing): Monitor payment API health      │
│                                                         │
│ Success Criteria:                                       │
│ - Requests don't hang > 5 seconds                       │
│ - Circuit opens within 30 seconds of failures          │
│ - Health checks update every minute                     │
│                                                         │
│ Dependencies:                                           │
│ ├─ All findings from Researcher                        │
│ ├─ All validated challenges from Critic                │
│ └─ Root cause confirmation: HIGH confidence            │
│                                                         │
│ Memory Stored:                                          │
│ ├─ SHORT-TERM: [P1, P2, P3, P4, criteria, timeline]  │
│ └─ LONG-TERM: [indexed plan with full rationale]      │
└─────────────────────────────────────────────────────────┘

FINAL MEMORY STATE
┌─────────────────────────────────────────────────────────┐
│ Investigation: debug-latency-001                        │
│ Total Duration: 20 minutes (13:42 - 14:02)             │
│ Final Confidence: HIGH                                  │
│ Status: ACTIONABLE                                      │
│                                                         │
│ Memory Summary:                                         │
│ ├─ Short-term entries: 12 (current thread)             │
│ ├─ Long-term entries: 12 (permanent record)            │
│ ├─ Cross-references: 6 (Critic→Researcher, Plan→All)  │
│ ├─ Hypothesis evolutions: 2 (DB→External→Confirmed)   │
│ └─ Confidence progression: exploring→low→medium→high   │
│                                                         │
│ Persona Contributions:                                  │
│ ├─ Researcher: 8 findings (analysis + re-analysis)    │
│ ├─ Critic: 3 challenges (validation feedback)          │
│ └─ Planner: 4 design phases (actionable remedy)        │
│                                                         │
│ Key Artifacts:                                          │
│ ├─ Root cause: Payment processor timeout              │
│ ├─ Evidence chain: 8 findings, 3 validations          │
│ ├─ Implementation plan: 4 phases, 6 hours total       │
│ └─ Success metrics: Defined and measurable            │
└─────────────────────────────────────────────────────────┘
```

## 4. Memory Access Patterns

```
SHORT-TERM CACHE (Fast Access)
┌─────────────────────────────────┐
│ Entry #1: Finding R1 [13:42]    │
│ Entry #2: Finding R2 [13:42]    │ LRU
│ Entry #3: Finding R3 [13:42]    │  │
│ Entry #4: Finding R4 [13:42]    │  │
│ Entry #5: Challenge C1 [13:48]  │  │
│ Entry #6: Challenge C2 [13:48]  │  │ Pressure: 40% capacity
│ Entry #7: Challenge C3 [13:48]  │  │ Age: 10-22 minutes
│ Entry #8: Finding R5 [13:55]    │  │
│ Entry #9: Finding R6 [13:55]    │  ▼
│ Entry #10: Finding R7 [13:55]   │  Evict older
│ Entry #11: Finding R8 [13:55]   │  entries
│ Entry #12: Plan P1-P4 [14:02]   │
│                                 │
│ Size: 245KB of 1MB              │
│ Last access: 14:05              │
└─────────────────────────────────┘
              │
      ┌───────┴──────────┐
      │ Summarize on     │
      │ cache pressure   │
      ▼                  ▼
┌─────────────────────────────────┐
│ LONG-TERM LOGS (Persistent)     │
│                                 │
│ Entry: log_1 [13:42] Researcher │
│ - findings: [R1, R2, R3, R4]    │
│ - hypothesis: "DB exhaustion"   │
│ - confidence: low               │
│ - indexed: persona, timestamp   │
│                                 │
│ Entry: log_2 [13:48] Critic     │
│ - challenges: [C1, C2, C3]      │
│ - references: [log_1 findings]  │
│ - alt_hypothesis: "External"    │
│ - confidence: medium            │
│ - indexed: with cross-refs      │
│                                 │
│ Entry: log_3 [13:55] Researcher │
│ - findings: [R5, R6, R7, R8]    │
│ - references: [log_1, log_2]    │
│ - hypothesis: "Payment outage"  │
│ - confidence: high              │
│ - validated_challenges: [C1-C3] │
│                                 │
│ Entry: log_4 [14:02] Planner    │
│ - plan: [P1, P2, P3, P4]        │
│ - references: [log_1-3 all]     │
│ - timeline: 6 hours             │
│ - success_criteria: [defined]   │
│                                 │
│ Indexes:                        │
│ - By investigation: [log_1-4]   │
│ - By persona: R[1,3], C[2], P[4]│
│ - By timestamp: Ordered         │
│ - By confidence: all indexed    │
│ - Full-text: searchable         │
└─────────────────────────────────┘
```

## 5. Persona Decision Tree

```
┌─────────────────────────────────────────────────────────┐
│ START: New Investigation                                │
│ Phase: DISCOVERY                                        │
│ Confidence: exploring                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Phase == DISCOVERY?    │
        └────────┬───────────────┘
                 │
               YES
                 │
                 ▼
     ┌───────────────────────────┐
     │ Route to RESEARCHER       │
     │ Task: Gather evidence     │
     │ Style: Thorough analysis  │
     └───────────┬───────────────┘
                 │
                 ▼
         ┌─────────────────┐
         │ Researcher Work │
         │ Updates:        │
         │ - findings      │
         │ - hypothesis    │
         │ - confidence    │
         └────────┬────────┘
                  │
                  ▼
      ┌──────────────────────────┐
      │ Update State             │
      │ Phase: VALIDATION        │
      │ Confidence: low/medium   │
      │ Hypothesis: defined      │
      └────────┬─────────────────┘
               │
               ▼
   ┌───────────────────────────┐
   │ Phase == VALIDATION?      │
   │ Confidence >= low?        │
   └─────┬──────────────┬──────┘
       YES              NO (confidence exploring)
        │                │
        │                └─→ [Route to RESEARCHER again]
        │
        ▼
   ┌──────────────────────────┐
   │ Route to CRITIC          │
   │ Task: Validate findings  │
   │ Style: Challenge focused │
   └──────┬───────────────────┘
          │
          ▼
   ┌───────────────────┐
   │ Critic Work       │
   │ Updates:          │
   │ - challenges      │
   │ - alternatives    │
   │ - validation      │
   └────────┬──────────┘
            │
            ▼
    ┌─────────────────────────┐
    │ Challenges Found?       │
    └─────┬──────────────┬────┘
        YES              NO
         │                │
         ▼                ▼
    RESEARCHER         PLANNER
    Re-investigate     Design Plan
         │                │
         └────┬────┬──────┘
              ▼    ▼
         ┌──────────────────────┐
         │ More iterations?     │
         │ Confidence < HIGH?   │
         │ Iterations < limit?  │
         └─────┬──────────┬─────┘
             YES           NO
              │             │
              └─→ [Continue cycle]
                           │
                           ▼
                    ┌─────────────────────┐
                    │ Investigation Ready │
                    │ Phase: COMPLETE     │
                    │ Confidence: HIGH    │
                    │ Findings: Indexed   │
                    │ Plan: Actionable    │
                    └─────────────────────┘
```

## 6. Cross-Persona Memory References

```
INVESTIGATION THREAD: debug-latency-001

FINDINGS MAP:
┌─────────────┐
│  RESEARCHER │
│   Analysis  │
└────────┬────┘
         │
    ┌────┴─────────────────────────────────┐
    │                                      │
    R1: P99 spike                          │
    R2: Connection pool exhaustion         │
    R3: Query patterns unchanged           │
    R4: No deployment changes              │
    R5: Payment timeout logs               │
    R6: Timeout at 13:30                   │
    R7: Incident window matches            │
    R8: API status page incident           │
    │
    │  ▲  Informed by
    │  │
    └──┴─────────────────────────┐
                                 │
                          ┌──────┴────────┐
                          │    CRITIC     │
                          │   Challenges  │
                          └────┬─────┬────┘
                               │     │
                    ┌──────────┘     └────────┐
                    │                        │
                C1: Symptom not cause        │
                C2: External dependency     │
                C3: Not query load          │
                    │                       │
                    └───────┬────────────────┘
                            │
                    Validated by
                            │
                            ▼
                    ┌──────────────┐
                    │   PLANNER    │
                    │   Roadmap    │
                    └──────┬───────┘
                           │
                    ┌──────┴──────────┐
                    │                 │
                P1: Timeout fallback  │
                P2: Circuit breaker   │
                P3: Health checks     │
                P4: Monitoring        │
```

## 7. Performance Characteristics

```
LATENCY PROFILE

Step 1: Invoke Researcher
├─ Load context from memory: 10ms (short-term) / 50ms (long-term)
├─ Prepare persona prompt: 5ms
├─ API call to Claude: 800-2000ms
├─ Store findings: 5ms (short-term) + async to long-term
└─ Total: ~820-2050ms

Step 2: Invoke Critic
├─ Load Researcher findings: 10ms (short-term cache hit)
├─ Prepare persona prompt: 5ms
├─ API call to Claude: 600-1500ms
├─ Store challenges: 5ms (short-term) + async to long-term
└─ Total: ~620-1520ms

Step 3: Invoke Researcher (Re-analysis)
├─ Load all prior context: 15ms (short-term) / 80ms (long-term)
├─ Prepare persona prompt: 5ms
├─ API call to Claude: 800-2000ms
├─ Store findings: 5ms + async to long-term
└─ Total: ~825-2085ms

Step 4: Invoke Planner
├─ Load all prior findings: 20ms (partial cache, partial fetch)
├─ Prepare persona prompt: 5ms
├─ API call to Claude: 900-2200ms
├─ Store plan: 10ms + async to long-term
└─ Total: ~935-2235ms

TOTAL INVESTIGATION TIME: ~3.2-7.9 seconds (excluding API variance)

MEMORY ACCESS PATTERNS:
- First reference to finding: Long-term (slower)
- Subsequent references: Short-term cache (faster)
- Memory hit rate increases through investigation
- Prefetching reduces latency for later personas

OPTIMIZATION OPPORTUNITIES:
- Parallel loading of multiple findings: -20ms
- Prompt template pre-compilation: -3ms
- Database query optimization: -10ms
- Predicted prefetching: -15ms
- Total possible: -48ms (6% improvement)
```

## 8. State Machine Model

```
INVESTIGATION STATE MACHINE

┌─────────────────────────────────────────────────────────┐
│ State: INITIAL                                          │
│ ├─ phase: DISCOVERY                                     │
│ ├─ hypothesis: null                                     │
│ ├─ confidence: exploring                                │
│ ├─ findings_count: 0                                    │
│ └─ persona_history: []                                  │
└──────────────┬──────────────────────────────────────────┘
               │ [Start Investigation]
               ▼
┌─────────────────────────────────────────────────────────┐
│ State: RESEARCHER_ANALYZING                             │
│ ├─ phase: DISCOVERY                                     │
│ ├─ current_persona: RESEARCHER                          │
│ ├─ confidence: exploring                                │
│ ├─ findings_count: 0                                    │
│ └─ start_time: <timestamp>                              │
└──────────────┬──────────────────────────────────────────┘
               │ [Researcher completes]
               ▼
┌─────────────────────────────────────────────────────────┐
│ State: ANALYZING_COMPLETE                               │
│ ├─ phase: VALIDATION                                    │
│ ├─ hypothesis: <defined from analysis>                  │
│ ├─ confidence: low                                      │
│ ├─ findings_count: 4                                    │
│ ├─ last_persona: RESEARCHER                             │
│ └─ analysis_duration: <time>                            │
└──────────────┬──────────────────────────────────────────┘
               │ [Route to next persona]
               ▼
┌─────────────────────────────────────────────────────────┐
│ State: CRITIC_VALIDATING                                │
│ ├─ phase: VALIDATION                                    │
│ ├─ current_persona: CRITIC                              │
│ ├─ hypothesis: <under challenge>                        │
│ ├─ challenges_found: 0 (in progress)                    │
│ └─ start_time: <timestamp>                              │
└──────────────┬──────────────────────────────────────────┘
               │ [Critic completes]
               ▼
        ┌──────────────────────┐
        │ Challenges Found?    │
        └──────┬──────────┬────┘
            YES (3)       NO
              │            │
              ▼            ▼
         RESEARCHER     PLANNING
         REFINING       PHASE
```

---

This comprehensive diagram set visualizes how persona-driven orchestration with temporal memory operates at multiple levels: system architecture, investigation flow, memory layout, access patterns, decision making, cross-references, and performance characteristics.

