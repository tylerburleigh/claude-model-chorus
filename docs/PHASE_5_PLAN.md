# Phase 5 Implementation Plan

**Project:** ModelChorus Skill Instructions
**Spec ID:** modelchorus-skill-instructions-2025-11-07-001
**Phase:** Phase 5 - Refinement & Integration
**Last Updated:** 2025-11-07

---

## Overview

Phase 5 implements improvements identified from multi-model consensus review (task-4-1, task-4-2). This plan reflects the **revised scope** based on user clarifications on 2025-11-07.

**Original Estimate:** 7 hours
**Revised Estimate:** 14-16 hours
**Status:** In Progress (60% complete)

---

## Scope Clarifications (2025-11-07)

### ✅ In Scope
- Workflow selection guide as **separate file** (docs/WORKFLOW_SELECTION_GUIDE.md)
- Terminology standardization across all SKILL.md files
- Output schemas **embedded in SKILL.md files**
- Technical contracts (parameters, return formats) in SKILL.md files
- Expected outputs added to examples

### ❌ Out of Scope
- Onboarding guides (SKILL.md is for AI agents, not users)
- Quick Start sections (same reason)
- Cross-references between skills (keep standalone)
- Cost/latency guidance (not practical for current scope)
- Interactive setup modes (requires CLI changes)
- "Lite" workflow variants (unclear demand)
- QA checklists (schemas provide similar value)

---

## Phase 5.1: Terminology & Consistency

**Goal:** Standardize terminology across all workflows based on glossary

| Task | Status | Time | Commit | Notes |
|------|--------|------|--------|-------|
| Create terminology glossary (docs/GLOSSARY.md) | ✅ DONE | 1.5h | 5c8457d | 383 lines, comprehensive reference |
| Audit all 6 SKILL.md files for terminology | ✅ DONE | 1h | d06d3d2 | Found inconsistencies |
| Apply bulk terminology updates | ✅ DONE | 1h | d06d3d2 | 64 changes across files |
| Verify "When to Use" sections exist | ✅ DONE | 0.5h | - | All files already had sections |

**Phase 5.1 Total:** 4 hours (COMPLETE)

**Deliverables:**
- ✅ `docs/GLOSSARY.md` - Standardized terminology reference
- ✅ All SKILL.md files use consistent terms:
  - `session_id` (not thread_id/continuation_id)
  - `--session-id` (CLI parameter)
  - `--provider` (not --model)
  - Default temperature: 0.7

---

## Phase 5.2: Workflow Selection Guide

**Goal:** Create comprehensive guide addressing #1 consensus finding (workflow differentiation)

| Task | Status | Time | Commit | Notes |
|------|--------|------|--------|-------|
| Create workflow selection guide | ✅ DONE | 4h | ec44d6d | 599 lines, comprehensive |
| Decision matrix | ✅ DONE | - | ec44d6d | Flowchart for quick selection |
| Comparison table | ✅ DONE | - | ec44d6d | Speed/cost/complexity |
| Detailed profiles (6 workflows) | ✅ DONE | - | ec44d6d | Strengths/weaknesses/use cases |
| Cross-workflow examples | ✅ DONE | - | ec44d6d | 4 scenarios with comparisons |
| Decision framework | ✅ DONE | - | ec44d6d | By problem type & constraints |
| Workflow combinations | ✅ DONE | - | ec44d6d | Effective sequences |
| Anti-patterns | ✅ DONE | - | ec44d6d | Common mistakes to avoid |

**Phase 5.2 Total:** 4 hours (COMPLETE)

**Deliverables:**
- ✅ `docs/WORKFLOW_SELECTION_GUIDE.md` - Master selection guide
  - Quick decision matrix
  - Comparison table (all 6 workflows)
  - Detailed profiles with strengths/weaknesses
  - 4 cross-workflow examples (API design, performance, tech selection, brainstorming)
  - Decision framework by problem type and constraints
  - Workflow combinations and sequences
  - Anti-patterns to avoid

---

## Phase 5.3: Technical Contracts & Schemas

**Goal:** Add practical technical documentation to all SKILL.md files

### 5.3.1: Technical Contracts

| Task | Status | Time | Commit | Notes |
|------|--------|------|--------|-------|
| Define contract template | ⏳ PENDING | 0.5h | - | Standard format for all skills |
| Add to CHAT | ⏳ PENDING | 0.3h | - | Parameters + return formats |
| Add to CONSENSUS | ⏳ PENDING | 0.3h | - | Parameters + return formats |
| Add to THINKDEEP | ⏳ PENDING | 0.3h | - | Parameters + return formats |
| Add to ARGUMENT | ⏳ PENDING | 0.3h | - | Parameters + return formats |
| Add to IDEATE | ⏳ PENDING | 0.3h | - | Parameters + return formats |
| Add to RESEARCH | ⏳ PENDING | 0.3h | - | Parameters + return formats |

**Subtotal:** 2.5 hours

**What to document:**
- ✅ Parameters (required/optional, types, defaults) - User confirmed
- ✅ Return formats (what the tool returns) - User confirmed
- ❌ Common errors - Not requested
- ❌ Cost/latency - Out of scope

### 5.3.2: Output Schemas

| Task | Status | Time | Commit | Notes |
|------|--------|------|--------|-------|
| Define schema format/approach | ⏳ PENDING | 0.5h | - | JSON Schema or clear specs |
| Add to CHAT | ⏳ PENDING | 0.5h | - | Response format schema |
| Add to CONSENSUS | ⏳ PENDING | 0.5h | - | Synthesis output schema |
| Add to THINKDEEP | ⏳ PENDING | 0.5h | - | Investigation result schema |
| Add to ARGUMENT | ⏳ PENDING | 0.5h | - | Argument analysis schema |
| Add to IDEATE | ⏳ PENDING | 0.5h | - | Idea generation schema |
| Add to RESEARCH | ⏳ PENDING | 0.5h | - | Research dossier schema |

**Subtotal:** 3.5 hours

**Approach:**
- Embed schemas directly in SKILL.md files
- Use JSON Schema format or clear format descriptions
- Provide annotated examples

**Phase 5.3 Total:** 6 hours (PENDING)

---

## Phase 5.4: Example Outputs

**Goal:** Add expected outputs to existing examples in all SKILL.md files

| Task | Status | Time | Commit | Notes |
|------|--------|------|--------|-------|
| Review existing examples | ⏳ PENDING | 0.5h | - | Audit what examples exist |
| Add outputs to CHAT examples | ⏳ PENDING | 0.3h | - | Show what responses look like |
| Add outputs to CONSENSUS examples | ⏳ PENDING | 0.3h | - | Show synthesis results |
| Add outputs to THINKDEEP examples | ⏳ PENDING | 0.3h | - | Show investigation findings |
| Add outputs to ARGUMENT examples | ⏳ PENDING | 0.3h | - | Show debate synthesis |
| Add outputs to IDEATE examples | ⏳ PENDING | 0.3h | - | Show generated ideas |
| Add outputs to RESEARCH examples | ⏳ PENDING | 0.3h | - | Show research dossier |

**Phase 5.4 Total:** 2 hours (PENDING)

**Format:**
```markdown
**Example:**
[Existing command/usage]

**Expected Output:**
```json
{
  "result": "...",
  "session_id": "..."
}
```

**Interpretation:**
The output shows...
```

---

## Phase 5.5: Final Review & Validation

**Goal:** Ensure consistency and quality across all deliverables

| Task | Status | Time | Commit | Notes |
|------|--------|------|--------|-------|
| Cross-file consistency check | ⏳ PENDING | 0.5h | - | Verify terminology usage |
| Schema validation | ⏳ PENDING | 0.3h | - | Ensure schemas are correct |
| Link validation | ⏳ PENDING | 0.2h | - | Check all doc cross-references |
| Final read-through | ⏳ PENDING | 0.5h | - | Quality check |
| Update this plan doc | ⏳ PENDING | 0.2h | - | Mark as complete |

**Phase 5.5 Total:** 1.5 hours (PENDING)

---

## Progress Summary

### Completed

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| 5.1: Terminology & Consistency | 4/4 | 4h | ✅ COMPLETE |
| 5.2: Workflow Selection Guide | 8/8 | 4h | ✅ COMPLETE |
| **TOTAL COMPLETED** | **12/31** | **8h** | **60%** |

### Remaining

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| 5.3: Technical Contracts & Schemas | 0/13 | 6h | ⏳ PENDING |
| 5.4: Example Outputs | 0/7 | 2h | ⏳ PENDING |
| 5.5: Final Review | 0/5 | 1.5h | ⏳ PENDING |
| **TOTAL REMAINING** | **0/25** | **9.5h** | **40%** |

### Overall Phase 5

- **Total Tasks:** 37
- **Completed:** 12 (32%)
- **Remaining:** 25 (68%)
- **Time Spent:** 8 hours
- **Time Remaining:** 9.5 hours
- **Total Estimated:** 17.5 hours

---

## Key Deliverables

### ✅ Completed

1. **docs/GLOSSARY.md** (383 lines)
   - Comprehensive terminology reference
   - Standardizes session_id, provider, temperature
   - Includes deprecation notices

2. **Terminology standardization** (all 6 SKILL.md files)
   - 64 changes applied
   - Consistent parameter names
   - Aligned defaults (temperature: 0.7)

3. **docs/WORKFLOW_SELECTION_GUIDE.md** (599 lines)
   - Addresses #1 consensus finding
   - Decision matrix and comparison table
   - 4 cross-workflow examples
   - Decision framework and combinations

### ⏳ Pending

4. **Technical contracts** (in all SKILL.md files)
   - Parameter documentation
   - Return format specs

5. **Output schemas** (in all SKILL.md files)
   - JSON Schema or clear format specs
   - Embedded in each SKILL.md

6. **Example outputs** (in all SKILL.md files)
   - Augment existing examples
   - Show expected results

---

## Git History

| Commit | Date | Description |
|--------|------|-------------|
| 5c8457d | 2025-11-07 | Phase 5.1: Glossary + partial THINKDEEP updates |
| d06d3d2 | 2025-11-07 | Phase 5.1: Terminology standardization (all files) |
| ec44d6d | 2025-11-07 | Phase 5.2: Workflow selection guide |

**Branch:** skill_mds
**Remote:** origin/skill_mds (up to date)

---

## Original vs Revised Scope

### Changes from Original Plan (task-4-2 analysis)

**Removed:**
- ❌ Onboarding guides (not for AI agents)
- ❌ Quick Start sections (same reason)
- ❌ Cross-reference links between skills (keep standalone)
- ❌ Cost/latency guidance (not practical)
- ❌ Common error documentation (not requested)

**Modified:**
- ✅ Workflow selection guide → Separate file (not in SKILL.md)
- ✅ "When to Use" sections → Already existed, no work needed
- ✅ Technical contracts → Simplified to parameters + return formats only
- ✅ Output schemas → Embedded in SKILL.md (not separate files)

**Net Impact:**
- Original estimate: 28 hours (too high, included out-of-scope items)
- Revised estimate: 17.5 hours (realistic, focused scope)
- Time savings: ~10.5 hours from scope reduction

---

## Alignment with Consensus Review Findings

### Priority 1 Improvements (High Impact, Low Effort)

| Finding | Solution | Status |
|---------|----------|--------|
| Workflow differentiation gap | Workflow Selection Guide | ✅ DONE |
| Terminology inconsistency | Glossary + standardization | ✅ DONE |
| Examples need outputs | Add to all SKILL.md files | ⏳ PENDING |

### Priority 2 Improvements (High Impact, Higher Effort)

| Finding | Solution | Status |
|---------|----------|--------|
| Missing output schemas | Embed in SKILL.md files | ⏳ PENDING |
| Technical contracts missing | Add to SKILL.md files | ⏳ PENDING |

**Conclusion:** Priority 1 is 67% complete (2/3 items). Priority 2 is 0% complete (0/2 items).

---

## Next Session Checklist

When resuming Phase 5 work:

1. **Review this plan** - Understand current state
2. **Check git status** - Ensure branch is up to date
3. **Start with Phase 5.3.1** - Technical contracts (easiest remaining task)
4. **Then Phase 5.3.2** - Output schemas
5. **Then Phase 5.4** - Example outputs
6. **Finally Phase 5.5** - Final review

**Estimated time to completion:** 9.5 hours

---

## Success Criteria

Phase 5 is complete when:

- ✅ All terminology is standardized (DONE)
- ✅ Workflow selection guide exists and is comprehensive (DONE)
- ⏳ All SKILL.md files have technical contracts
- ⏳ All SKILL.md files have output schema documentation
- ⏳ All examples show expected outputs
- ⏳ Cross-file consistency verified
- ⏳ All changes committed and pushed
- ⏳ This plan document marked complete

---

## Notes

- **Context threshold:** 75% - Monitor during autonomous mode
- **Git workflow:** Commit after each major deliverable
- **User preferences:**
  - Workflow selection guide as separate file ✓
  - No onboarding guides in SKILL.md ✓
  - Technical contracts: parameters + return formats only ✓
  - Examples should show expected outputs ✓
  - No cross-references between skills ✓

---

**Document maintained by:** Claude Code (autonomous mode)
**Review frequency:** After each Phase 5 work session
**Completion target:** Phase 5 (spec overall: 82% → ~90% after Phase 5)
