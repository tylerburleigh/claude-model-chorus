# ThinkDeepWorkflow API Mismatch Fixes

## Problem

The `ThinkDeepWorkflow.run()` method requires explicit step parameters, but tests and examples are calling it with a simple `prompt` parameter that doesn't exist in the current API.

### Current API Signature
```python
async def run(
    self,
    step: str,                      # REQUIRED: Step description
    step_number: int,               # REQUIRED: Current step index (1-based)
    total_steps: int,               # REQUIRED: Total steps in investigation
    next_step_required: bool,       # REQUIRED: Whether more steps needed
    findings: str,                  # REQUIRED: What was discovered
    hypothesis: Optional[str] = None,
    confidence: str = "exploring",
    continuation_id: Optional[str] = None,
    files: Optional[List[str]] = None,
    relevant_files: Optional[List[str]] = None,
    thinking_mode: str = "medium",
    skip_provider_check: bool = False,
    **kwargs,
) -> WorkflowResult:
```

### Old Format (Incorrect)
```python
result = await workflow.run(
    prompt="Investigate authentication patterns",
    files=["src/auth.py"]
)
```

### New Format (Correct)
```python
result = await workflow.run(
    step="Investigate authentication patterns",
    step_number=1,
    total_steps=1,
    next_step_required=False,
    findings="Checking authentication patterns",
    files=["src/auth.py"],
)
```

## Conversion Guidelines

### Single-Step Tests
```python
# Simple investigation
result = await workflow.run(
    step="Original prompt text",
    step_number=1,
    total_steps=1,
    next_step_required=False,
    findings="Brief description of what this step does",
    # ... preserve other params (files, confidence, etc.)
)
```

### Multi-Step Tests
```python
# Step 1
result1 = await workflow.run(
    step="First investigation step",
    step_number=1,
    total_steps=3,  # Total steps in this investigation
    next_step_required=True,  # More steps coming
    findings="Initial findings",
)
thread_id = result1.metadata["thread_id"]

# Step 2
result2 = await workflow.run(
    step="Second investigation step",
    step_number=2,
    total_steps=3,
    next_step_required=True,
    findings="Step 2 findings",
    continuation_id=thread_id,
)

# Step 3 (final)
result3 = await workflow.run(
    step="Final investigation step",
    step_number=3,
    total_steps=3,
    next_step_required=False,  # Last step
    findings="Conclusion findings",
    continuation_id=thread_id,
)
```

## Files to Fix

### ✅ Completed
- [x] **tests/test_thinkdeep_workflow.py** - 59 calls updated
  - All TestInvestigationStepExecution tests
  - All TestHypothesisEvolution tests
  - All TestConfidenceProgression tests
  - All TestEndToEndIntegration tests

### ✅ Recently Completed

2. **tests/test_thinkdeep_complex.py** (~23 calls) ✅
   - `TestArchitecturalDecisionScenarios`
   - `TestBugInvestigationScenarios`
   - `TestComplexMultiStepReasoning`

3. **tests/test_thinkdeep_expert_validation.py** (~22 calls) ✅
   - `TestExpertValidationTriggering`
   - `TestExpertValidationResultHandling`
   - `TestExpertValidationErrorHandling`
   - `TestExpertValidationWithHypotheses`

4. **tests/test_workflow_integration_chaining.py** (~3 calls) ✅
   - `TestConsensusThinkDeepChatChaining`

5. **examples/thinkdeep_example.py** (~9 calls) ✅
   - All example usage code updated

### ⏳ Remaining

None - All files have been updated!

## Test Assertion Updates Needed

After fixing the workflow.run() calls, some tests may have assertions that check for old metadata fields:

### Old Metadata Fields (Removed)
- `investigation_step` - No longer exists

### New Metadata Fields
- `step_number` - Current step number
- `total_steps` - Total steps in investigation
- `next_step_required` - Whether investigation continues
- `confidence` - Current confidence level
- All other fields remain the same

### Assertion Fix Examples
```python
# OLD (will fail)
assert result.metadata["investigation_step"] == 2

# NEW (correct)
assert result.metadata["step_number"] == 1
```

## Quick Fix Script Template

For batch updates, you can use regex patterns:

### Single-line calls
```regex
Find:    result\d* = await workflow\.run\(prompt="([^"]+)"([^)]*)\)
Replace: result = await workflow.run(\n    step="$1",\n    step_number=1,\n    total_steps=1,\n    next_step_required=False,\n    findings="Investigation step",$2\n)
```

### Multi-line calls
Requires manual review for step numbers and totals.

## Running Tests

After making fixes, verify with:

```bash
# Run all ThinkDeep tests
python -m pytest tests/test_thinkdeep*.py -v

# Run specific test file
python -m pytest tests/test_thinkdeep_workflow.py -v

# Run specific test
python -m pytest tests/test_thinkdeep_workflow.py::TestInvestigationStepExecution::test_single_investigation_step_execution -v
```

## Expected Results

After all fixes:
- **81 ThinkDeep tests should pass** (currently failing)
- **738 other tests continue to pass**
- No `prompt=` parameter usage in any test files
- All calls use explicit `step`, `step_number`, `total_steps`, `next_step_required`, `findings`

## Notes

- The explicit API provides better control over multi-step investigations
- The `findings` parameter should briefly describe what the step discovered or is investigating
- `next_step_required=True` indicates the investigation will continue
- `continuation_id` must be used for steps 2+ in a multi-step investigation
- Test files may need assertion updates after fixing calls (check for old metadata fields)
