# ModelChorus Testing Playbook

**Version:** 1.0
**Last Updated:** 2025-11-11
**Coverage Level:** Standard (Happy Path + Edge Cases + Error Handling)

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Testing Philosophy](#testing-philosophy)
4. [Workflow Testing](#workflow-testing)
   - [Argument Workflow](#1-argument-workflow)
   - [Chat Workflow](#2-chat-workflow)
   - [Consensus Workflow](#3-consensus-workflow)
   - [Ideate Workflow](#4-ideate-workflow)
   - [Router Workflow](#5-router-workflow)
   - [Study Workflow](#6-study-workflow)
   - [ThinkDeep Workflow](#7-thinkdeep-workflow)
5. [Cross-Cutting Concerns](#cross-cutting-concerns)
6. [Test Execution Guide](#test-execution-guide)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This playbook provides comprehensive testing scenarios for all ModelChorus workflows. Each workflow is tested across three layers:

1. **Workflow Layer** - Python implementation (`model_chorus.workflows`)
2. **Agent Layer** - Claude Code subagent wrappers (`/agents/*.md`)
3. **CLI Layer** - Command-line interface (`model-chorus` commands)

The playbook uses **real provider integration** to validate end-to-end functionality with actual AI models.

### Workflow Summary

| Workflow | Purpose | Key Feature | Threading |
|----------|---------|-------------|-----------|
| **Argument** | Dialectical analysis | 3-role debate (Creator/Skeptic/Moderator) | ✅ Yes |
| **Chat** | Simple conversation | Single-model interaction | ✅ Yes |
| **Consensus** | Multi-model consultation | Parallel execution, 5 strategies | ❌ No |
| **Ideate** | Creative brainstorming | Configurable creativity & quantity | ✅ Yes |
| **Router** | Workflow selection | Intelligent recommendation | ❌ No |
| **Study** | Multi-persona research | Role-based investigation | ✅ Yes |
| **ThinkDeep** | Systematic investigation | Hypothesis tracking, confidence levels | ✅ Yes |

---

## Prerequisites

### 1. API Keys Configuration

Create a `.env` file in the project root (if not already present):

```bash
# Required for testing
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional providers
CURSOR_API_KEY=...
CODEX_API_KEY=...
```

### 2. Install ModelChorus

```bash
# Install in development mode
pip install -e .

# Verify installation
model-chorus --help
```

### 3. Verify Provider Availability

```bash
# Quick provider check
python -c "
from model_chorus.providers import ClaudeProvider, GeminiProvider
import os

claude = ClaudeProvider(api_key=os.getenv('ANTHROPIC_API_KEY'))
gemini = GeminiProvider(api_key=os.getenv('GOOGLE_API_KEY'))
print('✅ Providers configured successfully')
"
```

### 4. Test Data Preparation

Create a test directory structure:

```bash
mkdir -p test_data/{files,outputs}
echo "def hello(): return 'world'" > test_data/files/sample.py
echo "# Test Document" > test_data/files/sample.md
```

---

## Testing Philosophy

### Testing Levels

1. **Happy Path** - Standard use cases with expected inputs
2. **Edge Cases** - Boundary conditions, unusual but valid inputs
3. **Error Handling** - Invalid inputs, provider failures, timeouts

### Success Criteria

For each test scenario:

- ✅ **Output Generated** - Workflow completes without crashing
- ✅ **Format Valid** - Response matches expected structure
- ✅ **Content Quality** - Output is relevant and coherent
- ✅ **Metadata Present** - Session IDs, provider info, usage stats included
- ✅ **Timing Acceptable** - Completes within expected time range

### Recording Results

Use this template for each test:

```markdown
**Test:** [Test Name]
**Status:** ✅ Pass / ⚠️ Partial / ❌ Fail
**Time:** [Execution time in seconds]
**Notes:** [Observations, issues, unexpected behavior]
```

---

## Workflow Testing

---

## 1. Argument Workflow

**Purpose:** Structured dialectical reasoning through three-role analysis (Creator/Skeptic/Moderator)

### 1.1 Workflow Layer Tests

> **Note:** A bug was discovered during testing where the `temperature` parameter was not being passed to the provider in the `ArgumentWorkflow`. This has been fixed in `model_chorus/src/model_chorus/workflows/argument/argument_workflow.py`.

#### Test 1.1.1: Basic Dialectical Analysis (Happy Path)

**Scenario:** Analyze a simple policy decision

**Python Implementation:**

```python
from model_chorus.workflows.argument import ArgumentWorkflow
from model_chorus.providers import ClaudeProvider
from model_chorus.core.conversation import ConversationMemory

# Setup
provider = ClaudeProvider()
memory = ConversationMemory()
workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

# Execute
result = await workflow.run(
    prompt="Should remote work be mandatory for software companies?",
    temperature=0.7
)

# Verify
if result.success:
    print(f"Session ID: {result.metadata.get('thread_id')}")
    print(f"Provider: {result.metadata.get('provider')}")
    print(f"Roles: {result.metadata.get('roles_executed')}")
    print(f"\nResult:\n{result.synthesis}")
else:
    print(f"Workflow failed: {result.error}")
```

**Expected Output:**
- Session ID format: `argument-thread-[uuid]`
- Roles executed: `['creator', 'skeptic', 'moderator']`
- Result structure:
  ```
  CREATOR (Pro):
  [Arguments supporting remote work...]

  SKEPTIC (Con):
  [Counter-arguments and challenges...]

  MODERATOR (Synthesis):
  [Balanced analysis and recommendations...]
  ```

**Success Criteria:**
- ✅ All three roles present in output
- ✅ Creator presents pro arguments
- ✅ Skeptic challenges assumptions
- ✅ Moderator provides balanced synthesis
- ✅ Session ID generated for threading
- ✅ Token usage tracked in metadata
- ✅ Execution time: 30-90 seconds

---

#### Test 1.1.2: Conversation Threading

**Scenario:** Continue previous argument with follow-up question

```python
# Using session_id from previous test
result1 = await workflow.run(
    prompt="Should remote work be mandatory for software companies?",
    temperature=0.7
)
session_id = result1.metadata.get('thread_id')

result2 = await workflow.run(
    prompt="What about hybrid work models as a compromise?",
    continuation_id=session_id,
    temperature=0.7
)

if result2.success:
    print(f"Same session: {result2.metadata.get('thread_id') == session_id}")
    print(f"Round 2 result:\n{result2.synthesis}")
else:
    print(f"Workflow failed: {result2.error}")
```

**Expected Output:**
- Same session ID maintained
- Moderator references previous analysis
- Builds on prior context

**Success Criteria:**
- ✅ Session ID matches previous
- ✅ Context from round 1 acknowledged
- ✅ Analysis incorporates prior discussion

---

#### Test 1.1.3: File Context Integration

**Scenario:** Analyze code architecture decision with file context

```python
result = await workflow.run(
    prompt="Should we adopt microservices architecture for this codebase?",
    files=["test_data/files/sample.py"],
    temperature=0.7
)
```

**Expected Output:**
- Analysis references file content
- Specific code patterns mentioned

**Success Criteria:**
- ✅ File content incorporated into analysis
- ✅ Specific references to code structure

---

#### Test 1.1.4: Edge Case - Very Short Prompt

**Scenario:** Minimal prompt to test handling

```python
result = await workflow.run(
    prompt="AI?",
    temperature=0.7
)
```

**Expected Output:**
- Workflow requests clarification or
- Provides general AI analysis

**Success Criteria:**
- ✅ Does not crash
- ✅ Generates coherent response

---

#### Test 1.1.5: Edge Case - Very Long Prompt

**Scenario:** Lengthy, complex argument

```python
long_prompt = """
Evaluate this comprehensive proposal for organizational change:
1. Restructure teams by product lines instead of functional areas
2. Implement quarterly OKR cycles with strict accountability
3. Mandate 20% time for innovation projects
4. Establish cross-functional squads with embedded QA
5. Move to continuous deployment with feature flags
6. Create centralized platform team for shared infrastructure
Consider technical debt, team morale, hiring constraints, and market competition.
"""

result = await workflow.run(prompt=long_prompt, temperature=0.7)
```

**Expected Output:**
- Comprehensive analysis covering all points
- Structured debate on each aspect

**Success Criteria:**
- ✅ All proposal points addressed
- ✅ Coherent multi-faceted analysis

---

#### Test 1.1.6: Error Handling - Invalid Temperature

**Scenario:** Out-of-range temperature value

```python
from pydantic import ValidationError

try:
    result = await workflow.run(
        prompt="Test prompt",
        temperature=2.5  # Invalid
    )
    if not result.success:
        print(f"✅ Expected error: {result.error}")
    else:
        print("Unexpected success.")
except ValidationError as e:
    print(f"✅ Expected error: {e}")
```

**Expected Behavior:**
- Validation error raised or
- Temperature clamped to valid range (0.0-1.0)

---

#### Test 1.1.7: Different Providers

**Scenario:** Test with Gemini provider

**Note:** The Gemini provider does not support the `temperature` parameter.

```python
from model_chorus.providers import GeminiProvider

gemini = GeminiProvider()
workflow = ArgumentWorkflow(provider=gemini, conversation_memory=memory)

result = await workflow.run(
    prompt="Should we use TypeScript or JavaScript?"
)

print(f"Provider: {result.metadata.get('provider')}")
print(f"Model: {result.metadata.get('model')}")
```

**Success Criteria:**
- ✅ Works with alternative provider
- ✅ Metadata reflects correct provider

---

### 1.2 Agent Layer Tests

#### Test 1.2.1: Agent Invocation via Claude Code

**Scenario:** Use argument agent in Claude Code environment

**Setup:** Create test agent invocation context

```python
# Simulated agent call (as would happen in Claude Code)
agent_input = {
    "prompt": "Should we migrate to Kubernetes?",
    "temperature": 0.7
}

# Agent validation happens here (from agents/argument.md)
# Required: prompt ✅
# Optional: continue (continuation_id)

# Agent invokes skill
from model_chorus.workflows.argument import ArgumentWorkflow
result = await ArgumentWorkflow(provider=provider, conversation_memory=memory).run(**agent_input)
```

**Success Criteria:**
- ✅ Agent validates required parameters
- ✅ Successfully invokes workflow
- ✅ Returns formatted results

---

#### Test 1.2.2: Agent Parameter Validation

**Scenario:** Missing required parameter

```python
# Missing 'prompt'
agent_input = {
    "temperature": 0.7
}

# Expected: Agent should error before invoking workflow
# Implementation in agents/argument.md should catch this
```

**Success Criteria:**
- ✅ Validation error for missing prompt
- ✅ Clear error message

---

#### Test 1.2.3: Agent Threading Support

**Scenario:** Continue conversation via agent

```python
# First call
result1 = await workflow.run(prompt="Should we use GraphQL?")
session_id = result1['session_id']

# Continuation via agent
agent_input = {
    "prompt": "What about REST vs GraphQL?",
    "continue": session_id
}

result2 = await workflow.run(
    prompt=agent_input["prompt"],
    continuation_id=agent_input["continue"]
)
```

**Success Criteria:**
- ✅ Session ID maintained
- ✅ Context preserved

---

### 1.3 CLI Layer Tests

#### Test 1.3.1: Basic CLI Execution

**Note:** The CLI does not currently support the `--temperature` option.

**Command:**

```bash
model-chorus argument "Should we adopt TDD practices?" --provider claude
```

**Expected Output:**

```
CREATOR (Pro):
[Arguments for TDD...]

SKEPTIC (Con):
[Challenges and costs of TDD...]

MODERATOR (Synthesis):
[Balanced recommendation...]

Session ID: argument-thread-xxxxx
Provider: claude
Model: claude-sonnet-4-5
```

**Success Criteria:**
- ✅ Clean formatted output
- ✅ Session ID displayed
- ✅ All three roles visible
- ✅ Exit code 0

---

#### Test 1.3.2: CLI with File Context

**Command:**

```bash
model-chorus argument \
  --prompt "Should we refactor this code?" \
  --file test_data/files/sample.py \
  --provider claude
```

**Success Criteria:**
- ✅ File content read successfully
- ✅ Analysis references code
- ✅ Exit code 0

---

#### Test 1.3.3: CLI Continuation

**Blocked:** This test is currently blocked. The `model-chorus` CLI does not support JSON output, which prevents capturing the `session_id` required for testing conversation continuation. A feature request should be made to add JSON output to the CLI.

---

#### Test 1.3.4: CLI Error Handling

**Command:**

```bash
# Missing required prompt
model-chorus argument --provider claude
```

**Expected Output:**

```
Error: Missing required parameter: prompt
Usage: model-chorus argument --prompt "..." [OPTIONS]
```

**Success Criteria:**
- ✅ Clear error message
- ✅ Exit code 1 or 2
- ✅ Usage hint provided

---

## 2. Chat Workflow

**Purpose:** Single-model conversational interaction with threading support

### 2.1 Workflow Layer Tests

#### Test 2.1.1: Basic Conversation (Happy Path)

**Scenario:** Simple question-answer interaction

```python
from model_chorus.workflows.chat import ChatWorkflow

workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

result = await workflow.run(
    prompt="Explain the difference between async and sync Python code.",
    temperature=0.5
)

if result.success:
    print(f"Session ID: {result.metadata.get('thread_id')}")
    print(f"Response:\n{result.synthesis}")
else:
    print(f"Workflow failed: {result.error}")
```

**Expected Output:**
- Session ID format: `thread-[uuid]`
- Clear explanation of async vs sync
- Code examples likely included

**Success Criteria:**
- ✅ Coherent, relevant response
- ✅ Session ID generated
- ✅ Metadata includes provider and model
- ✅ Execution time: 5-20 seconds

---

#### Test 2.1.2: Multi-Turn Conversation

**Scenario:** Follow-up questions maintaining context

```python
# Turn 1
result1 = await workflow.run(
    prompt="What is a Python decorator?"
)
session_id = result1.metadata.get('thread_id')
print(f"Turn 1 Session ID: {session_id}")
print(f"Turn 1 Response:\n{result1.synthesis}")

# Turn 2
result2 = await workflow.run(
    prompt="Can you show me a practical example?",
    continuation_id=session_id
)
print(f"Turn 2 Session ID: {result2.metadata.get('thread_id')}")
print(f"Turn 2 Response:\n{result2.synthesis}")

# Turn 3
result3 = await workflow.run(
    prompt="How does it compare to class-based approaches?",
    continuation_id=session_id
)
print(f"Turn 3 Session ID: {result3.metadata.get('thread_id')}")
print(f"Turn 3 Response:\n{result3.synthesis}")
```

**Success Criteria:**
- ✅ Same session ID across turns
- ✅ Turn 2 provides example (builds on turn 1)
- ✅ Turn 3 compares decorator to classes (context maintained)

---

#### Test 2.1.3: File Context

**Scenario:** Ask about specific code file

```python
result = await workflow.run(
    prompt="What does this code do? Can you improve it?",
    files=["test_data/files/sample.py"]
)
```

**Success Criteria:**
- ✅ References file content
- ✅ Explains functionality
- ✅ Provides improvement suggestions

---

#### Test 2.1.4: System Prompt Override

**Scenario:** Custom system instructions

```python
result = await workflow.run(
    prompt="Explain machine learning.",
    system_prompt="You are a teacher explaining to a 10-year-old. Use simple analogies."
)
```

**Success Criteria:**
- ✅ Response uses simple language
- ✅ Includes analogies
- ✅ Age-appropriate explanation

---

> **Note:** A bug was discovered during testing where the `ChatWorkflow` would time out when given an empty prompt. This has been fixed in `model_chorus/src/model_chorus/workflows/chat.py` by adding a validation check to the `run` method.

#### Test 2.1.5: Edge Case - Empty Prompt

**Scenario:** Empty or whitespace-only prompt

```python
try:
    result = await workflow.run(prompt="   ")
except ValueError as e:
    print(f"✅ Expected validation error: {e}")
```

**Expected Behavior:**
- Validation error or
- Request for clarification

---

#### Test 2.1.6: Different Providers

**Scenario:** Test with Gemini

```python
from model_chorus.providers import GeminiProvider

gemini_workflow = ChatWorkflow(
    provider=GeminiProvider(),
    conversation_memory=ConversationMemory()
)

result = await gemini_workflow.run(
    prompt="What are the benefits of functional programming?"
)

print(f"Provider: {result['metadata']['provider']}")
```

**Success Criteria:**
- ✅ Works with alternative provider
- ✅ Quality response generated

---

### 2.2 Agent Layer Tests

#### Test 2.2.1: Agent Basic Invocation

**Scenario:** Simple agent call

```python
agent_input = {
    "prompt": "How do I handle errors in async Python?",
    "provider": "claude"
}

result = await ChatWorkflow(provider=provider, conversation_memory=memory).run(
    prompt=agent_input["prompt"]
)
```

**Success Criteria:**
- ✅ Agent validates parameters
- ✅ Returns formatted response

---

#### Test 2.2.2: Agent Threading

**Scenario:** Continue via agent

```python
# Initial
result1 = await workflow.run(prompt="What is REST?")

# Continue
agent_input = {
    "prompt": "How does it differ from GraphQL?",
    "continue": result1['session_id']
}

result2 = await workflow.run(
    prompt=agent_input["prompt"],
    continuation_id=agent_input["continue"]
)
```

**Success Criteria:**
- ✅ Context maintained
- ✅ Comparison builds on REST explanation

---

### 2.3 CLI Layer Tests

#### Test 2.3.1: Basic CLI Chat

**Command:**

```bash
model-chorus chat \
  --prompt "What is the difference between git merge and git rebase?" \
  --provider claude
```

**Expected Output:**

```
[Clear explanation of merge vs rebase with examples]

Session ID: thread-xxxxx
Provider: claude
Model: claude-sonnet-4-5
```

**Success Criteria:**
- ✅ Quality explanation
- ✅ Session ID provided
- ✅ Exit code 0

---

#### Test 2.3.2: CLI Multi-Turn

**Note:** The `model-chorus` CLI does not currently support JSON output, so the session ID must be manually extracted from the output of the first command.

**Commands:**

```bash
# Turn 1
model-chorus chat "Explain Python generators" --provider claude
# Manually copy the session ID from the output

# Turn 2
model-chorus chat "Show me a practical use case" --continue <session_id> --provider claude
```

**Success Criteria:**
- ✅ Session maintained
- ✅ Follow-up relevant to generators

---

#### Test 2.3.3: CLI with Files

**Command:**

```bash
model-chorus chat \
  --prompt "Review this code for best practices" \
  --files test_data/files/sample.py \
  --provider claude
```

**Success Criteria:**
- ✅ Code analyzed
- ✅ Best practices mentioned
- ✅ Exit code 0

---

## 3. Consensus Workflow

**Purpose:** Multi-model consultation with parallel execution and configurable synthesis strategies

### 3.1 Workflow Layer Tests

#### Test 3.1.1: All Responses Strategy (Happy Path)

**Scenario:** Get separate responses from multiple models

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
gemini_provider = GeminiProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

# Execute
request = GenerationRequest(prompt="What are the key principles of good API design?")
result = await workflow.execute(request, strategy=ConsensusStrategy.ALL_RESPONSES)

# Verify
if result.consensus_response:
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Providers succeeded: {[p.provider_name for p in workflow.get_providers()]}")
    print(f"\nResult:\n{result.consensus_response}")
else:
    print("Workflow failed.")
```

**Expected Output:**

```
=== CLAUDE ===
[Claude's response on API design...]

=== GEMINI ===
[Gemini's response on API design...]

Strategy: all_responses
Providers: ['claude', 'gemini']
Execution time: X.XX seconds
```

**Success Criteria:**
- ✅ Both providers' responses included
- ✅ Responses clearly labeled
- ✅ Metadata shows both succeeded
- ✅ Execution time: 10-30 seconds (parallel)
- ✅ No session ID (consensus doesn't support threading)

---

#### Test 3.1.2: Synthesize Strategy

**Scenario:** Combine multiple perspectives

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
gemini_provider = GeminiProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

# Execute
request = GenerationRequest(prompt="Should we use SQL or NoSQL for a social media application?")
result = await workflow.execute(request, strategy=ConsensusStrategy.SYNTHESIZE)

# Verify
if result.consensus_response:
    print(f"Synthesized result:\n{result.consensus_response}")
else:
    print("Workflow failed.")
```

**Expected Output:**

```
Based on consultation with multiple AI models:

1. [First perspective on SQL vs NoSQL]
2. [Second perspective]
3. [Third perspective]

[Synthesis of key themes and recommendations]
```

**Success Criteria:**
- ✅ Structured numbered perspectives
- ✅ Synthesis section present
- ✅ Multiple viewpoints represented

---

#### Test 3.1.3: Majority Strategy

**Scenario:** Most common response wins

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
gemini_provider = GeminiProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

# Execute
request = GenerationRequest(prompt="Is Python pass-by-value or pass-by-reference?")
result = await workflow.execute(request, strategy=ConsensusStrategy.MAJORITY)

# Verify
if result.consensus_response:
    print(f"Majority answer:\n{result.consensus_response}")
else:
    print("Workflow failed.")
```

**Expected Output:**
- Single answer (the most common one)
- Typically: "Pass-by-object-reference" or similar

**Success Criteria:**
- ✅ Single unified answer
- ✅ Not multiple separate responses

---

#### Test 3.1.4: Weighted Strategy

**Scenario:** Most comprehensive response

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
gemini_provider = GeminiProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

# Execute
request = GenerationRequest(prompt="Explain the CAP theorem.")
result = await workflow.execute(request, strategy=ConsensusStrategy.WEIGHTED)

# Verify
if result.consensus_response:
    print(f"Most comprehensive:\n{result.consensus_response}")
else:
    print("Workflow failed.")
```

**Success Criteria:**
- ✅ Longest/most detailed response selected
- ✅ Single response (not multiple)

---

#### Test 3.1.5: First Valid Strategy

**Scenario:** Fastest response with failover

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
gemini_provider = GeminiProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

# Execute
request = GenerationRequest(prompt="What is the time complexity of binary search?")
result = await workflow.execute(request, strategy=ConsensusStrategy.FIRST_VALID)

# Verify
if result.consensus_response:
    print(f"First valid response:\n{result.consensus_response}")
    print(f"First provider: {list(result.provider_results.keys())[0]}")
else:
    print("Workflow failed.")
```

**Success Criteria:**
- ✅ Single response from fastest provider
- ✅ Execution faster than all_responses
- ✅ Only one provider in succeeded list

---

#### Test 3.1.6: Edge Case - Single Provider

**Scenario:** Consensus with only one provider

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
workflow = ConsensusWorkflow(providers=[claude_provider])

# Execute
request = GenerationRequest(prompt="Test prompt")
result = await workflow.execute(request, strategy=ConsensusStrategy.ALL_RESPONSES)

# Verify
if result.consensus_response:
    print(f"Result:\n{result.consensus_response}")
else:
    print("Workflow failed.")
```

**Success Criteria:**
- ✅ Works with single provider
- ✅ No errors

---

#### Test 3.1.7: Error Handling - Provider Failure

**Scenario:** One provider fails

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, ModelProvider
from model_chorus.providers.base_provider import GenerationRequest

class InvalidProvider(ModelProvider):
    def __init__(self):
        super().__init__("invalid-provider", "invalid-model")

    async def generate(self, request: GenerationRequest, **kwargs) -> str:
        raise ValueError("This provider is invalid")

    def supports_vision(self) -> bool:
        return False

# Setup
claude_provider = ClaudeProvider()
invalid_provider = InvalidProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, invalid_provider])

# Execute
request = GenerationRequest(prompt="Test prompt")
result = await workflow.execute(request, strategy=ConsensusStrategy.SYNTHESIZE)

# Verify
print(f"Succeeded: {[p.provider_name for p in workflow.get_providers() if p.provider_name not in result.failed_providers]}")
print(f"Failed: {result.failed_providers}")
if result.consensus_response:
    print(f"Result:\n{result.consensus_response}")
else:
    print("Workflow failed to produce a result.")
```

**Expected Behavior:**
- ✅ Claude succeeds, invalid fails
- ✅ Result still generated from successful provider
- ✅ Metadata shows which failed

---

#### Test 3.1.8: Timeout Handling

**Scenario:** Custom timeout

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, ModelProvider
from model_chorus.providers.base_provider import GenerationRequest, GenerationResponse
import asyncio

class SlowProvider(ModelProvider):
    def __init__(self):
        super().__init__("slow-provider", "slow-model")

    async def generate(self, request: GenerationRequest, **kwargs) -> GenerationResponse:
        await asyncio.sleep(10)
        return GenerationResponse(content="This should not be returned", model="slow-model")

    def supports_vision(self) -> bool:
        return False

# Setup
claude_provider = ClaudeProvider()
slow_provider = SlowProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, slow_provider], default_timeout=5.0)

# Execute
request = GenerationRequest(prompt="Complex question requiring deep analysis...")
result = await workflow.execute(request)

# Verify
print(f"Succeeded: {[p for p in result.provider_results.keys()]}")
print(f"Failed: {result.failed_providers}")
if result.consensus_response:
    print(f"Result:\n{result.consensus_response}")
else:
    print("Workflow failed to produce a result.")
```

**Success Criteria:**
- ✅ Respects timeout setting
- ✅ Some providers may timeout
- ✅ Returns partial results if available

---

#### Test 3.1.9: File Context with Consensus

**Scenario:** Multiple models analyze same file

**Note:** The `GenerationRequest` object does not currently support a `files` parameter, so the file content must be manually read and added to the prompt.

```python
from model_chorus.workflows.consensus import ConsensusWorkflow, ConsensusStrategy
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.providers.base_provider import GenerationRequest

# Setup
claude_provider = ClaudeProvider()
gemini_provider = GeminiProvider()
workflow = ConsensusWorkflow(providers=[claude_provider, gemini_provider])

# Read file content
with open("test_data/files/sample.py", "r") as f:
    file_content = f.read()

# Execute
request = GenerationRequest(
    prompt=f"What improvements would you suggest for this code?\n\n```python\n{file_content}\n```"
)
result = await workflow.execute(request, strategy=ConsensusStrategy.ALL_RESPONSES)

# Verify
if result.consensus_response:
    print(f"Result:\n{result.consensus_response}")
else:
    print("Workflow failed.")
```

**Success Criteria:**
- ✅ Both models reference file
- ✅ Different perspectives on improvements

---

### 3.2 CLI Layer Tests

#### Test 3.2.1: Basic Consensus CLI

**Note:** To specify multiple providers, the `--provider` option must be repeated for each one.

**Command:**

```bash
model-chorus consensus "What are SOLID principles?" \
  --provider claude \
  --provider gemini \
  --strategy all_responses
```

**Expected Output:**

```
            Consensus Results
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Provider ┃ Status    ┃ Response Length ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ claude   │ ✓ Success │      ... chars │
│ gemini   │ ✓ Success │      ... chars │
└──────────┴───────────┴─────────────────┘

Consensus Response:

## CLAUDE
[Response from Claude...]

---

## GEMINI
[Response from Gemini...]
```

**Success Criteria:**
- ✅ Both responses visible
- ✅ Metadata displayed in a table
- ✅ Exit code 0

---

#### Test 3.2.2: CLI Synthesize Strategy

**Command:**

```bash
model-chorus consensus \
  --prompt "Compare React vs Vue for a new project" \
  --provider claude gemini \
  --strategy synthesize
```

**Success Criteria:**
- ✅ Synthesized output
- ✅ Multiple perspectives combined
- ✅ Exit code 0

---

#### Test 3.2.3: CLI with Files

**Command:**

```bash
model-chorus consensus \
  --prompt "Analyze this code for security issues" \
  --files test_data/files/sample.py \
  --provider claude gemini \
  --strategy all_responses
```

**Success Criteria:**
- ✅ Both models analyze file
- ✅ Security insights provided
- ✅ Exit code 0

---

#### Test 3.2.4: CLI Error - Invalid Strategy

**Command:**

```bash
model-chorus consensus \
  --prompt "Test" \
  --strategy invalid_strategy
```

**Expected Output:**

```
Error: Invalid strategy 'invalid_strategy'
Valid strategies: all_responses, synthesize, majority, weighted, first_valid
```

**Success Criteria:**
- ✅ Clear error message
- ✅ Lists valid options
- ✅ Exit code 1

---

## 4. Ideate Workflow

**Purpose:** Creative brainstorming and idea generation with configurable creativity levels and structured output

### 4.1 Workflow Layer Tests

#### Test 4.1.1: Standard Ideation (Happy Path)

**Scenario:** Generate default number of ideas

```python
from model_chorus.workflows.ideate import IdeateWorkflow

workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

result = await workflow.run(
    prompt="Generate features for a productivity app",
    num_ideas=5,
    temperature=0.7
)

print(f"Session ID: {result['session_id']}")
print(f"Ideas requested: {result['metadata']['num_ideas_requested']}")
print(f"Ideas generated: {result['metadata']['num_ideas_generated']}")
print(f"\nIdeas:\n{result['result']}")
```

**Expected Output:**

```
IDEA 1: [Title]
[Description...]

IDEA 2: [Title]
[Description...]

...

SYNTHESIS:
[Analysis of common themes, recommendations, prioritization]

Session ID: ideate-thread-xxxxx
Ideas: 5/5
Temperature: 0.7
```

**Success Criteria:**
- ✅ Exactly 5 ideas generated
- ✅ Each idea has title and description
- ✅ Synthesis section present
- ✅ Session ID for threading
- ✅ Execution time: 20-45 seconds

---

#### Test 4.1.2: High Creativity Mode

**Scenario:** Innovative, bold ideas

```python
result = await workflow.run(
    prompt="Innovative ways to reduce carbon emissions in cities",
    num_ideas=6,
    temperature=0.9
)
```

**Expected Output:**
- Creative, unconventional ideas
- Higher risk/reward concepts
- Potentially disruptive approaches

**Success Criteria:**
- ✅ Ideas are innovative and bold
- ✅ Synthesis acknowledges risk levels

---

#### Test 4.1.3: Low Creativity Mode

**Scenario:** Conservative, practical ideas

```python
result = await workflow.run(
    prompt="Improve our customer onboarding process",
    num_ideas=4,
    temperature=0.3
)
```

**Expected Output:**
- Practical, incremental improvements
- Low-risk approaches
- Proven patterns

**Success Criteria:**
- ✅ Ideas are conservative and practical
- ✅ Focus on optimizations vs innovations

---

#### Test 4.1.4: Large Idea Count

**Scenario:** Exhaustive ideation

```python
result = await workflow.run(
    prompt="Ways to improve developer productivity",
    num_ideas=12,
    temperature=0.7
)

print(f"Generated: {result['metadata']['num_ideas_generated']}")
```

**Success Criteria:**
- ✅ All 12 ideas generated
- ✅ Ideas remain distinct and valuable
- ✅ Synthesis handles large set

---

#### Test 4.1.5: Small Idea Count

**Scenario:** Quick exploration

```python
result = await workflow.run(
    prompt="Quick wins for website performance",
    num_ideas=3,
    temperature=0.5
)
```

**Success Criteria:**
- ✅ Exactly 3 ideas
- ✅ Ideas are high-impact
- ✅ Faster execution than default

---

#### Test 4.1.6: Conversation Threading

**Scenario:** Iterative refinement

```python
# Round 1
result1 = await workflow.run(
    prompt="Mobile app features for fitness tracking",
    num_ideas=5
)

# Round 2 - refine based on constraints
result2 = await workflow.run(
    prompt="From those ideas, which work best for beginners? Expand on those.",
    continuation_id=result1.metadata.get('thread_id'),
    num_ideas=3
)
```

**Success Criteria:**
- ✅ Round 2 references round 1 ideas
- ✅ Refinement apparent
- ✅ Same session ID

---

#### Test 4.1.7: Constrained Ideation

**Scenario:** Ideas with specific constraints

```python
result = await workflow.run(
    prompt="Low-cost marketing strategies for a B2B SaaS startup",
    system_prompt="Focus on strategies under $1000/month budget. Prioritize organic growth.",
    num_ideas=5
)
```

**Success Criteria:**
- ✅ Ideas respect budget constraint
- ✅ Focus on organic strategies
- ✅ Synthesis acknowledges constraints

---

#### Test 4.1.8: File Context Ideation

**Scenario:** Generate ideas based on existing code

```python
result = await workflow.run(
    prompt="Suggest improvements and new features for this code",
    files=["test_data/files/sample.py"],
    num_ideas=4
)
```

**Success Criteria:**
- ✅ Ideas reference actual code
- ✅ Suggestions are specific
- ✅ Mix of improvements and features

---

#### Test 4.1.9: Edge Case - Invalid Idea Count

**Scenario:** Out of range num_ideas

```python
try:
    result = await workflow.run(
        prompt="Test",
        num_ideas=0  # Invalid
    )
except ValueError as e:
    print(f"✅ Expected error: {e}")
```

**Expected Behavior:**
- Validation error for num_ideas < 1

---

### 4.2 Agent Layer Tests

#### Test 4.2.1: Agent Basic Ideation

```python
agent_input = {
    "prompt": "New product ideas for developers",
    "num_ideas": 5
}

result = await workflow.run(**agent_input)
```

**Success Criteria:**
- ✅ Agent validates parameters
- ✅ 5 ideas generated

---

#### Test 4.2.2: Agent with Threading

```python
# Initial
result1 = await workflow.run(
    prompt="Features for a code editor",
    num_ideas=5
)

# Continue
agent_input = {
    "prompt": "Focus on AI-assisted features",
    "continue": result1['session_id'],
    "num_ideas": 3
}

result2 = await workflow.run(
    prompt=agent_input["prompt"],
    continuation_id=agent_input["continue"],
    num_ideas=agent_input["num_ideas"]
)
```

**Success Criteria:**
- ✅ Refinement to AI features
- ✅ Context maintained

---

### 4.3 CLI Layer Tests

#### Test 4.3.1: Basic Ideation CLI

**Command:**

```bash
model-chorus ideate \
  --prompt "Marketing campaign ideas for a new app launch" \
  --num-ideas 5 \
  --temperature 0.7 \
  --provider claude
```

**Expected Output:**

```
IDEA 1: [Title]
[Description]

IDEA 2: [Title]
[Description]

...

SYNTHESIS:
[Analysis and recommendations]

Session ID: ideate-thread-xxxxx
Ideas: 5/5
Temperature: 0.7
```

**Success Criteria:**
- ✅ 5 formatted ideas
- ✅ Synthesis present
- ✅ Exit code 0

---

#### Test 4.3.2: CLI High Creativity

**Command:**

```bash
model-chorus ideate \
  --prompt "Radical innovations for urban transportation" \
  --num-ideas 6 \
  --temperature 0.95 \
  --provider claude
```

**Success Criteria:**
- ✅ Bold, innovative ideas
- ✅ Higher creativity evident
- ✅ Exit code 0

---

#### Test 4.3.3: CLI with Continuation

**Command:**

```bash
# Round 1
SESSION=$(model-chorus ideate \
  --prompt "E-commerce features" \
  --num-ideas 5 \
  --format json | jq -r '.session_id')

# Round 2
model-chorus ideate \
  --prompt "Focus on mobile-first features" \
  --num-ideas 3 \
  --continue "$SESSION"
```

**Success Criteria:**
- ✅ Refinement to mobile
- ✅ Session maintained

---

#### Test 4.3.4: CLI with Files

**Command:**

```bash
model-chorus ideate \
  --prompt "Suggest refactoring ideas for this code" \
  --files test_data/files/sample.py \
  --num-ideas 4 \
  --provider claude
```

**Success Criteria:**
- ✅ Code-specific ideas
- ✅ References file content

---

## 5. Router Workflow

**Purpose:** Intelligent workflow selection that analyzes user requests and recommends optimal ModelChorus workflow

**Note:** Router is a meta-skill that helps decide which workflow to use. Testing focuses on recommendation accuracy.

### 5.1 Workflow Layer Tests

#### Test 5.1.1: Simple Question → Chat

**Scenario:** Router recommends Chat for straightforward question

```python
from model_chorus.workflows.router import RouterWorkflow

# Note: Router may be implemented differently; adjust based on actual API
router = RouterWorkflow(provider=provider)

result = await router.run(
    user_request="What is the difference between REST and GraphQL?"
)

print(f"Recommended workflow: {result['workflow']}")
print(f"Confidence: {result['confidence']}")
print(f"Rationale: {result['rationale']}")
print(f"Parameters: {result['parameters']}")
```

**Expected Output:**

```
Workflow: CHAT
Confidence: high
Rationale: Simple informational question requiring single-model explanation.
Parameters: {
  "prompt": "What is the difference between REST and GraphQL?",
  "temperature": 0.5
}
```

**Success Criteria:**
- ✅ Recommends CHAT workflow
- ✅ Confidence is high
- ✅ Rationale mentions simplicity/information

---

#### Test 5.1.2: Multi-Model Question → Consensus

**Scenario:** Router recommends Consensus for cross-validation

```python
result = await router.run(
    user_request="What are the best practices for securing JWT tokens? I want multiple perspectives to ensure I'm not missing anything."
)

print(f"Recommended: {result['workflow']}")
print(f"Parameters: {result['parameters']}")
```

**Expected Output:**

```
Workflow: CONSENSUS
Confidence: high
Rationale: User explicitly wants multiple perspectives for comprehensive validation.
Parameters: {
  "prompt": "What are best practices for securing JWT tokens?",
  "provider": ["claude", "gemini"],
  "strategy": "synthesize"
}
```

**Success Criteria:**
- ✅ Recommends CONSENSUS
- ✅ Multiple providers suggested
- ✅ Synthesize strategy (for combining perspectives)

---

#### Test 5.1.3: Policy Decision → Argument

**Scenario:** Router recommends Argument for pro/con analysis

```python
result = await router.run(
    user_request="We're considering requiring code reviews for all PRs, even small bug fixes. Should we do this?"
)

print(f"Recommended: {result['workflow']}")
```

**Expected Output:**

```
Workflow: ARGUMENT
Confidence: high
Rationale: Policy decision requiring structured pro/con analysis and balanced recommendation.
Parameters: {
  "prompt": "Should we require code reviews for all PRs, including small bug fixes?",
  "temperature": 0.7
}
```

**Success Criteria:**
- ✅ Recommends ARGUMENT
- ✅ Rationale mentions decision-making/debate

---

#### Test 5.1.4: Brainstorming → Ideate

**Scenario:** Router recommends Ideate for creative generation

```python
result = await router.run(
    user_request="I need creative ideas for improving our developer onboarding process. Give me lots of options."
)

print(f"Recommended: {result['workflow']}")
print(f"Parameters: {result['parameters']}")
```

**Expected Output:**

```
Workflow: IDEATE
Confidence: high
Rationale: Creative brainstorming request with desire for multiple options.
Parameters: {
  "prompt": "Creative ideas for improving developer onboarding",
  "num_ideas": 8,
  "temperature": 0.7
}
```

**Success Criteria:**
- ✅ Recommends IDEATE
- ✅ num_ideas > 5 (user wants "lots")
- ✅ Moderate to high temperature

---

#### Test 5.1.5: Complex Investigation → ThinkDeep

**Scenario:** Router recommends ThinkDeep for systematic debugging

```python
result = await router.run(
    user_request="Our production system is experiencing intermittent 500 errors. The logs show database timeouts, but only during peak hours. I need to systematically investigate the root cause."
)

print(f"Recommended: {result['workflow']}")
```

**Expected Output:**

```
Workflow: THINKDEEP
Confidence: high
Rationale: Complex debugging scenario requiring systematic investigation with evidence gathering and hypothesis tracking.
Parameters: {
  "step": "Begin investigation of intermittent 500 errors with database timeouts during peak hours",
  "step_number": 1,
  "total_steps": 5,
  "hypothesis": "Database connection pool exhaustion during high load",
  "confidence": "exploring",
  "next_step_required": true
}
```

**Success Criteria:**
- ✅ Recommends THINKDEEP
- ✅ Initial hypothesis suggested
- ✅ Multi-step approach indicated

---

#### Test 5.1.6: Multi-Persona Research → Study

**Scenario:** Router recommends Study for collaborative analysis

```python
result = await router.run(
    user_request="I want to understand our authentication system from multiple expert perspectives - security, architecture, and performance."
)

print(f"Recommended: {result['workflow']}")
print(f"Parameters: {result['parameters']}")
```

**Expected Output:**

```
Workflow: STUDY
Confidence: high
Rationale: Requires multiple expert perspectives (security, architecture, performance) for comprehensive analysis.
Parameters: {
  "scenario": "Analyze authentication system",
  "persona": ["SecurityExpert", "Architect", "PerformanceEngineer"],
  "files": []
}
Alternatives: [
  {
    "workflow": "CONSENSUS",
    "rationale": "Could use multiple models instead of personas"
  }
]
```

**Success Criteria:**
- ✅ Recommends STUDY
- ✅ Suggests appropriate personas
- ✅ May offer CONSENSUS as alternative

---

#### Test 5.1.7: Ambiguous Request → Low Confidence

**Scenario:** Router asks clarifying question

```python
result = await router.run(
    user_request="Help me with the API."
)

print(f"Confidence: {result['confidence']}")
print(f"Clarifying question: {result['clarifying_question']}")
```

**Expected Output:**

```
Workflow: CHAT (tentative)
Confidence: low
Clarifying question: Are you looking to (a) design a new API, (b) debug an existing API issue, (c) choose between API approaches, or (d) learn about API concepts?
```

**Success Criteria:**
- ✅ Confidence is low
- ✅ Clarifying question provided
- ✅ Question offers specific options

---

#### Test 5.1.8: Multi-Step Workflow Sequence

**Scenario:** Router suggests workflow sequence

```python
result = await router.run(
    user_request="We need to decide whether to migrate to microservices, generate implementation ideas, then systematically plan the migration."
)

print(f"Primary workflow: {result['workflow']}")
print(f"Sequence: {result['sequence']}")
```

**Expected Output:**

```
Workflow: ARGUMENT
Confidence: medium
Sequence: [
  {
    "step": 1,
    "workflow": "ARGUMENT",
    "purpose": "Evaluate microservices decision with pro/con analysis"
  },
  {
    "step": 2,
    "workflow": "IDEATE",
    "purpose": "Generate implementation approaches and strategies"
  },
  {
    "step": 3,
    "workflow": "THINKDEEP",
    "purpose": "Systematically plan migration steps with validation"
  }
]
```

**Success Criteria:**
- ✅ Multi-step sequence identified
- ✅ Logical workflow progression
- ✅ Each step has clear purpose

---

#### Test 5.1.9: Constraint-Based Routing

**Scenario:** Time-limited constraint affects recommendation

```python
result = await router.run(
    user_request="Quick analysis of whether we should use Redis or Memcached",
    explicit_constraints={"time_limited": True}
)

print(f"Recommended: {result['workflow']}")
print(f"Rationale: {result['rationale']}")
```

**Expected Output:**

```
Workflow: CHAT
Confidence: high
Rationale: Time constraint favors single-model quick response over multi-model consensus.
Alternatives: [
  {
    "workflow": "CONSENSUS",
    "rationale": "Would provide multiple perspectives but takes longer"
  }
]
```

**Success Criteria:**
- ✅ Chooses faster workflow (CHAT)
- ✅ Rationale mentions time constraint
- ✅ Offers more thorough alternative

---

### 5.2 CLI Layer Tests

#### Test 5.2.1: Router CLI Basic

**Command:**

```bash
model-chorus router \
  --request "Should we adopt GraphQL for our API?"
```

**Expected Output:**

```
Recommended Workflow: ARGUMENT
Confidence: high

Rationale: Technology decision requiring structured pro/con analysis.

Suggested Parameters:
  prompt: "Should we adopt GraphQL for our API?"
  temperature: 0.7

How to execute:
  model-chorus argument --prompt "Should we adopt GraphQL for our API?" --temperature 0.7

Alternatives:
  - CONSENSUS: Get multiple model perspectives on GraphQL adoption
```

**Success Criteria:**
- ✅ Clear recommendation
- ✅ Executable command provided
- ✅ Alternatives listed
- ✅ Exit code 0

---

#### Test 5.2.2: Router with Constraints

**Command:**

```bash
model-chorus router \
  --request "Comprehensive security analysis of our auth system" \
  --constraint time_limited
```

**Expected Output:**

```
Recommended Workflow: CHAT (with constraints)
Confidence: medium

Note: Time constraint limits depth. Consider STUDY workflow if time permits for multi-perspective security analysis.
```

**Success Criteria:**
- ✅ Constraint acknowledged
- ✅ Trade-off explained
- ✅ Better alternative mentioned

---

## 6. Study Workflow

**Purpose:** Persona-based collaborative research with intelligent role-based orchestration and conversation threading

**Note:** Study workflow may not have an agent wrapper yet. Focus on workflow and CLI testing.

### 6.1 Workflow Layer Tests

#### Test 6.1.1: Default Two-Persona Investigation (Happy Path)

**Scenario:** Use default Researcher + Critic personas

```python
from model_chorus.workflows.study import StudyWorkflow

workflow = StudyWorkflow(provider=provider)

result = await workflow.run(
    scenario="How does the authentication system in our codebase work?",
    files=["test_data/files/sample.py"]
)

print(f"Thread ID: {result['thread_id']}")
print(f"Round: {result['investigation_round']}")
print(f"Personas: {result['personas_used']}")
print(f"Phase: {result['routing_metadata']['phase']}")
print(f"\nSynthesis:\n{result['synthesis']}")
```

**Expected Output:**

```
RESEARCHER:
[Systematic analysis of auth system...]

CRITIC:
[Challenges assumptions, identifies edge cases...]

SYNTHESIS:
[Integrated findings with key insights]

Thread ID: [uuid]
Round: 1
Personas: ['Researcher', 'Critic']
Phase: exploration
```

**Success Criteria:**
- ✅ Both personas contribute
- ✅ Researcher provides systematic analysis
- ✅ Critic challenges and questions
- ✅ Synthesis integrates both perspectives
- ✅ Thread ID for continuation
- ✅ Execution time: 30-60 seconds

---

#### Test 6.1.2: Custom Persona Selection

**Scenario:** Specify expert personas

```python
result = await workflow.run(
    scenario="Analyze this code for security vulnerabilities and performance bottlenecks",
    persona=["SecurityExpert", "PerformanceEngineer"],
    files=["test_data/files/sample.py"]
)

print(f"Personas: {result['personas_used']}")
```

**Expected Output:**

```
SECURITY EXPERT:
[Security-focused analysis: SQL injection risks, XSS, auth issues...]

PERFORMANCE ENGINEER:
[Performance analysis: N+1 queries, caching, algorithmic complexity...]

SYNTHESIS:
[Combined security + performance recommendations]
```

**Success Criteria:**
- ✅ Security Expert focuses on vulnerabilities
- ✅ Performance Engineer focuses on optimization
- ✅ Synthesis balances both concerns

---

#### Test 6.1.3: Multi-Turn Investigation

**Scenario:** Deep dive across multiple rounds

```python
# Round 1: Exploration
result1 = await workflow.run(
    scenario="Understand the caching strategy in this system",
    files=["test_data/files/sample.py"]
)

# Round 2: Deep dive
result2 = await workflow.run(
    scenario="Investigate cache invalidation edge cases",
    continuation_id=result1['thread_id']
)

# Round 3: Validation
result3 = await workflow.run(
    scenario="Are there race conditions in the cache updates?",
    continuation_id=result1['thread_id']
)

print(f"Round 1 phase: {result1['routing_metadata']['phase']}")
print(f"Round 2 phase: {result2['routing_metadata']['phase']}")
print(f"Round 3 phase: {result3['routing_metadata']['phase']}")
```

**Expected Output:**

```
Round 1: exploration
Round 2: deep_dive
Round 3: validation
```

**Success Criteria:**
- ✅ Phase progression logical
- ✅ Each round builds on previous
- ✅ Same thread ID maintained
- ✅ Investigation deepens over rounds

---

#### Test 6.1.4: Code Review Scenario

**Scenario:** Multi-perspective code review

```python
result = await workflow.run(
    scenario="Comprehensive code review",
    persona=["CodeReviewer", "SecurityExpert", "Architect"],
    files=["test_data/files/sample.py"]
)

print(f"Personas: {result['personas_used']}")
```

**Expected Output:**

```
CODE REVIEWER:
[Style, best practices, readability, testing...]

SECURITY EXPERT:
[Security issues, vulnerabilities...]

ARCHITECT:
[Design patterns, structure, scalability...]

SYNTHESIS:
[Prioritized action items across all dimensions]
```

**Success Criteria:**
- ✅ Each persona focuses on their domain
- ✅ Synthesis prioritizes findings
- ✅ Minimal overlap between personas

---

#### Test 6.1.5: Learning Scenario

**Scenario:** Learn complex topic systematically

```python
result = await workflow.run(
    scenario="Explain how OAuth 2.0 works, from basic concepts to implementation details",
    persona=["TeachingExpert", "PracticalEngineer"]
)
```

**Expected Output:**

```
TEACHING EXPERT:
[Clear explanation with analogies, building from basics...]

PRACTICAL ENGINEER:
[Implementation examples, common pitfalls, real-world usage...]

SYNTHESIS:
[Learning path with theoretical + practical balance]
```

**Success Criteria:**
- ✅ Teaching Expert uses pedagogical approach
- ✅ Practical Engineer provides examples
- ✅ Combined learning experience

---

#### Test 6.1.6: Architecture Decision

**Scenario:** Multi-expert design consultation

```python
result = await workflow.run(
    scenario="Should we use a monolith or microservices architecture for a new project with 5 developers and moderate scale?",
    persona=["Architect", "PracticalEngineer", "PerformanceEngineer"]
)
```

**Success Criteria:**
- ✅ Architect discusses patterns and trade-offs
- ✅ Practical Engineer considers team size and complexity
- ✅ Performance Engineer analyzes scalability
- ✅ Synthesis provides clear recommendation

---

#### Test 6.1.7: Edge Case - Single Persona

**Scenario:** Only one persona (unusual but valid)

```python
result = await workflow.run(
    scenario="Quick security check",
    persona=["SecurityExpert"],
    files=["test_data/files/sample.py"]
)

print(f"Personas: {result['personas_used']}")
```

**Success Criteria:**
- ✅ Works with single persona
- ✅ No errors
- ✅ Synthesis still provided

---

#### Test 6.1.8: Verbose Mode

**Scenario:** Detailed execution information

```python
result = await workflow.run(
    scenario="Test scenario",
    verbose=True
)

# Check for detailed metadata
print(f"Routing decisions: {result['routing_metadata']['routing_decisions']}")
```

**Success Criteria:**
- ✅ Additional debug info in metadata
- ✅ Routing decisions visible

---

#### Test 6.1.9: Different Provider

**Scenario:** Use Gemini instead of default

```python
from model_chorus.providers import GeminiProvider

gemini_workflow = StudyWorkflow(provider=GeminiProvider())

result = await gemini_workflow.run(
    scenario="Analyze this pattern",
    files=["test_data/files/sample.py"]
)

print(f"Provider: {result['provider']}")
print(f"Model: {result['model']}")
```

**Success Criteria:**
- ✅ Works with alternative provider
- ✅ Metadata reflects Gemini

---

### 6.2 CLI Layer Tests

#### Test 6.2.1: Basic Study CLI

**Command:**

```bash
model-chorus study \
  --scenario "How does error handling work in this codebase?" \
  --files test_data/files/sample.py \
  --provider claude
```

**Expected Output:**

```
RESEARCHER:
[Systematic analysis...]

CRITIC:
[Challenges and edge cases...]

SYNTHESIS:
[Key findings and recommendations]

Thread ID: [uuid]
Investigation round: 1
Phase: exploration
```

**Success Criteria:**
- ✅ Both personas visible
- ✅ Synthesis provided
- ✅ Thread ID shown
- ✅ Exit code 0

---

#### Test 6.2.2: CLI with Custom Personas

**Command:**

```bash
model-chorus study \
  --scenario "Security and performance analysis" \
  --persona SecurityExpert PerformanceEngineer \
  --files test_data/files/sample.py
```

**Success Criteria:**
- ✅ Specified personas used
- ✅ Domain-specific analysis
- ✅ Exit code 0

---

#### Test 6.2.3: CLI Multi-Turn

**Command:**

```bash
# Round 1
THREAD=$(model-chorus study \
  --scenario "Understand the data flow" \
  --files test_data/files/sample.py \
  --format json | jq -r '.thread_id')

# Round 2
model-chorus study \
  --scenario "What are potential bottlenecks?" \
  --continue "$THREAD"

# Round 3
model-chorus study \
  --scenario "Suggest optimizations" \
  --continue "$THREAD"
```

**Success Criteria:**
- ✅ Thread maintained across rounds
- ✅ Investigation deepens
- ✅ Context preserved

---

#### Test 6.2.4: CLI Verbose Mode

**Command:**

```bash
model-chorus study \
  --scenario "Test investigation" \
  --verbose
```

**Expected Output:**

```
[Detailed routing information]
[Persona selection decisions]
[Phase transitions]

[Standard output...]
```

**Success Criteria:**
- ✅ Additional debug info shown
- ✅ Routing decisions visible

---

## 7. ThinkDeep Workflow

**Purpose:** Extended reasoning with systematic investigation, hypothesis tracking, and confidence progression

### 7.1 Workflow Layer Tests

#### Test 7.1.1: Single-Step Investigation (Happy Path)

**Scenario:** Simple investigation with immediate conclusion

```python
from model_chorus.workflows.thinkdeep import ThinkDeepWorkflow

workflow = ThinkDeepWorkflow(provider=provider, conversation_memory=memory)

result = await workflow.run(
    step="Investigate why Python function is slow",
    step_number=1,
    total_steps=1,
    findings="Function has nested loop with O(n²) complexity",
    next_step_required=False,  # This is the final step
    hypothesis="Algorithmic complexity is the bottleneck",
    confidence="high"
)

print(f"Session ID: {result['session_id']}")
print(f"Confidence: {result['metadata']['confidence']}")
print(f"Conclusion:\n{result['result']}")
```

**Expected Output:**

```
ANALYSIS:
The investigation confirms algorithmic complexity as the root cause...

CONCLUSION:
High confidence in hypothesis. Recommend refactoring to use hash map for O(n) lookup...

Session ID: thinkdeep-[uuid]
Confidence: high
Next step required: False
```

**Success Criteria:**
- ✅ Analysis references findings
- ✅ Conclusion provided (not next steps)
- ✅ Confidence level reflected in output
- ✅ Session ID generated
- ✅ Execution time: 10-25 seconds

---

#### Test 7.1.2: Multi-Step Investigation

**Scenario:** Complex debugging requiring multiple steps

```python
# Step 1: Initial exploration
result1 = await workflow.run(
    step="Examine server logs for error patterns",
    step_number=1,
    total_steps=4,
    findings="500 errors occur only during 2-3pm, correlate with DB connection pool warnings",
    next_step_required=True,
    hypothesis="",  # No hypothesis yet
    confidence="exploring"
)

# Step 2: Hypothesis formation
result2 = await workflow.run(
    step="Check database connection pool configuration",
    step_number=2,
    total_steps=4,
    findings="Pool size is 10, peak concurrent requests reach 50+",
    continuation_id=result1['session_id'],
    next_step_required=True,
    hypothesis="Connection pool exhaustion during peak load",
    confidence="medium"
)

# Step 3: Validation
result3 = await workflow.run(
    step="Analyze request timing and connection wait times",
    step_number=3,
    total_steps=4,
    findings="Average connection wait time is 5+ seconds during peak, requests timeout after 5s",
    continuation_id=result1['session_id'],
    next_step_required=True,
    hypothesis="Connection pool exhaustion causing request timeouts",
    confidence="high"
)

# Step 4: Confirmation
result4 = await workflow.run(
    step="Test with increased pool size in staging",
    step_number=4,
    total_steps=4,
    findings="500 errors eliminated with pool size of 30",
    continuation_id=result1['session_id'],
    next_step_required=False,
    hypothesis="Insufficient connection pool size is the root cause",
    confidence="certain"
)

print(f"Final confidence: {result4['metadata']['confidence']}")
print(f"Final conclusion:\n{result4['result']}")
```

**Expected Output:**

```
Step 1: Exploring...
Step 2: Hypothesis formed...
Step 3: Evidence mounting...
Step 4: CONFIRMED

Root cause: Insufficient database connection pool size (10) for peak load (50+ concurrent requests).

Recommendation: Increase pool size to 30-50 based on peak traffic patterns.

Confidence: certain
```

**Success Criteria:**
- ✅ Same session ID throughout
- ✅ Confidence progresses: exploring → medium → high → certain
- ✅ Hypothesis evolves and strengthens
- ✅ Final step provides clear recommendation
- ✅ All findings incorporated

---

#### Test 7.1.3: Hypothesis Evolution

**Scenario:** Initial hypothesis proven wrong, new one formed

```python
# Step 1: Wrong hypothesis
result1 = await workflow.run(
    step="Check for memory leaks",
    step_number=1,
    total_steps=3,
    findings="Memory usage stable, no leaks detected",
    next_step_required=True,
    hypothesis="Memory leak causing crashes",
    confidence="low"
)

# Step 2: Pivot
result2 = await workflow.run(
    step="Examine CPU usage patterns",
    step_number=2,
    total_steps=3,
    findings="CPU spikes to 100% during crashes, deadlock in thread pool",
    continuation_id=result1['session_id'],
    next_step_required=True,
    hypothesis="Deadlock in thread pool causing crashes",  # New hypothesis
    confidence="medium"
)

# Step 3: Confirm
result3 = await workflow.run(
    step="Reproduce deadlock with thread dumps",
    step_number=3,
    total_steps=3,
    findings="Confirmed circular wait between threads A and B",
    continuation_id=result1['session_id'],
    next_step_required=False,
    hypothesis="Thread deadlock is the root cause",
    confidence="very_high"
)
```

**Success Criteria:**
- ✅ Initial hypothesis acknowledged as incorrect
- ✅ Pivot to new hypothesis noted
- ✅ Confidence resets appropriately
- ✅ Final analysis explains the journey

---

#### Test 7.1.4: File Tracking

**Scenario:** Track which files have been examined

```python
result = await workflow.run(
    step="Review authentication logic",
    step_number=1,
    total_steps=2,
    findings="JWT verification missing expiration check",
    files_checked="auth/jwt_handler.py,auth/middleware.py",
    relevant_files="auth/jwt_handler.py",
    next_step_required=True,
    hypothesis="JWT expiration not validated",
    confidence="high"
)

print(f"Files checked: {result['metadata']['files_checked']}")
print(f"Relevant files: {result['metadata']['relevant_files']}")
```

**Success Criteria:**
- ✅ Files tracked in metadata
- ✅ Distinction between checked and relevant

---

#### Test 7.1.5: Context Tracking

**Scenario:** Track relevant functions/methods

```python
result = await workflow.run(
    step="Identify vulnerable code paths",
    step_number=1,
    total_steps=2,
    findings="Input validation missing in user creation endpoint",
    relevant_context="UserController.create_user,UserValidator.validate_email",
    next_step_required=True,
    hypothesis="Missing input validation allows injection",
    confidence="medium"
)

print(f"Relevant context: {result['metadata']['relevant_context']}")
```

**Success Criteria:**
- ✅ Context tracked in metadata
- ✅ Function/method names preserved

---

#### Test 7.1.6: Issue Tracking

**Scenario:** Track discovered issues with severity

```python
import json

issues = [
    {"description": "SQL injection in search endpoint", "severity": "high"},
    {"description": "Missing rate limiting", "severity": "medium"}
]

result = await workflow.run(
    step="Security audit findings",
    step_number=1,
    total_steps=1,
    findings="Multiple security issues identified",
    issues_found=json.dumps(issues),
    next_step_required=False,
    hypothesis="Multiple security vulnerabilities present",
    confidence="high"
)

print(f"Issues: {result['metadata']['issues_found']}")
```

**Success Criteria:**
- ✅ Issues parsed and tracked
- ✅ Severity levels preserved

---

#### Test 7.1.7: Thinking Mode Configuration

**Scenario:** Different thinking depths

```python
# Minimal thinking
result_min = await workflow.run(
    step="Quick check",
    step_number=1,
    total_steps=1,
    findings="No issues",
    thinking_mode="minimal",
    next_step_required=False
)

# Maximum thinking
result_max = await workflow.run(
    step="Deep analysis",
    step_number=1,
    total_steps=1,
    findings="Complex issue",
    thinking_mode="max",
    next_step_required=False
)

print(f"Minimal mode time: {result_min.get('execution_time', 'N/A')}")
print(f"Max mode time: {result_max.get('execution_time', 'N/A')}")
```

**Success Criteria:**
- ✅ Minimal mode faster
- ✅ Max mode more thorough
- ✅ Metadata reflects thinking mode

---

#### Test 7.1.8: Edge Case - Very Long Investigation

**Scenario:** Many steps with persistence

```python
session_id = None

for i in range(1, 11):  # 10 steps
    result = await workflow.run(
        step=f"Investigation step {i}",
        step_number=i,
        total_steps=10,
        findings=f"Finding from step {i}",
        continuation_id=session_id,
        next_step_required=(i < 10),
        hypothesis="Working hypothesis",
        confidence="medium" if i < 9 else "high"
    )
    session_id = result['session_id']

print(f"Completed 10-step investigation: {session_id}")
```

**Success Criteria:**
- ✅ All steps complete
- ✅ Session maintained
- ✅ No degradation over steps

---

#### Test 7.1.9: Error Handling - Missing Required Fields

**Scenario:** Missing required parameters

```python
try:
    result = await workflow.run(
        step="Test",
        # Missing step_number, total_steps, findings, next_step_required
    )
except (ValueError, TypeError) as e:
    print(f"✅ Expected validation error: {e}")
```

**Expected Behavior:**
- Validation error for missing fields

---

### 7.2 Agent Layer Tests

#### Test 7.2.1: Agent Basic Investigation

```python
agent_input = {
    "step": "Identify performance bottleneck",
    "step_number": 1,
    "total_steps": 3,
    "findings": "Database query taking 2+ seconds",
    "next_step_required": True,
    "hypothesis": "Missing database index",
    "confidence": "medium"
}

result = await workflow.run(**agent_input)
```

**Success Criteria:**
- ✅ Agent validates all required fields
- ✅ Investigation proceeds

---

#### Test 7.2.2: Agent Multi-Step

```python
# Step 1
result1 = await workflow.run(
    step="Initial investigation",
    step_number=1,
    total_steps=2,
    findings="First finding",
    next_step_required=True,
    hypothesis="Initial hypothesis",
    confidence="low"
)

# Step 2 via agent
agent_input = {
    "step": "Confirmation",
    "step_number": 2,
    "total_steps": 2,
    "findings": "Confirmed",
    "continue": result1['session_id'],
    "next_step_required": False,
    "hypothesis": "Confirmed hypothesis",
    "confidence": "high"
}

result2 = await workflow.run(
    step=agent_input["step"],
    step_number=agent_input["step_number"],
    total_steps=agent_input["total_steps"],
    findings=agent_input["findings"],
    continuation_id=agent_input["continue"],
    next_step_required=agent_input["next_step_required"],
    hypothesis=agent_input["hypothesis"],
    confidence=agent_input["confidence"]
)
```

**Success Criteria:**
- ✅ Session maintained
- ✅ Investigation progresses

---

### 7.3 CLI Layer Tests

#### Test 7.3.1: Basic ThinkDeep CLI

**Command:**

```bash
model-chorus thinkdeep \
  --step "Investigate slow API response" \
  --step-number 1 \
  --total-steps 1 \
  --findings "Database query is the bottleneck" \
  --hypothesis "Missing index on user_id column" \
  --confidence high \
  --provider claude
```

**Expected Output:**

```
ANALYSIS:
The findings confirm database query performance as the bottleneck...

RECOMMENDATION:
Add index on user_id column to improve query performance from O(n) to O(log n).

Session ID: thinkdeep-[uuid]
Step: 1/1
Confidence: high
```

**Success Criteria:**
- ✅ Analysis provided
- ✅ Recommendation given
- ✅ Session ID shown
- ✅ Exit code 0

---

#### Test 7.3.2: CLI Multi-Step Investigation

**Command:**

```bash
# Step 1
SESSION=$(model-chorus thinkdeep \
  --step "Check error logs" \
  --step-number 1 \
  --total-steps 3 \
  --findings "Errors spike at midnight" \
  --next-step-required \
  --confidence exploring \
  --format json | jq -r '.session_id')

# Step 2
model-chorus thinkdeep \
  --step "Check scheduled jobs" \
  --step-number 2 \
  --total-steps 3 \
  --findings "Backup job runs at midnight, locks database" \
  --hypothesis "Backup job blocks application" \
  --confidence medium \
  --next-step-required \
  --continue "$SESSION"

# Step 3
model-chorus thinkdeep \
  --step "Verify with job rescheduling" \
  --step-number 3 \
  --total-steps 3 \
  --findings "No errors after moving backup to 3am" \
  --hypothesis "Backup job was blocking application" \
  --confidence certain \
  --continue "$SESSION"
```

**Success Criteria:**
- ✅ All steps complete
- ✅ Session maintained
- ✅ Confidence progression visible
- ✅ Final conclusion provided

---

#### Test 7.3.3: CLI with File Tracking

**Command:**

```bash
model-chorus thinkdeep \
  --step "Review authentication code" \
  --step-number 1 \
  --total-steps 2 \
  --findings "Password hashing uses MD5" \
  --files-checked "auth/password.py,auth/user.py" \
  --relevant-files "auth/password.py" \
  --hypothesis "Weak password hashing algorithm" \
  --confidence high \
  --next-step-required
```

**Success Criteria:**
- ✅ Files tracked in output
- ✅ Metadata shows checked vs relevant

---

#### Test 7.3.4: CLI with Issue Tracking

**Command:**

```bash
model-chorus thinkdeep \
  --step "Security audit results" \
  --step-number 1 \
  --total-steps 1 \
  --findings "Multiple critical issues found" \
  --issues-found '[{"description":"SQL injection","severity":"high"},{"description":"XSS","severity":"high"}]' \
  --hypothesis "Multiple security vulnerabilities" \
  --confidence high
```

**Success Criteria:**
- ✅ Issues displayed
- ✅ Severity shown
- ✅ Exit code 0

---

#### Test 7.3.5: CLI Error - Missing Required Param

**Command:**

```bash
model-chorus thinkdeep \
  --step "Test" \
  --step-number 1
  # Missing: total-steps, findings
```

**Expected Output:**

```
Error: Missing required parameters: total_steps, findings
Usage: model-chorus thinkdeep --step "..." --step-number N --total-steps N --findings "..." [OPTIONS]
```

**Success Criteria:**
- ✅ Clear error message
- ✅ Lists missing parameters
- ✅ Exit code 1

---

## Cross-Cutting Concerns

### Provider Compatibility Matrix

Test each workflow with different providers to ensure compatibility.

| Workflow | Claude | Gemini | Codex | Cursor |
|----------|--------|--------|-------|--------|
| Argument | ✅ Primary | ✅ Supported | ⚠️ Test | ⚠️ Test |
| Chat | ✅ Primary | ✅ Supported | ⚠️ Test | ⚠️ Test |
| Consensus | ✅ Included | ✅ Included | ⚠️ Optional | ⚠️ Optional |
| Ideate | ✅ Primary | ✅ Supported | ⚠️ Test | ⚠️ Test |
| Router | ✅ Primary | ⚠️ Test | ⚠️ Test | ⚠️ Test |
| Study | ✅ Supported | ✅ Primary | ⚠️ Test | ⚠️ Test |
| ThinkDeep | ✅ Primary | ✅ Supported | ⚠️ Test | ⚠️ Test |

**Legend:**
- ✅ Primary - Recommended and well-tested
- ✅ Supported - Tested and works well
- ✅ Included - Used in multi-provider scenarios
- ⚠️ Test - Requires testing (if API key available)
- ⚠️ Optional - Can be included if available

---

### Performance Benchmarks

Expected execution times (with real providers):

| Workflow | Simple Query | Complex Query | Multi-Turn |
|----------|--------------|---------------|------------|
| **Argument** | 30-60s | 60-120s | +20s/turn |
| **Chat** | 5-15s | 15-30s | +5s/turn |
| **Consensus** | 15-30s (parallel) | 30-60s | N/A |
| **Ideate** | 20-40s (5 ideas) | 40-90s (10 ideas) | +20s/turn |
| **Router** | 5-10s | 10-20s | N/A |
| **Study** | 30-60s (2 personas) | 60-120s (3+ personas) | +30s/turn |
| **ThinkDeep** | 10-25s/step | 25-45s/step | +10s/step |

**Notes:**
- Times are for Claude Sonnet 4.5 (gemini may vary)
- Parallel execution (Consensus) faster than sequential
- Multi-turn adds incremental time
- Network latency affects all timings

---

### Common Error Scenarios

#### 1. API Key Missing or Invalid

**Test:**

```bash
unset ANTHROPIC_API_KEY
model-chorus chat --prompt "Test" --provider claude
```

**Expected:**

```
Error: ANTHROPIC_API_KEY not found in environment
Please set your API key: export ANTHROPIC_API_KEY=sk-ant-...
```

---

#### 2. Provider Timeout

**Test:**

```python
result = await workflow.run(
    prompt="Very complex prompt...",
    timeout=1.0  # Too short
)
```

**Expected:**
- Timeout error or
- Partial results with timeout warning

---

#### 3. Rate Limiting

**Test:**

```python
# Rapid-fire requests
for i in range(100):
    try:
        result = await workflow.run(prompt=f"Query {i}")
    except RateLimitError as e:
        print(f"Rate limited at query {i}: {e}")
        break
```

**Expected:**
- Rate limit error after threshold
- Clear error message with retry guidance

---

#### 4. Invalid File Path

**Test:**

```bash
model-chorus chat \
  --prompt "Review this code" \
  --files /nonexistent/file.py
```

**Expected:**

```
Error: File not found: /nonexistent/file.py
```

---

#### 5. Invalid Parameter Values

**Test:**

```bash
# Invalid temperature
model-chorus chat --prompt "Test" --temperature 5.0

# Invalid strategy
model-chorus consensus --prompt "Test" --strategy invalid

# Invalid num_ideas
model-chorus ideate --prompt "Test" --num-ideas 0
```

**Expected:**
- Clear validation error for each
- Suggested valid ranges/values

---

### Memory and Threading Tests

#### Test: Session Persistence

```python
# Create session
result1 = await workflow.run(prompt="First query")
session_id = result1['session_id']

# Continue after delay
import time
time.sleep(60)  # 1 minute gap

result2 = await workflow.run(
    prompt="Follow-up query",
    continuation_id=session_id
)
```

**Success Criteria:**
- ✅ Session persists across time gap
- ✅ Context maintained

---

#### Test: Invalid Session ID

```python
try:
    result = await workflow.run(
        prompt="Test",
        continuation_id="invalid-session-id"
    )
except ValueError as e:
    print(f"✅ Expected error: {e}")
```

**Expected:**
- Error for invalid session ID

---

#### Test: Cross-Workflow Session Contamination

```python
# Create chat session
chat_result = await chat_workflow.run(prompt="Chat query")
chat_session = chat_result['session_id']

# Try to use in argument workflow
try:
    arg_result = await argument_workflow.run(
        prompt="Argument query",
        continuation_id=chat_session  # Wrong workflow type
    )
except ValueError as e:
    print(f"✅ Expected error: {e}")
```

**Expected:**
- Error or warning about session type mismatch

---

## Test Execution Guide

### 1. Pre-Test Checklist

- [ ] API keys configured in `.env`
- [ ] ModelChorus installed (`pip install -e .`)
- [ ] Test data directory created
- [ ] Provider connectivity verified
- [ ] Test results template prepared

---

### 2. Test Execution Order

**Phase 1: Smoke Tests (30 minutes)**
- One happy path test per workflow
- Verify basic functionality
- Confirm all providers accessible

**Phase 2: Standard Coverage (2-3 hours)**
- All happy path tests
- Common edge cases
- Error handling scenarios
- CLI tests

**Phase 3: Extended Tests (optional, 1-2 hours)**
- Cross-cutting concerns
- Performance benchmarking
- Memory/threading tests
- Provider compatibility matrix

---

### 3. Test Execution Commands

**Run all workflow tests:**

```bash
# Python tests
pytest tests/ -v

# CLI smoke tests
bash test_data/cli_smoke_tests.sh
```

**Run specific workflow:**

```bash
# Argument
pytest tests/test_argument_workflow.py -v

# Chat
pytest tests/test_chat_workflow.py -v

# Consensus
pytest tests/test_consensus_workflow.py -v

# Ideate
pytest tests/test_ideate_workflow.py -v

# Study
pytest tests/workflows/study/ -v

# ThinkDeep
pytest tests/test_thinkdeep_workflow.py -v
```

---

### 4. Results Documentation

Create `test_results/YYYY-MM-DD_test_results.md`:

```markdown
# Test Results: [Date]

## Environment
- ModelChorus Version: [version]
- Python Version: [version]
- Providers Tested: Claude, Gemini

## Summary
- Total Tests: X
- Passed: Y
- Failed: Z
- Warnings: W

## Workflow Results

### Argument Workflow
- Test 1.1.1: ✅ Pass (35s)
- Test 1.1.2: ✅ Pass (42s)
- Test 1.1.3: ⚠️ Partial (File context weaker than expected)
...

## Issues Found
1. [Issue description]
   - Severity: [High/Medium/Low]
   - Workflow: [Name]
   - Reproduction steps: [...]

## Performance Notes
- [Observations on timing]
- [Provider comparison]
```

---

### 5. Continuous Testing Strategy

**Daily Smoke Tests:**
```bash
# Quick validation (5 minutes)
./scripts/daily_smoke_test.sh
```

**Weekly Full Tests:**
```bash
# Comprehensive testing (3 hours)
./scripts/weekly_full_test.sh
```

**Pre-Release Tests:**
```bash
# Complete coverage (4-5 hours)
./scripts/release_test.sh
```

---

## Troubleshooting

### Issue: "Provider timeout"

**Symptoms:**
- Requests timeout after 120s
- No response received

**Solutions:**
1. Check network connectivity
2. Verify API key is valid
3. Increase timeout: `--timeout 300`
4. Check provider status page
5. Reduce prompt complexity

---

### Issue: "Rate limit exceeded"

**Symptoms:**
- Error after multiple requests
- "429 Too Many Requests"

**Solutions:**
1. Add delay between requests
2. Use different API key (if available)
3. Wait for rate limit reset
4. Check provider tier limits

---

### Issue: "Session not found"

**Symptoms:**
- Continuation fails
- "Invalid session ID"

**Solutions:**
1. Verify session ID format (check prefix: `thread-`, `argument-thread-`, etc.)
2. Ensure session hasn't expired
3. Check workflow type matches (can't use chat session in argument)
4. Verify conversation memory is properly configured

---

### Issue: "File not found"

**Symptoms:**
- CLI can't find specified file
- "No such file or directory"

**Solutions:**
1. Use absolute paths
2. Verify file exists: `ls -la path/to/file`
3. Check file permissions
4. Ensure file is readable

---

### Issue: "Poor quality responses"

**Symptoms:**
- Responses are irrelevant
- Output doesn't match expected format

**Solutions:**
1. Review prompt clarity
2. Add more context via `system_prompt`
3. Include relevant files
4. Try different provider
5. Adjust temperature
6. Check if prompt is too vague

---

### Issue: "Memory/threading not working"

**Symptoms:**
- Context not maintained across turns
- Each request seems independent

**Solutions:**
1. Verify continuation_id format
2. Check ConversationMemory is configured
3. Ensure same workflow instance is used
4. Verify session ID is being passed correctly

---

### Issue: "JSON parsing error in CLI"

**Symptoms:**
- `jq` command fails
- "parse error: Invalid numeric literal"

**Solutions:**
1. Use `--format json` flag
2. Verify output is valid JSON: `model-chorus ... | python -m json.tool`
3. Check for extra output (warnings, debug messages)

---

## Appendix

### A. Test Data Templates

**test_data/files/sample.py:**
```python
def hello():
    return 'world'
```

**test_data/files/sample.md:**
```markdown
# Test Document
This is a sample markdown file for testing.
```

---

### B. Environment Setup Script

**scripts/setup_test_env.sh:**

```bash
#!/bin/bash
set -e

echo "Setting up ModelChorus test environment..."

# Create directories
mkdir -p test_data/{files,outputs}
mkdir -p test_results

# Create sample files
echo "def hello(): return 'world'" > test_data/files/sample.py
echo "# Test Document" > test_data/files/sample.md

# Verify API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set"
else
    echo "✅ Claude API key found"
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Warning: GOOGLE_API_KEY not set"
else
    echo "✅ Gemini API key found"
fi

echo "✅ Test environment setup complete"
```

---

### C. Quick Reference Card

**Common Commands:**

```bash
# Chat
model-chorus chat --prompt "..." [--continue SESSION_ID]

# Consensus
model-chorus consensus --prompt "..." --provider claude gemini --strategy synthesize

# Argument
model-chorus argument --prompt "..." [--temperature 0.7]

# Ideate
model-chorus ideate --prompt "..." --num-ideas 5 [--temperature 0.7]

# Study
model-chorus study --scenario "..." [--persona Expert1 Expert2]

# ThinkDeep
model-chorus thinkdeep \
  --step "..." \
  --step-number N \
  --total-steps N \
  --findings "..." \
  [--hypothesis "..."] \
  [--confidence LEVEL] \
  [--next-step-required]

# Router
model-chorus router --request "..."
```

**Extract Session ID (JSON):**

```bash
SESSION=$(model-chorus chat --prompt "..." --format json | jq -r '.session_id')
```

**Common Flags:**

- `--provider PROVIDER` - Choose provider (claude, gemini, codex, cursor-agent)
- `--temperature FLOAT` - Creativity level (0.0-1.0)
- `--files PATH [PATH...]` - Include file context
- `--format json` - JSON output
- `--verbose` - Detailed output
- `--help` - Show help

---

### D. Success Metrics

**Quality Metrics:**
- Response relevance: High (addresses prompt directly)
- Format compliance: 100% (matches expected structure)
- Context retention: High (multi-turn coherence)

**Performance Metrics:**
- Chat: < 30s for standard queries
- Consensus: < 60s for 2 providers (parallel)
- Argument: < 90s for complete analysis
- Ideate: < 45s for 5 ideas
- Study: < 60s for 2 personas
- ThinkDeep: < 30s per step

**Reliability Metrics:**
- Success rate: > 95%
- Timeout rate: < 2%
- Error handling: 100% (graceful)

---

**End of Testing Playbook**

---

## Changelog

**v1.0 (2025-11-11):**
- Initial playbook creation
- Coverage: All 7 workflows (Argument, Chat, Consensus, Ideate, Router, Study, ThinkDeep)
- Test layers: Workflow, Agent, CLI
- Real provider integration
- Standard coverage level (happy path + edge cases + error handling)
