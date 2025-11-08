"""
ModelChorus Workflow Examples

This module provides comprehensive examples for using ModelChorus workflows:
- ARGUMENT: Structured dialectical reasoning and argument analysis
- IDEATE: Creative brainstorming with configurable parameters

Each workflow demonstrates:
- Basic usage patterns
- Advanced features (continuation, file context, parameters)
- Error handling
- Best practices

Requirements:
- ModelChorus installed: pip install modelchorus
- Provider CLI tools configured (claude, gemini, codex, etc.)
- API keys set in environment variables
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "modelchorus" / "src"))

from modelchorus.workflows import ArgumentWorkflow, IdeateWorkflow
from modelchorus.providers import ClaudeProvider, GeminiProvider
from modelchorus.core.conversation import ConversationMemory


# ============================================================================
# ARGUMENT WORKFLOW EXAMPLES
# ============================================================================

async def example_argument_basic():
    """
    Example 1: Basic argument analysis.

    Demonstrates the simplest use case - analyzing a single argument/claim
    through dialectical reasoning (Creator → Skeptic → Moderator).
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Argument Analysis")
    print("="*80 + "\n")

    # Initialize provider and memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create workflow
    workflow = ArgumentWorkflow(
        provider=provider,
        conversation_memory=memory
    )

    # Run analysis
    prompt = "Universal basic income would significantly reduce poverty and inequality."

    print(f"Analyzing argument: {prompt}\n")

    result = await workflow.run(
        prompt=prompt,
        temperature=0.7
    )

    if result.success:
        print("✓ Analysis Complete\n")

        # Show each role's perspective
        for step in result.steps:
            role_name = step.metadata.get('name', 'Step')
            print(f"--- {role_name} ---")
            print(step.content)
            print()

        # Show final synthesis
        if result.synthesis:
            print("--- Final Synthesis ---")
            print(result.synthesis)
            print()

        # Show thread ID for continuation
        thread_id = result.metadata.get('thread_id')
        print(f"Thread ID for continuation: {thread_id}")
    else:
        print(f"✗ Analysis failed: {result.error}")


async def example_argument_with_files():
    """
    Example 2: Argument analysis with file context.

    Shows how to provide supporting documents/context files to enrich
    the argument analysis with specific data or background information.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Argument Analysis with File Context")
    print("="*80 + "\n")

    # Create sample context file
    context_file = Path("examples/ubi_data.txt")
    context_file.parent.mkdir(exist_ok=True)
    context_file.write_text("""
    UBI Pilot Study Results:
    - Finland (2017-2018): Recipients reported higher well-being, no significant employment impact
    - Stockton, CA (2019-2021): 25% increase in full-time employment among recipients
    - Kenya (2016-present): Reduced hunger by 15%, increased school enrollment by 20%

    Economic Data:
    - US poverty rate: 11.4% (2020)
    - Cost estimate for $1000/month UBI: ~$3.9 trillion annually
    - Current federal welfare spending: ~$1.1 trillion annually
    """)

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    prompt = "UBI pilot studies show promising results, suggesting widespread implementation would work."

    print(f"Analyzing: {prompt}")
    print(f"Context file: {context_file}\n")

    result = await workflow.run(
        prompt=prompt,
        files=[str(context_file)],
        temperature=0.7
    )

    if result.success:
        print("✓ Analysis Complete (with context)\n")
        print(result.synthesis)

    # Cleanup
    context_file.unlink(missing_ok=True)


async def example_argument_continuation():
    """
    Example 3: Continuing an argument analysis.

    Demonstrates how to use conversation threading to continue
    a previous analysis with follow-up questions or new angles.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Argument Continuation (Threading)")
    print("="*80 + "\n")

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    # Initial analysis
    print("Initial analysis:")
    result1 = await workflow.run(
        prompt="Remote work increases productivity for knowledge workers."
    )

    if not result1.success:
        print(f"✗ Failed: {result1.error}")
        return

    thread_id = result1.metadata['thread_id']
    print(f"✓ Complete. Thread ID: {thread_id}\n")

    # Continue the conversation
    print("Continuing analysis with follow-up:")
    result2 = await workflow.run(
        prompt="But what about collaboration and creativity?",
        continuation_id=thread_id
    )

    if result2.success:
        print("✓ Continuation complete\n")
        print(result2.synthesis)
        print(f"\nConversation length: {result2.metadata.get('conversation_length', 0)} messages")


async def example_argument_custom_config():
    """
    Example 4: Argument analysis with custom configuration.

    Shows how to customize the argument workflow with different
    parameters like temperature, max_tokens, and system prompts.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Configuration")
    print("="*80 + "\n")

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

    prompt = "Cryptocurrency should replace traditional banking systems."

    print(f"Analyzing with custom config: {prompt}\n")

    result = await workflow.run(
        prompt=prompt,
        system_prompt="Focus on economic impacts and regulatory challenges.",
        temperature=0.5,  # Lower temperature for more focused analysis
        max_tokens=2000
    )

    if result.success:
        print("✓ Analysis complete\n")
        print(result.synthesis)


# ============================================================================
# IDEATE WORKFLOW EXAMPLES
# ============================================================================

async def example_ideate_basic():
    """
    Example 5: Basic ideation/brainstorming.

    Demonstrates simple creative idea generation with default parameters.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Basic Ideation")
    print("="*80 + "\n")

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    prompt = "New features for a personal task management application"

    print(f"Generating ideas for: {prompt}\n")

    result = await workflow.run(
        prompt=prompt,
        num_ideas=5,
        temperature=0.9  # High creativity
    )

    if result.success:
        print("✓ Ideation Complete\n")

        # Show generated ideas
        for i, step in enumerate(result.steps, 1):
            idea_name = step.metadata.get('name', f'Idea {i}')
            print(f"--- {idea_name} ---")
            print(step.content)
            print()

        # Show summary
        if result.synthesis:
            print("--- Summary & Recommendations ---")
            print(result.synthesis)


async def example_ideate_high_creativity():
    """
    Example 6: High-creativity brainstorming.

    Uses maximum temperature for highly creative, unconventional ideas.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: High-Creativity Ideation")
    print("="*80 + "\n")

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    prompt = "Unconventional marketing campaigns for a sustainable fashion brand"

    print(f"Generating creative ideas: {prompt}")
    print("Temperature: 1.0 (maximum creativity)\n")

    result = await workflow.run(
        prompt=prompt,
        num_ideas=7,
        temperature=1.0  # Maximum creativity
    )

    if result.success:
        print("✓ Generated highly creative ideas\n")
        print(result.synthesis)


async def example_ideate_with_constraints():
    """
    Example 7: Ideation with constraints/criteria.

    Demonstrates how to guide idea generation with specific constraints
    or requirements via system prompts.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Ideation with Constraints")
    print("="*80 + "\n")

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    prompt = "Revenue streams for a developer tools startup"

    constraints = """
    Constraints:
    - Must be implementable within 6 months
    - Budget under $50,000
    - Target audience: individual developers and small teams
    - Must align with open-source values
    """

    print(f"Prompt: {prompt}")
    print(f"Constraints: {constraints}\n")

    result = await workflow.run(
        prompt=prompt,
        system_prompt=constraints,
        num_ideas=6,
        temperature=0.85
    )

    if result.success:
        print("✓ Generated constrained ideas\n")
        print(result.synthesis)


async def example_ideate_refine():
    """
    Example 8: Refining specific ideas through continuation.

    Shows how to use threading to drill down into specific ideas
    and develop them further.
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Idea Refinement via Continuation")
    print("="*80 + "\n")

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    # Initial brainstorming
    print("Initial brainstorming:")
    result1 = await workflow.run(
        prompt="Gamification features for an online learning platform",
        num_ideas=4
    )

    if not result1.success:
        print(f"✗ Failed: {result1.error}")
        return

    thread_id = result1.metadata['thread_id']
    print(f"✓ Generated {len(result1.steps)} ideas\n")

    # Refine a specific idea
    print("Refining idea #2 (achievement system):")
    result2 = await workflow.run(
        prompt="Expand on idea #2. Provide detailed implementation steps and engagement metrics.",
        continuation_id=thread_id,
        num_ideas=1
    )

    if result2.success:
        print("✓ Refinement complete\n")
        print(result2.synthesis)


# ============================================================================
# CROSS-WORKFLOW PATTERNS
# ============================================================================

async def example_error_handling():
    """
    Example 13: Proper error handling across workflows.

    Demonstrates best practices for handling workflow failures.
    """
    print("\n" + "="*80)
    print("EXAMPLE 13: Error Handling Best Practices")
    print("="*80 + "\n")

    try:
        provider = ClaudeProvider()
        memory = ConversationMemory()
        workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

        result = await workflow.run(
            prompt="Test prompt with intentional issues",
            max_tokens=10  # Too small, may cause issues
        )

        if result.success:
            print("✓ Workflow succeeded")
            print(result.synthesis)
        else:
            # Handle workflow failure
            print(f"✗ Workflow failed: {result.error}")
            print("Fallback actions:")
            print("  - Log error for debugging")
            print("  - Retry with adjusted parameters")
            print("  - Notify user of issue")

    except Exception as e:
        # Handle unexpected exceptions
        print(f"✗ Unexpected error: {e}")
        print("Recovery actions:")
        print("  - Check API key configuration")
        print("  - Verify provider CLI is installed")
        print("  - Check network connectivity")


async def example_output_management():
    """
    Example 14: Managing workflow outputs.

    Shows how to save, load, and process workflow results.
    """
    print("\n" + "="*80)
    print("EXAMPLE 14: Output Management")
    print("="*80 + "\n")

    import json
    from datetime import datetime

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = IdeateWorkflow(provider=provider, conversation_memory=memory)

    result = await workflow.run(
        prompt="Ideas for improving code review processes",
        num_ideas=5
    )

    if result.success:
        # Save to file
        output_file = Path("examples/ideation_output.json")

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": "Ideas for improving code review processes",
            "success": result.success,
            "ideas": [
                {
                    "name": step.metadata.get('name', f'Idea {i}'),
                    "content": step.content,
                    "metadata": step.metadata
                }
                for i, step in enumerate(result.steps, 1)
            ],
            "synthesis": result.synthesis,
            "metadata": result.metadata
        }

        output_file.write_text(json.dumps(output_data, indent=2))
        print(f"✓ Output saved to: {output_file}")

        # Load and process
        loaded_data = json.loads(output_file.read_text())
        print(f"✓ Loaded {len(loaded_data['ideas'])} ideas from file")

        # Cleanup
        output_file.unlink(missing_ok=True)


async def example_provider_comparison():
    """
    Example 15: Comparing results across providers.

    Demonstrates running the same query with different providers
    to compare outputs.
    """
    print("\n" + "="*80)
    print("EXAMPLE 15: Multi-Provider Comparison")
    print("="*80 + "\n")

    prompt = "Is TypeScript worth adopting for a small team?"

    providers = [
        ("Claude", ClaudeProvider()),
        ("Gemini", GeminiProvider()),
    ]

    memory = ConversationMemory()

    for name, provider in providers:
        print(f"\n--- Provider: {name} ---\n")

        try:
            workflow = ArgumentWorkflow(provider=provider, conversation_memory=memory)

            result = await workflow.run(
                prompt=prompt,
                temperature=0.7
            )

            if result.success:
                print(f"✓ {name} analysis complete")
                print(f"Model: {result.metadata.get('model', 'unknown')}")
                print(f"Synthesis length: {len(result.synthesis)} chars")
                # Could compare outputs, quality, speed, etc.
            else:
                print(f"✗ {name} failed: {result.error}")

        except Exception as e:
            print(f"✗ {name} error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        ("Argument: Basic", example_argument_basic),
        ("Argument: With Files", example_argument_with_files),
        ("Argument: Continuation", example_argument_continuation),
        ("Argument: Custom Config", example_argument_custom_config),
        ("Ideate: Basic", example_ideate_basic),
        ("Ideate: High Creativity", example_ideate_high_creativity),
        ("Ideate: With Constraints", example_ideate_with_constraints),
        ("Ideate: Refinement", example_ideate_refine),
        ("Error Handling", example_error_handling),
        ("Output Management", example_output_management),
        ("Provider Comparison", example_provider_comparison),
    ]

    print("\n" + "="*80)
    print("MODELCHORUS WORKFLOW EXAMPLES")
    print("="*80)
    print(f"\nRunning {len(examples)} examples...\n")

    for name, example_func in examples:
        print(f"\nRunning: {name}")
        try:
            await example_func()
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Example failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)


async def run_specific_example(example_name: str):
    """Run a specific example by name."""
    examples_map = {
        "argument_basic": example_argument_basic,
        "argument_files": example_argument_with_files,
        "argument_continuation": example_argument_continuation,
        "argument_config": example_argument_custom_config,
        "ideate_basic": example_ideate_basic,
        "ideate_creative": example_ideate_high_creativity,
        "ideate_constraints": example_ideate_with_constraints,
        "ideate_refine": example_ideate_refine,
        "error_handling": example_error_handling,
        "output_management": example_output_management,
        "provider_comparison": example_provider_comparison,
    }

    example_func = examples_map.get(example_name)
    if example_func:
        print(f"\nRunning example: {example_name}\n")
        await example_func()
    else:
        print(f"✗ Unknown example: {example_name}")
        print(f"Available examples: {', '.join(examples_map.keys())}")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1]
        asyncio.run(run_specific_example(example_name))
    else:
        # Run all examples
        asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
