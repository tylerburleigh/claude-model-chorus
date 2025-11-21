"""
Multi-step investigation example using ThinkDeepWorkflow.

This example demonstrates how to:
1. Start a systematic investigation with ThinkDeep
2. Continue multi-turn investigations with hypothesis tracking
3. Track confidence progression across investigation steps
4. Accumulate findings and evidence
5. Use expert validation for additional insights
6. Inspect investigation state and progress
"""

import asyncio
from pathlib import Path

from model_chorus.core.conversation import ConversationMemory
from model_chorus.providers import ClaudeProvider, GeminiProvider
from model_chorus.workflows import (
    ConfidenceLevel,
    ThinkDeepWorkflow,
)


async def basic_investigation_example():
    """Basic single-step investigation."""
    print("=" * 60)
    print("Example 1: Basic Investigation")
    print("=" * 60)

    # Create provider and conversation memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create ThinkDeep workflow
    workflow = ThinkDeepWorkflow(provider=provider, conversation_memory=memory)

    # Start investigation
    result = await workflow.run(
        step="Why might a Python web application be experiencing intermittent 500 errors?",
        step_number=1,
        total_steps=1,
        next_step_required=False,
        findings="Investigating intermittent 500 errors in Python web application",
        temperature=0.7,
    )

    if result.success:
        thread_id = result.metadata["thread_id"]
        confidence = result.metadata["confidence"]
        hypotheses_count = result.metadata["hypotheses_count"]

        print(f"Thread ID: {thread_id}")
        print(f"Investigation Step: {result.metadata['step_number']}")
        print(f"Confidence Level: {confidence}")
        print(f"Hypotheses Tracked: {hypotheses_count}")
        print(f"\nFindings:\n{result.synthesis[:300]}...\n")
    else:
        print(f"Error: {result.error}\n")


async def multi_step_investigation_example():
    """
    Multi-step investigation showing hypothesis evolution and confidence progression.

    This example demonstrates a systematic investigation of a performance issue,
    showing how hypotheses are formed, tested, and confidence evolves across steps.
    """
    print("=" * 60)
    print("Example 2: Multi-Step Investigation with Hypothesis Evolution")
    print("=" * 60)

    # Create sample files for investigation
    sample_files = {
        "/tmp/api_handler.py": """
async def get_user_data(user_id):
    '''Fetch user data from database.'''
    # Synchronous database call in async function
    user = db.query(User).filter(User.id == user_id).first()

    # Fetch related data
    posts = db.query(Post).filter(Post.user_id == user_id).all()
    comments = db.query(Comment).filter(Comment.user_id == user_id).all()

    return {
        'user': user,
        'posts': posts,
        'comments': comments
    }
""",
        "/tmp/database.py": """
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Synchronous database engine
engine = create_engine('postgresql://localhost/mydb')
Session = sessionmaker(bind=engine)
db = Session()
""",
        "/tmp/config.py": """
# Application configuration
DATABASE_POOL_SIZE = 5
REQUEST_TIMEOUT = 30
ENABLE_QUERY_LOGGING = False
""",
    }

    # Write sample files
    for filepath, content in sample_files.items():
        Path(filepath).write_text(content)

    # Create provider and conversation memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create ThinkDeep workflow
    workflow = ThinkDeepWorkflow(provider=provider, conversation_memory=memory)

    # Step 1: Initial investigation - examine API handler
    print("\n--- Investigation Step 1: Analyze API Handler ---")
    result1 = await workflow.run(
        step="Investigate this API handler for potential performance issues. What could cause slowdowns?",
        step_number=1,
        total_steps=3,
        next_step_required=True,
        findings="Analyzing API handler code for performance bottlenecks",
        files=["/tmp/api_handler.py"],
        temperature=0.7,
    )

    if not result1.success:
        print(f"Error: {result1.error}")
        return

    thread_id = result1.metadata["thread_id"]
    print(f"Thread ID: {thread_id}")
    print(f"Confidence: {result1.metadata['confidence']}")
    print(f"Hypotheses: {result1.metadata['hypotheses_count']}")
    print(f"Files Examined: {result1.metadata['files_examined']}")
    print(f"\nFindings:\n{result1.synthesis[:400]}...\n")

    # Step 2: Continue investigation - examine database configuration
    print("\n--- Investigation Step 2: Check Database Configuration ---")
    result2 = await workflow.run(
        step="Now examine the database configuration. Does this support or contradict our hypothesis about async/sync mixing?",
        step_number=2,
        total_steps=3,
        next_step_required=True,
        findings="Checking database configuration for async/sync issues",
        continuation_id=thread_id,
        files=["/tmp/database.py"],
        temperature=0.7,
    )

    if result2.success:
        print(f"Investigation Step: {result2.metadata['step_number']}")
        print(f"Confidence: {result2.metadata['confidence']}")
        print(f"Hypotheses: {result2.metadata['hypotheses_count']}")
        print(f"Total Files Examined: {result2.metadata['files_examined']}")
        print(f"\nFindings:\n{result2.synthesis[:400]}...\n")

    # Step 3: Final investigation - check configuration
    print("\n--- Investigation Step 3: Review Configuration Settings ---")
    result3 = await workflow.run(
        step="Review the application configuration. Are there any settings that could exacerbate the performance issues we've identified?",
        step_number=3,
        total_steps=3,
        next_step_required=False,
        findings="Reviewing configuration settings for performance impact",
        continuation_id=thread_id,
        files=["/tmp/config.py"],
        temperature=0.7,
    )

    if result3.success:
        print(f"Investigation Step: {result3.metadata['step_number']}")
        print(f"Confidence: {result3.metadata['confidence']}")
        print(f"Hypotheses: {result3.metadata['hypotheses_count']}")
        print(f"Total Files Examined: {result3.metadata['files_examined']}")
        print(f"\nFindings:\n{result3.synthesis[:400]}...\n")

    # Show final investigation state
    print("\n--- Final Investigation State ---")
    state = workflow.get_investigation_state(thread_id)
    if state:
        print(f"Total Steps Completed: {len(state.steps)}")
        print(f"Final Confidence: {state.current_confidence}")
        print(f"Total Hypotheses: {len(state.hypotheses)}")
        print(f"Total Files Examined: {len(state.relevant_files)}")

        if state.hypotheses:
            print("\nHypotheses:")
            for i, hyp in enumerate(state.hypotheses, 1):
                print(f"  {i}. [{hyp.status.upper()}] {hyp.hypothesis}")
                if hyp.evidence:
                    print(f"     Evidence: {len(hyp.evidence)} items")

    # Clean up sample files
    for filepath in sample_files.keys():
        Path(filepath).unlink(missing_ok=True)


async def investigation_with_expert_validation():
    """
    Investigation with expert validation from a different model.

    Demonstrates how ThinkDeep can use a second model for validation
    and additional insights when confidence hasn't reached "certain" level.
    """
    print("=" * 60)
    print("Example 3: Investigation with Expert Validation")
    print("=" * 60)

    # Create sample authentication code
    auth_code = Path("/tmp/auth_service.py")
    auth_code.write_text(
        """
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "hardcoded-secret-key-123"

def create_token(user_id):
    '''Create JWT token for user.'''
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    '''Verify JWT token.'''
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['user_id']
    except:
        return None
"""
    )

    # Create primary provider and expert provider
    primary_provider = ClaudeProvider()
    expert_provider = GeminiProvider()
    memory = ConversationMemory()

    # Create ThinkDeep workflow with expert validation
    workflow = ThinkDeepWorkflow(
        provider=primary_provider,
        expert_provider=expert_provider,
        conversation_memory=memory,
        config={"enable_expert_validation": True},
    )

    # Investigate security issues
    print("\n--- Investigation: Security Analysis ---")
    result = await workflow.run(
        step="Analyze this authentication code for security vulnerabilities. What are the risks?",
        step_number=1,
        total_steps=1,
        next_step_required=False,
        findings="Analyzing authentication code for security vulnerabilities",
        files=[str(auth_code)],
        temperature=0.7,
    )

    if result.success:
        thread_id = result.metadata["thread_id"]
        expert_performed = result.metadata["expert_validation_performed"]

        print(f"Thread ID: {thread_id}")
        print(f"Confidence: {result.metadata['confidence']}")
        print(
            f"Expert Validation: {'✓ Performed' if expert_performed else '✗ Not performed'}"
        )

        # Primary findings
        print(f"\n--- Primary Investigation ({primary_provider.provider_name}) ---")
        print(result.synthesis[:500])

        # Expert validation (if performed)
        if expert_performed and len(result.steps) > 1:
            print(f"\n--- Expert Validation ({expert_provider.provider_name}) ---")
            print(result.steps[-1].content[:500])
    else:
        print(f"Error: {result.error}")

    # Clean up
    auth_code.unlink()


async def hypothesis_management_example():
    """
    Demonstrate manual hypothesis management and state inspection.

    Shows how to programmatically add, update, and track hypotheses
    during an investigation.
    """
    print("=" * 60)
    print("Example 4: Hypothesis Management and State Inspection")
    print("=" * 60)

    # Create provider and workflow
    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ThinkDeepWorkflow(provider=provider, conversation_memory=memory)

    # Start investigation
    print("\n--- Starting Investigation ---")
    result = await workflow.run(
        step="Investigate why a machine learning model's accuracy dropped from 95% to 75%.",
        step_number=1,
        total_steps=1,
        next_step_required=False,
        findings="Investigating ML model accuracy drop",
        temperature=0.7,
    )

    if not result.success:
        print(f"Error: {result.error}")
        return

    thread_id = result.metadata["thread_id"]
    print(f"Thread ID: {thread_id}")
    print(f"Initial Confidence: {result.metadata['confidence']}\n")

    # Manually add a hypothesis
    print("--- Adding Manual Hypothesis ---")
    workflow.add_hypothesis(
        thread_id,
        hypothesis_text="Data distribution has changed (data drift)",
        evidence=["User report mentions recent changes in input data"],
    )

    # Add another hypothesis
    workflow.add_hypothesis(
        thread_id, hypothesis_text="Training data was contaminated", evidence=[]
    )

    # Get investigation summary
    summary = workflow.get_investigation_summary(thread_id)
    if summary:
        print(f"Total Hypotheses: {summary['total_hypotheses']}")
        print(f"Active Hypotheses: {summary['active_hypotheses']}")
        print(f"Validated Hypotheses: {summary['validated_hypotheses']}")
        print(f"Investigation Complete: {summary['is_complete']}")

    # Show all active hypotheses
    print("\n--- Active Hypotheses ---")
    active_hypotheses = workflow.get_active_hypotheses(thread_id)
    for i, hyp in enumerate(active_hypotheses, 1):
        print(f"{i}. {hyp.hypothesis}")
        print(f"   Evidence items: {len(hyp.evidence)}")

    # Update hypothesis status
    print("\n--- Validating Hypothesis ---")
    if active_hypotheses:
        first_hypothesis = active_hypotheses[0].hypothesis
        workflow.update_hypothesis(
            thread_id,
            first_hypothesis,
            new_evidence=[
                "Analysis confirmed: Input data statistics differ significantly"
            ],
            new_status="validated",
        )
        print(f"Validated: {first_hypothesis}")

    # Update confidence level
    workflow.update_confidence(thread_id, ConfidenceLevel.HIGH.value)
    print(f"\nUpdated Confidence: {workflow.get_confidence(thread_id)}")

    # Final state
    print("\n--- Final Investigation State ---")
    final_summary = workflow.get_investigation_summary(thread_id)
    if final_summary:
        print(f"Confidence: {final_summary['confidence']}")
        print(f"Total Steps: {final_summary['total_steps']}")
        print(f"Validated Hypotheses: {final_summary['validated_hypotheses']}")
        print(f"Active Hypotheses: {final_summary['active_hypotheses']}")


async def confidence_progression_example():
    """
    Demonstrate confidence progression through investigation steps.

    Shows how confidence should naturally increase as evidence accumulates
    and hypotheses are validated.
    """
    print("=" * 60)
    print("Example 5: Confidence Progression")
    print("=" * 60)

    provider = ClaudeProvider()
    memory = ConversationMemory()
    workflow = ThinkDeepWorkflow(provider=provider, conversation_memory=memory)

    # Track confidence across steps
    confidence_levels = []

    # Step 1: Initial exploration (confidence: exploring/low)
    print("\n--- Step 1: Initial Exploration ---")
    result1 = await workflow.run(
        step="A Python script crashes with 'RecursionError: maximum recursion depth exceeded'. What could be the cause?",
        step_number=1,
        total_steps=3,
        next_step_required=True,
        findings="Exploring possible causes of RecursionError",
        temperature=0.7,
    )

    if result1.success:
        thread_id = result1.metadata["thread_id"]
        confidence_levels.append(result1.metadata["confidence"])
        print(f"Confidence: {result1.metadata['confidence']}")

        # Step 2: Gather evidence (confidence: medium)
        print("\n--- Step 2: Hypothesis Testing ---")
        result2 = await workflow.run(
            step="Let's test the hypothesis: Is it an infinite recursion problem? What evidence would support this?",
            step_number=2,
            total_steps=3,
            next_step_required=True,
            findings="Testing infinite recursion hypothesis and gathering evidence",
            continuation_id=thread_id,
            temperature=0.7,
        )

        if result2.success:
            confidence_levels.append(result2.metadata["confidence"])
            print(f"Confidence: {result2.metadata['confidence']}")

            # Step 3: Validate hypothesis (confidence: high/very_high)
            print("\n--- Step 3: Validation ---")
            result3 = await workflow.run(
                step="Given the evidence we've found, can we conclude with high confidence what the root cause is?",
                step_number=3,
                total_steps=3,
                next_step_required=False,
                findings="Validating hypothesis with high confidence conclusion",
                continuation_id=thread_id,
                temperature=0.7,
            )

            if result3.success:
                confidence_levels.append(result3.metadata["confidence"])
                print(f"Confidence: {result3.metadata['confidence']}")

    # Show confidence progression
    print("\n--- Confidence Progression ---")
    confidence_order = [level.value for level in ConfidenceLevel]
    for i, conf in enumerate(confidence_levels, 1):
        conf_index = confidence_order.index(conf) if conf in confidence_order else 0
        progress_bar = "█" * (conf_index + 1) + "░" * (
            len(confidence_order) - conf_index - 1
        )
        print(f"Step {i}: {conf:15} [{progress_bar}]")


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "THINKDEEP WORKFLOW EXAMPLES" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Example 1: Basic investigation
    await basic_investigation_example()

    # Example 2: Multi-step investigation
    await multi_step_investigation_example()

    # Example 3: Expert validation
    await investigation_with_expert_validation()

    # Example 4: Hypothesis management
    await hypothesis_management_example()

    # Example 5: Confidence progression
    await confidence_progression_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("• ThinkDeep provides systematic investigation with hypothesis tracking")
    print(
        "• Confidence progresses from 'exploring' to 'certain' as evidence accumulates"
    )
    print("• Multi-turn investigations maintain state across conversation turns")
    print("• Expert validation provides additional validation from different models")
    print("• Investigation state can be inspected and managed programmatically")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
