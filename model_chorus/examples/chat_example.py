"""
Multi-turn conversation example using ChatWorkflow.

This example demonstrates how to:
1. Create a new conversation
2. Continue an existing conversation
3. Include file context in conversations
4. Track conversation history
"""

import asyncio
from pathlib import Path

from model_chorus.providers import ClaudeProvider
from model_chorus.workflows import ChatWorkflow
from model_chorus.core.conversation import ConversationMemory


async def basic_chat_example():
    """Basic chat conversation without continuation."""
    print("=" * 60)
    print("Example 1: Basic Chat")
    print("=" * 60)

    # Create provider and conversation memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create chat workflow
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Send a message
    result = await workflow.run(
        prompt="What is quantum computing in one sentence?",
        temperature=0.7,
    )

    if result.success:
        print(f"Thread ID: {result.metadata['thread_id']}")
        print(f"Response: {result.synthesis}\n")
    else:
        print(f"Error: {result.error}\n")


async def multi_turn_conversation_example():
    """Multi-turn conversation with continuation."""
    print("=" * 60)
    print("Example 2: Multi-Turn Conversation")
    print("=" * 60)

    # Create provider and conversation memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create chat workflow
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # First turn: Start conversation
    print("\n--- Turn 1 ---")
    result1 = await workflow.run(
        prompt="What is the difference between classical and quantum computing?",
        temperature=0.7,
    )

    if not result1.success:
        print(f"Error: {result1.error}")
        return

    thread_id = result1.metadata["thread_id"]
    print(f"Thread ID: {thread_id}")
    print(f"Response: {result1.synthesis[:200]}...\n")

    # Second turn: Continue conversation
    print("\n--- Turn 2 ---")
    result2 = await workflow.run(
        prompt="Can you give me a practical example?",
        continuation_id=thread_id,
        temperature=0.7,
    )

    if result2.success:
        print(f"Thread ID: {result2.metadata['thread_id']}")
        print(f"Conversation length: {result2.metadata['conversation_length']} messages")
        print(f"Response: {result2.synthesis[:200]}...\n")

    # Third turn: Continue with more specific question
    print("\n--- Turn 3 ---")
    result3 = await workflow.run(
        prompt="How does quantum entanglement factor into this?",
        continuation_id=thread_id,
        temperature=0.7,
    )

    if result3.success:
        print(f"Thread ID: {result3.metadata['thread_id']}")
        print(f"Conversation length: {result3.metadata['conversation_length']} messages")
        print(f"Response: {result3.synthesis[:200]}...\n")


async def chat_with_file_context_example():
    """Chat with file context included."""
    print("=" * 60)
    print("Example 3: Chat with File Context")
    print("=" * 60)

    # Create a sample file for demonstration
    sample_file = Path("/tmp/sample_code.py")
    sample_file.write_text(
        """
def fibonacci(n):
    '''Calculate Fibonacci number recursively.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(fibonacci(10))
"""
    )

    # Create provider and conversation memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create chat workflow
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Send message with file context
    result = await workflow.run(
        prompt="Review this code and suggest improvements for performance.",
        files=[str(sample_file)],
        temperature=0.7,
    )

    if result.success:
        print(f"Thread ID: {result.metadata['thread_id']}")
        print(f"Response: {result.synthesis}\n")
    else:
        print(f"Error: {result.error}\n")

    # Clean up
    sample_file.unlink()


async def conversation_tracking_example():
    """Demonstrate conversation history tracking."""
    print("=" * 60)
    print("Example 4: Conversation Tracking")
    print("=" * 60)

    # Create provider and conversation memory
    provider = ClaudeProvider()
    memory = ConversationMemory()

    # Create chat workflow
    workflow = ChatWorkflow(provider=provider, conversation_memory=memory)

    # Start conversation
    print("\n--- Starting conversation ---")
    result1 = await workflow.run(
        prompt="Explain machine learning.",
    )

    if not result1.success:
        print(f"Error: {result1.error}")
        return

    thread_id = result1.metadata["thread_id"]

    # Check conversation history
    thread = workflow.get_thread(thread_id)
    if thread:
        print(f"\nConversation history:")
        print(f"  Thread ID: {thread.thread_id}")
        print(f"  Messages: {len(thread.messages)}")
        print(f"  Created: {thread.created_at}")

        # Show messages
        for i, msg in enumerate(thread.messages, 1):
            print(f"\n  Message {i} ({msg.role}):")
            print(f"    Content preview: {msg.content[:100]}...")
            if msg.files:
                print(f"    Files: {', '.join(msg.files)}")


async def main():
    """Run all examples."""
    # Example 1: Basic chat
    await basic_chat_example()

    # Example 2: Multi-turn conversation
    await multi_turn_conversation_example()

    # Example 3: Chat with file context
    await chat_with_file_context_example()

    # Example 4: Conversation tracking
    await conversation_tracking_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
