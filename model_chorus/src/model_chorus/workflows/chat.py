"""
Chat workflow for single-model peer consultation with conversation threading.

This module implements the ChatWorkflow which provides simple, straightforward
consultation with a single AI model while maintaining conversation continuity
through threading.
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

from ..core.base_workflow import BaseWorkflow, WorkflowResult, WorkflowStep
from ..core.conversation import ConversationMemory
from ..core.prompts import prepend_system_constraints
from ..providers import ModelProvider, GenerationRequest, GenerationResponse
from ..core.models import ConversationMessage
from ..core.progress import emit_workflow_start, emit_workflow_complete

logger = logging.getLogger(__name__)


class ChatWorkflow(BaseWorkflow):
    """
    Simple single-model chat workflow with conversation continuity.

    This workflow provides straightforward peer consultation with a single AI model,
    supporting conversation threading for multi-turn interactions. Unlike multi-model
    workflows like Consensus, Chat focuses on simplicity and conversational flow.

    Key features:
    - Single provider (not multi-model)
    - Conversation threading via continuation_id
    - Inherits conversation support from BaseWorkflow
    - Automatic conversation history management
    - Simple request/response pattern

    The ChatWorkflow is ideal for:
    - Quick second opinions from an AI model
    - Iterative conversations and refinement
    - Simple consultations without orchestration overhead
    - Building conversational applications

    Example:
        >>> from model_chorus.providers import ClaudeProvider
        >>> from model_chorus.workflows import ChatWorkflow
        >>> from model_chorus.core.conversation import ConversationMemory
        >>>
        >>> # Create provider and conversation memory
        >>> provider = ClaudeProvider()
        >>> memory = ConversationMemory()
        >>>
        >>> # Create workflow
        >>> workflow = ChatWorkflow(provider, conversation_memory=memory)
        >>>
        >>> # First message (creates new conversation)
        >>> result1 = await workflow.run("What is quantum computing?")
        >>> thread_id = result1.metadata.get('thread_id')
        >>> print(result1.synthesis)
        >>>
        >>> # Follow-up message (continues conversation)
        >>> result2 = await workflow.run(
        ...     "How does it differ from classical computing?",
        ...     continuation_id=thread_id
        ... )
        >>> print(result2.synthesis)
        >>>
        >>> # Check conversation history
        >>> thread = workflow.get_thread(thread_id)
        >>> print(f"Total messages: {len(thread.messages)}")
    """

    def __init__(
        self,
        provider: ModelProvider,
        fallback_providers: Optional[List[ModelProvider]] = None,
        config: Optional[Dict[str, Any]] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize ChatWorkflow with a single provider.

        Args:
            provider: ModelProvider instance to use for generation
            fallback_providers: Optional list of fallback providers to try if primary fails
            config: Optional configuration dictionary
            conversation_memory: Optional ConversationMemory for multi-turn conversations

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        super().__init__(
            name="Chat",
            description="Single-model peer consultation with conversation threading",
            config=config,
            conversation_memory=conversation_memory
        )
        self.provider = provider
        self.fallback_providers = fallback_providers or []

        logger.info(f"ChatWorkflow initialized with provider: {provider.provider_name}")

    async def run(
        self,
        prompt: str,
        continuation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        skip_provider_check: bool = False,
        **kwargs
    ) -> WorkflowResult:
        """
        Execute chat workflow with optional conversation continuation.

        This method handles both fresh conversations and continuations of existing
        threads. If a continuation_id is provided and conversation_memory is available,
        the conversation history will be loaded and included in the context.

        Args:
            prompt: The user's message/query
            continuation_id: Optional thread ID to continue an existing conversation
            files: Optional list of file paths to include in conversation context
            skip_provider_check: Skip provider availability check (faster startup)
            **kwargs: Additional parameters passed to provider.generate()
                     (e.g., temperature, max_tokens, system_prompt)

        Returns:
            WorkflowResult containing:
                - success: True if generation succeeded
                - synthesis: The model's response
                - steps: Single step with the model's response
                - metadata: thread_id, model info, and other details

        Raises:
            Exception: If provider.generate() fails

        Example:
            >>> # Fresh conversation
            >>> result = await workflow.run("Explain recursion")
            >>>
            >>> # Continuation
            >>> result2 = await workflow.run(
            ...     "Give me an example",
            ...     continuation_id=result.metadata['thread_id']
            ... )
            >>>
            >>> # With file context
            >>> result3 = await workflow.run(
            ...     "Review this code",
            ...     files=["/path/to/file.py"]
            ... )
        """
        # Handle empty prompt gracefully
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to chat workflow")
            return WorkflowResult(
                success=False,
                error="Prompt cannot be empty.",
                metadata={"error_type": "validation_error"}
            )

        logger.info(
            f"Starting chat workflow - prompt length: {len(prompt)}, "
            f"continuation: {continuation_id is not None}, "
            f"files: {len(files) if files else 0}"
        )

        # Check provider availability
        if not skip_provider_check:
            has_available, available, unavailable = await self.check_provider_availability(
                self.provider, self.fallback_providers
            )

            if not has_available:
                from ..providers.cli_provider import ProviderUnavailableError
                error_msg = "No providers available for chat:\n"
                for name, error in unavailable:
                    error_msg += f"  - {name}: {error}\n"
                raise ProviderUnavailableError(
                    "all",
                    error_msg,
                    [
                        "Check installations: model-chorus list-providers --check",
                        "Install missing providers or update .model-chorusrc"
                    ]
                )

            if unavailable and logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Some providers unavailable: {[n for n, _ in unavailable]}")
                logger.info(f"Will use available providers: {available}")

        # Generate or use thread ID
        # Track whether this is truly a continuation (valid existing thread)
        is_valid_continuation = False

        # Validate continuation_id if provided
        if continuation_id:
            # Check if the thread actually exists in conversation memory
            if self.conversation_memory:
                existing_thread = self.get_thread(continuation_id)
                if existing_thread:
                    thread_id = continuation_id
                    is_valid_continuation = True
                else:
                    # Thread doesn't exist, create a new one instead
                    logger.warning(
                        f"Continuation ID '{continuation_id}' not found in conversation memory. "
                        f"Creating new conversation instead."
                    )
                    thread_id = self.conversation_memory.create_thread(workflow_name=self.name)
                    is_valid_continuation = False
            else:
                # No conversation memory, just use the provided ID
                thread_id = continuation_id
                is_valid_continuation = True  # Assume valid if no memory to check
        else:
            # Create new thread if conversation memory available
            if self.conversation_memory:
                thread_id = self.conversation_memory.create_thread(
                    workflow_name=self.name
                )
            else:
                thread_id = str(uuid.uuid4())

        # Initialize result
        result = WorkflowResult(success=False)

        try:
            # Build the full prompt with conversation history and file context if available
            full_prompt = self._build_prompt_with_history(prompt, thread_id, files)

            # Prepare system prompt with read-only constraints
            custom_system_prompt = kwargs.get('system_prompt')
            final_system_prompt = prepend_system_constraints(custom_system_prompt)

            # Create generation request with read-only constraints prepended
            request_kwargs = {k: v for k, v in kwargs.items() if k != 'system_prompt'}
            request = GenerationRequest(
                prompt=full_prompt,
                system_prompt=final_system_prompt,
                continuation_id=thread_id,
                **request_kwargs
            )

            logger.info(f"Sending request to provider: {self.provider.provider_name}")

            # Emit workflow start
            emit_workflow_start("chat")

            # Generate response from provider with fallback
            response, used_provider, failed = await self._execute_with_fallback(
                request, self.provider, self.fallback_providers
            )
            if failed:
                logger.warning(f"Providers failed before success: {', '.join(failed)}")

            logger.info(
                f"Received response from {used_provider}: "
                f"{len(response.content)} chars"
            )

            # Add user message to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "user",
                    prompt,
                    files=files,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name
                )

            # Add assistant response to conversation history
            if self.conversation_memory:
                self.add_message(
                    thread_id,
                    "assistant",
                    response.content,
                    workflow_name=self.name,
                    model_provider=self.provider.provider_name,
                    model_name=response.model
                )

            # Build successful result
            result.success = True
            result.synthesis = response.content
            result.add_step(
                step_number=1,
                content=response.content,
                model=response.model
            )

            # Add metadata
            result.metadata.update({
                'thread_id': thread_id,
                'provider': self.provider.provider_name,
                'model': response.model,
                'usage': response.usage,
                'stop_reason': response.stop_reason,
                'is_continuation': is_valid_continuation,
                'conversation_length': self._get_conversation_length(thread_id)
            })

            logger.info(f"Chat workflow completed successfully for thread: {thread_id}")

            # Emit workflow complete
            emit_workflow_complete("chat")

        except Exception as e:
            logger.error(f"Chat workflow failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.metadata['thread_id'] = thread_id

        # Store result
        self._result = result
        return result

    def _build_prompt_with_history(
        self,
        prompt: str,
        thread_id: str,
        files: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt with conversation history and file context if available.

        Args:
            prompt: Current user prompt
            thread_id: Thread ID to load history from
            files: Optional list of file paths to include in context

        Returns:
            Full prompt string with conversation history and file contents prepended
        """
        context_parts = []

        # Add file contents if provided
        if files:
            context_parts.append("File context:\n")
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    context_parts.append(f"\n--- File: {file_path} ---\n")
                    context_parts.append(file_content)
                    context_parts.append(f"\n--- End of {file_path} ---\n")
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    context_parts.append(f"\n--- File: {file_path} (Failed to read: {e}) ---\n")

        # If no conversation memory, return prompt with file context only
        if not self.conversation_memory:
            if context_parts:
                context_parts.append(f"\n{prompt}")
                return "\n".join(context_parts)
            return prompt

        # Try to load conversation history
        messages = self.resume_conversation(thread_id)

        # Add conversation history
        if messages:
            context_parts.append("\nPrevious conversation:\n")
            for msg in messages:
                role_label = msg.role.upper()
                context_parts.append(f"{role_label}: {msg.content}\n")

                # Include file references from history if present
                if msg.files:
                    context_parts.append(f"  [Referenced files: {', '.join(msg.files)}]\n")

            # Add current user prompt with USER: prefix when there's history
            context_parts.append(f"\nUSER: {prompt}")
        else:
            # No history - just add prompt (with file context if any)
            if context_parts:
                context_parts.append(f"\n{prompt}")
            else:
                return prompt

        full_prompt = "\n".join(context_parts)

        logger.debug(
            f"Built prompt with {len(messages) if messages else 0} previous messages, "
            f"{len(files) if files else 0} files, total length: {len(full_prompt)}"
        )

        return full_prompt

    def _get_conversation_length(self, thread_id: str) -> int:
        """
        Get the number of messages in a conversation thread.

        Args:
            thread_id: Thread ID to check

        Returns:
            Number of messages in the thread, or 0 if not available
        """
        if not self.conversation_memory:
            return 0

        thread = self.get_thread(thread_id)
        if not thread:
            return 0

        return len(thread.messages)

    def get_provider(self) -> ModelProvider:
        """
        Get the configured provider.

        Returns:
            The ModelProvider instance used by this workflow
        """
        return self.provider

    def clear_conversation(self, thread_id: str) -> bool:
        """
        Clear conversation history for a specific thread.

        Args:
            thread_id: Thread ID to clear

        Returns:
            True if cleared successfully, False if not available
        """
        if not self.conversation_memory:
            return False

        # ConversationMemory doesn't have a delete method yet,
        # so this is a placeholder for future implementation
        logger.warning("clear_conversation not yet implemented in ConversationMemory")
        return False

    def get_message_count(self, thread_id: Optional[str] = None) -> int:
        """
        Get total message count across all threads or for a specific thread.

        Args:
            thread_id: Optional thread ID to count messages for.
                      If None, returns total across all threads.

        Returns:
            Number of messages
        """
        if not self.conversation_memory:
            return 0

        if thread_id:
            return self._get_conversation_length(thread_id)

        # Total across all threads would require accessing all threads
        # This is a placeholder for future enhancement
        logger.warning("Total message count across threads not yet implemented")
        return 0

    def validate_config(self) -> bool:
        """
        Validate the workflow configuration.

        Checks that the provider is properly configured and has a valid API key.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.provider is None:
            logger.error("Provider is None")
            return False

        if not self.provider.validate_api_key():
            logger.warning(f"Provider {self.provider.provider_name} API key validation failed")
            return False

        return True

    def __repr__(self) -> str:
        """String representation of the workflow."""
        memory_status = "with memory" if self.conversation_memory else "no memory"
        return (
            f"ChatWorkflow(provider='{self.provider.provider_name}', "
            f"{memory_status})"
        )
