# ModelChorus Examples

This directory contains examples demonstrating how to use ModelChorus.

## Quick Start Examples

### Basic Workflow (`basic_workflow.py`)

Demonstrates the fundamental workflow pattern in ModelChorus:
- Creating a custom workflow by inheriting from `BaseWorkflow`
- Registering workflows using the `@WorkflowRegistry.register()` decorator
- Creating workflow requests with `WorkflowRequest`
- Executing workflows and processing results

**Run it:**
```bash
cd examples
python basic_workflow.py
```

**What you'll learn:**
- How to define a custom workflow class
- How to use the workflow registry
- How to create and execute workflow requests
- How to process workflow results and steps

### Provider Integration (`provider_integration.py`)

Shows how to integrate AI providers with ModelChorus:
- Implementing a custom provider by inheriting from `ModelProvider`
- Defining model capabilities and configurations
- Creating generation requests
- Processing provider responses

**Run it:**
```bash
cd examples
python provider_integration.py
```

**What you'll learn:**
- How to create a custom provider implementation
- How to define model capabilities
- How to check for vision and other capabilities
- How to generate responses from providers

## Example Structure

Each example is self-contained and can be run independently:

```
examples/
├── README.md                    # This file
├── basic_workflow.py           # Basic workflow example
└── provider_integration.py     # Provider integration example
```

## Next Steps

After running these examples:

1. **Explore Real Providers**: Check the `modelchorus/providers/` directory for actual provider implementations (Anthropic, OpenAI, Google)

2. **Try Built-in Workflows**: Explore the `modelchorus/workflows/` directory for production-ready workflows like ThinkDeep, Debug, and Consensus

3. **Build Custom Workflows**: Use these examples as templates to create your own specialized workflows

4. **Read the Documentation**: Visit [https://modelchorus.readthedocs.io](https://modelchorus.readthedocs.io) for comprehensive guides

## Notes

- These examples use placeholder implementations to demonstrate the API structure
- Real provider implementations require valid API keys
- Actual workflow implementations are more sophisticated than these examples
- See the main README.md for installation and setup instructions
