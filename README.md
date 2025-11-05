# ModelChorus - Claude Code Plugin

Multi-model AI workflow orchestration for Claude Code. Provides powerful skills for debugging, code review, consensus building, deep thinking, and more.

## Overview

ModelChorus brings sophisticated multi-model AI workflows to Claude Code through an easy-to-use plugin. Access advanced workflows like systematic debugging, comprehensive code reviews, and multi-perspective consensus building directly from your Claude Code sessions.

## Installation

### As a Claude Code Plugin

Install directly from Claude Code:

```
/plugin add https://github.com/tylerburleigh/claude-model-chorus
```

Or add locally for development:

```
/plugin add /path/to/claude-model-chorus
```

### Post-Install Setup

After installing the plugin, install the Python package:

```bash
cd ~/.claude/plugins/model-chorus/modelchorus
pip install -e .
```

## Available Skills

The plugin provides 7 powerful workflow skills:

### 1. **Chat** - Collaborative Thinking Partner
```
Use Skill(model-chorus:chat) for:
- Brainstorming and idea exploration
- Getting second opinions on technical decisions
- Quick consultations on development questions
```

### 2. **ThinkDeep** - Deep Problem Analysis
```
Use Skill(model-chorus:thinkdeep) for:
- Complex architecture decisions
- Performance challenge analysis
- Security assessments
- Multi-step investigation and reasoning
```

### 3. **Debug** - Systematic Debugging
```
Use Skill(model-chorus:debug) for:
- Root cause analysis of bugs
- Investigating mysterious errors
- Performance debugging
- Race conditions and timing issues
```

### 4. **Consensus** - Multi-Model Decision Making
```
Use Skill(model-chorus:consensus) for:
- Technology evaluation
- Architecture decisions
- Feature proposal reviews
- Risk assessments
- Getting diverse AI perspectives
```

### 5. **CodeReview** - Comprehensive Code Review
```
Use Skill(model-chorus:codereview) for:
- Quality, security, performance, architecture review
- Pre-commit code validation
- Pull request analysis
- Finding bugs and vulnerabilities
```

### 6. **PreCommit** - Git Change Validation
```
Use Skill(model-chorus:precommit) for:
- Reviewing changes before commit
- Security scanning for exposed secrets
- Test coverage verification
- Impact assessment
```

### 7. **Planner** - Complex Task Planning
```
Use Skill(model-chorus:planner) for:
- Breaking down complex projects
- System design planning
- Migration strategies
- Incremental planning with revisions
```

## Quick Start Commands

The plugin provides convenient slash commands:

- `/mc-chat` - Quick chat with external AI models
- `/mc-debug` - Start debugging workflow
- `/mc-review` - Launch code review

### Examples

```
/mc-chat What's the best caching strategy for a high-traffic API?
```

```
/mc-debug API endpoint randomly returns 500 errors under load
```

```
/mc-review src/auth/middleware.py --focus security
```

## Skill Usage Examples

### Example 1: Debug a Performance Issue

```
Use Skill(model-chorus:debug) to investigate:

Issue: "Memory usage grows from 200MB to 4GB over 24 hours"

Files:
- /home/user/project/upload/handler.py
- /home/user/project/storage/files.py

Steps to reproduce:
1. Start service
2. Upload 1000 files
3. Wait 24 hours
4. Memory at 4GB, never releases
```

### Example 2: Get Consensus on Architecture

```
Use Skill(model-chorus:consensus) to evaluate:

Proposal: "Migrate from REST to GraphQL for our API"

Models to consult:
- gpt-5-pro (stance: for) - Argue benefits
- gemini-2.5-pro (stance: against) - Identify risks
- gpt-5-codex (stance: neutral) - Balanced analysis

Files:
- /home/user/project/api/endpoints.py
- /home/user/project/docs/api_spec.yaml
```

### Example 3: Comprehensive Code Review

```
Use Skill(model-chorus:codereview) to review:

Files:
- /home/user/project/auth/middleware.py
- /home/user/project/auth/jwt_handler.py
- /home/user/project/tests/test_auth.py

Review type: full
Focus: security, test coverage
```

### Example 4: Pre-Commit Validation

```
Use Skill(model-chorus:precommit) to validate:

Repository: /home/user/project
Review: staged changes
Focus: security, missing tests, completeness
```

## Requirements

- **Python**: >=3.9
- **Claude Code**: Latest version
- **Zen MCP Server**: Required for workflow execution (usually auto-installed)

## Python Package

ModelChorus is also available as a standalone Python package for direct use:

```bash
cd modelchorus/
pip install -e .
```

See [`modelchorus/README.md`](modelchorus/README.md) for Python package documentation.

## CLI Usage

The Python package includes a CLI for running workflows:

```bash
# Run consensus workflow
modelchorus consensus "Explain quantum computing" -p claude -p gemini

# List available providers
modelchorus list-providers

# Show version
modelchorus version
```

## Supported AI Models

ModelChorus supports multiple AI providers:

### Via Zen MCP (Skills)
- Claude (Sonnet, Opus)
- GPT-5 Pro, GPT-5 Codex, GPT-5, GPT-5 Mini
- Gemini 2.5 Pro
- And more (see `/listmodels`)

### Via CLI (Direct)
- Claude (Anthropic)
- Codex (OpenAI CLI)
- Gemini (Google)
- Cursor Agent (Cursor CLI)

## Architecture

```
claude-model-chorus/
├── .claude-plugin/          # Plugin configuration
│   ├── plugin.json          # Plugin manifest
│   └── marketplace.json     # Marketplace distribution
├── skills/                  # Skill definitions
│   ├── chat/
│   ├── thinkdeep/
│   ├── debug/
│   ├── consensus/
│   ├── codereview/
│   ├── precommit/
│   └── planner/
├── commands/                # Slash commands
│   ├── mc-chat.md
│   ├── mc-debug.md
│   └── mc-review.md
└── modelchorus/            # Python package
    ├── src/modelchorus/    # Source code
    ├── tests/              # Test suite
    └── README.md           # Python package docs
```

## Development

### Local Plugin Development

1. Clone the repository:
```bash
git clone https://github.com/tylerburleigh/claude-model-chorus.git
cd claude-model-chorus
```

2. Install Python package:
```bash
cd modelchorus
pip install -e ".[dev]"
```

3. Add plugin to Claude Code:
```
/plugin add /path/to/claude-model-chorus
```

4. Test skills:
```
Use Skill(model-chorus:chat): "Test question"
```

### Running Tests

```bash
cd modelchorus/
pytest
```

### Code Quality

```bash
# Format
black .

# Lint
ruff check .

# Type check
mypy modelchorus
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Troubleshooting

**Plugin not loading?**
- Check `.claude/plugins/model-chorus/` exists
- Verify `plugin.json` is valid JSON
- Restart Claude Code

**Skills not working?**
- Ensure Zen MCP server is installed
- Check Python package is installed
- Verify API keys are configured

**Command not found?**
- Check `commands/` directory exists
- Verify command files are `.md` format
- Restart Claude Code

## Links

- **GitHub**: https://github.com/tylerburleigh/claude-model-chorus
- **Issues**: https://github.com/tylerburleigh/claude-model-chorus/issues
- **Documentation**: See individual skill SKILL.md files
- **Python Package**: [`modelchorus/README.md`](modelchorus/README.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

ModelChorus leverages the power of multiple AI models to deliver robust, well-reasoned results for complex development tasks. Built for the Claude Code community.
