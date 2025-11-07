---
name: modelchorus-setup
description: First-time setup for ModelChorus - configures project settings and permissions
---

# ModelChorus First-Time Setup

This command performs first-time setup for ModelChorus in your project.

## What This Does

1. Checks if the `modelchorus` Python package is installed
2. Interactively prompts for configuration preferences
3. Creates `.modelchorusrc` configuration file
4. Configures `.claude/settings.local.json` with necessary permissions for ModelChorus CLI
5. Validates the configuration

## Workflow

### Step 1: Check Package Installation

First, check if the modelchorus package is installed:

```bash
python -m modelchorus.cli.setup check-install
```

The script will return JSON with an `installed` field.

**If `installed: false`:**

Display message:
```
⚠️  ModelChorus package is not installed

The modelchorus package needs to be installed before setup can continue.
Please install it using one of these methods:

  • From PyPI: pip install modelchorus
  • Development mode: pip install -e .

After installation, run /modelchorus-setup again.
```

Then exit the setup (do not proceed).

**If `installed: true`:**

Display message:
```
✅ ModelChorus package is installed (version X.X.X)
```

Then proceed to Step 2.

### Step 2: Check Configuration File

Check if `.modelchorusrc` configuration file already exists:

```bash
python -m modelchorus.cli.setup check-config --project .
```

The script will return JSON with an `exists` field.

**If `exists: true`:**

Display message with the path:
```
✅ Configuration file already exists: <path>

You can:
  • View config: modelchorus config show
  • Validate config: modelchorus config validate
  • Edit manually: <path>
```

Then skip to Step 3.

**If `exists: false`:**

Proceed to create configuration by gathering user preferences.

#### Gather Configuration Preferences

**IMPORTANT:** Use a tiered approach to configuration. First ask which tier, then ask questions based on that tier.

##### Tier Selection

Use AskUserQuestion tool to ask about setup tier:

**Question 1 - Setup Tier:**
- **Header**: "Setup Tier"
- **Question**: "How much configuration do you want to set up now?"
- **Options**:
  1. Quick Setup - Just the essentials (provider, temperature, timeout, max_tokens)
  2. Standard Setup - Common workflows configured (Quick + consensus, research, thinkdeep)
  3. Advanced Setup - Full configuration (Standard + all 6 workflows with detailed options)

Based on the tier chosen, proceed with the appropriate question set below.

---

##### QUICK TIER Questions

If user chose "Quick Setup", ask these 4 questions:

**Question 1:**
- **Header**: "Provider"
- **Question**: "Which AI provider should be the default for all workflows?"
- **Options**:
  1. Claude (Anthropic) - Recommended for most tasks
  2. Gemini (Google) - Good for creative tasks
  3. Codex (OpenAI) - Alternative option
  4. Cursor Agent - For cursor integration

**Question 2:**
- **Header**: "Temperature"
- **Question**: "What temperature should be used for generation (creativity level)?"
- **Options**:
  1. Conservative (0.5) - More focused and deterministic
  2. Balanced (0.7) - Recommended default
  3. Creative (0.9) - More varied and creative

**Question 3:**
- **Header**: "Timeout"
- **Question**: "What timeout (in seconds) should be used for provider requests?"
- **Options**:
  1. Fast (60s) - For quick responses
  2. Standard (120s) - Recommended default
  3. Extended (180s) - For complex tasks

**Question 4:**
- **Header**: "Max Tokens"
- **Question**: "What should be the maximum response length?"
- **Options**:
  1. Short (1000) - For brief responses
  2. Medium (2000) - Recommended default
  3. Long (4000) - For detailed responses

Then create config:
```bash
python -m modelchorus.cli.setup create-tiered-config --project . \
  --tier quick \
  --provider <chosen_provider> \
  --temperature <chosen_temperature> \
  --timeout <chosen_timeout> \
  --max-tokens <chosen_max_tokens>
```

---

##### STANDARD TIER Questions

If user chose "Standard Setup", ask Quick tier questions PLUS these additional questions:

**Questions 1-4:** Same as Quick tier

**Question 5:**
- **Header**: "Multi-Model"
- **Question**: "Which providers should be used for multi-model workflows (consensus, research)?"
- **multiSelect**: true
- **Options**:
  1. Claude (Anthropic) - Best for reasoning and analysis
  2. Gemini (Google) - Good for creative tasks
  3. Codex (OpenAI) - Alternative perspective

**Question 6:**
- **Header**: "Consensus"
- **Question**: "How should consensus workflow combine multiple model responses?"
- **Options**:
  1. Show all responses - Display each model's response separately
  2. Synthesize - Combine responses into one unified answer
  3. Vote - Use majority opinion for binary questions

**Question 7:**
- **Header**: "Research"
- **Question**: "What citation style should research workflow use?"
- **Options**:
  1. Informal - Simple mentions without formal citations
  2. Academic - Formal academic style with references
  3. APA - APA citation format
  4. MLA - MLA citation format

**Question 8:**
- **Header**: "Research"
- **Question**: "How thorough should the research workflow be?"
- **Options**:
  1. Quick - Fast overview of the topic
  2. Thorough - Balanced depth and speed (recommended)
  3. Comprehensive - Deep dive with extensive sources

**Question 9:**
- **Header**: "ThinkDeep"
- **Question**: "What reasoning depth should thinkdeep workflow use?"
- **Options**:
  1. Low - Basic reasoning with fewer steps
  2. Medium - Balanced reasoning (recommended)
  3. High - Deep reasoning with extensive analysis

Then create config:
```bash
python -m modelchorus.cli.setup create-tiered-config --project . \
  --tier standard \
  --provider <chosen_provider> \
  --temperature <chosen_temperature> \
  --timeout <chosen_timeout> \
  --max-tokens <chosen_max_tokens> \
  --consensus-providers <space_separated_providers> \
  --consensus-strategy <chosen_strategy> \
  --research-citation <chosen_citation> \
  --research-depth <chosen_depth> \
  --thinkdeep-mode <chosen_mode>
```

---

##### ADVANCED TIER Questions

If user chose "Advanced Setup", ask Standard tier questions PLUS:

**Questions 1-9:** Same as Standard tier

**Question 10:**
- **Header**: "Ideate"
- **Question**: "Which providers should be used for ideate workflow (creative brainstorming)?"
- **multiSelect**: true
- **Options**:
  1. Claude (Anthropic) - Structured creative thinking
  2. Gemini (Google) - Highly creative ideas
  3. Codex (OpenAI) - Alternative perspective

Then create config (same command as Standard, with additional --ideate-providers):
```bash
python -m modelchorus.cli.setup create-tiered-config --project . \
  --tier advanced \
  --provider <chosen_provider> \
  --temperature <chosen_temperature> \
  --timeout <chosen_timeout> \
  --max-tokens <chosen_max_tokens> \
  --consensus-providers <space_separated_providers> \
  --consensus-strategy <chosen_strategy> \
  --research-citation <chosen_citation> \
  --research-depth <chosen_depth> \
  --thinkdeep-mode <chosen_mode> \
  --ideate-providers <space_separated_providers>
```

---

##### Value Mappings

Use these mappings when constructing the CLI command:

**Temperature mapping:**
- Conservative → 0.5
- Balanced → 0.7
- Creative → 0.9

**Timeout mapping:**
- Fast → 60.0
- Standard → 120.0
- Extended → 180.0

**Max tokens mapping:**
- Short → 1000
- Medium → 2000
- Long → 4000

**Provider mapping:**
- Claude (Anthropic) → claude
- Gemini (Google) → gemini
- Codex (OpenAI) → codex
- Cursor Agent → cursor-agent

**Consensus strategy mapping:**
- Show all responses → all_responses
- Synthesize → synthesize
- Vote → vote

**Research citation mapping:**
- Informal → informal
- Academic → academic
- APA → apa
- MLA → mla

**Research depth mapping:**
- Quick → quick
- Thorough → thorough
- Comprehensive → comprehensive

**ThinkDeep mode mapping:**
- Low → low
- Medium → medium
- High → high

The script will create the `.modelchorusrc` file and return JSON with `success: true` if successful.

**On success:**
```
✅ Created configuration file: .modelchorusrc

Your settings:
  • Default provider: <provider>
  • Temperature: <temperature>
  • Timeout: <timeout>s
  • Max tokens: <max_tokens> (if configured)
  • Tier: <tier>
```

**On error:**
```
❌ Failed to create configuration file

Error: <error_message>

You can create the file manually by copying .modelchorusrc.example
```

#### Add Configuration to .gitignore

After successfully creating the config file, add it to `.gitignore`:

```bash
python -m modelchorus.cli.setup add-to-gitignore --project .
```

This will add `.modelchorusrc` and its variants to the project's `.gitignore` file.

**On success:**
```
✅ Added .modelchorusrc to .gitignore
```

If the entries are already present, the command will report that and continue gracefully.

Then proceed to Step 3.

### Step 3: Validate Configuration

Validate the configuration file:

```bash
python -m modelchorus.cli.setup validate-config --project .
```

**If validation succeeds:**
```
✅ Configuration is valid!
```

**If validation fails:**
```
❌ Configuration has errors:

<error_details>

Please fix the configuration file and run validation again.
```

Then proceed to Step 4.

### Step 4: Check Permissions

Check if Claude Code permissions are configured:

```bash
python -m modelchorus.cli.setup check-permissions --project .
```

The script will return JSON with a `configured` field.

**If `configured: true`:**

Display message:
```
✅ ModelChorus permissions are already configured!
```

Then proceed to Step 5.

**If `configured: false`:**

Ask user if they want to add permissions:

Use AskUserQuestion tool:
- **Header**: "Permissions"
- **Question**: "Add ModelChorus CLI permissions to Claude Code settings?"
- **Options**:
  1. Yes, add permissions (recommended)
  2. No, skip for now

**If user chooses "Yes, add permissions":**

Run the setup command:

```bash
python -m modelchorus.cli.setup add-permissions --project .
```

This will add the permission `Bash(modelchorus:*)` to `.claude/settings.local.json`.

**On success:**
```
✅ Added ModelChorus permissions to Claude Code

Permissions added:
  • Bash(modelchorus:*)

Saved to: .claude/settings.local.json
```

**If user chooses "No, skip for now":**
```
⏭️  Skipped permissions setup

To add permissions later, run:
  python -m modelchorus.cli.setup add-permissions --project .
```

Then proceed to Step 5.

### Step 5: Show Success & Next Steps

After successful configuration, display a summary:

```
═══════════════════════════════════════════════════════════
           ModelChorus Setup Complete
═══════════════════════════════════════════════════════════

✅ Package: Installed (version X.X.X)

✅ Configuration: .modelchorusrc

✅ Permissions: Configured (.claude/settings.local.json)

═══════════════════════════════════════════════════════════

Next steps:
• Try chatting: modelchorus chat "Hello, how are you?"
• View config: modelchorus config show
• See all commands: modelchorus --help
• List providers: modelchorus list-providers

For more info, see the ModelChorus README.
═══════════════════════════════════════════════════════════
```

## What Gets Configured

### Configuration File (`.modelchorusrc`)

The setup creates a `.modelchorusrc` file in YAML format with:

**Global Settings:**
- `default_provider` - Default AI provider (claude, gemini, codex, cursor-agent)
- `generation.temperature` - Default temperature (0.0-1.0)
- `generation.timeout` - Default timeout in seconds
- `generation.max_tokens` - Optional max tokens limit

**Example configuration:**
```yaml
# ModelChorus Configuration
default_provider: claude

generation:
  temperature: 0.7
  timeout: 120.0
```

You can manually edit this file later or use `modelchorus config init` to regenerate.

### Permissions (`.claude/settings.local.json`)

The setup adds this permission to the `allow` list:

- `Bash(modelchorus:*)` - Allow all ModelChorus CLI commands

This permission is project-specific and non-destructive - it only allows running the modelchorus CLI tool.

## Important Notes

**Package Installation:**
- Setup will NOT install the package for you
- You must install `modelchorus` before running setup
- Use `pip install modelchorus` or `pip install -e .` for development

**Configuration:**
- Setup is **non-destructive** - won't overwrite existing config files
- Can be run multiple times safely (skips already-configured items)
- Configuration is project-specific (stored in project's `.modelchorusrc`)

**Permissions:**
- Permissions are added to project's `.claude/settings.local.json`
- Only adds missing permissions, preserves existing ones
- Can be managed manually in settings.local.json if needed

**Frequency:**
- You only need to run this once per project
- Run again if you need to reconfigure settings
- Configuration files can be edited manually anytime

## Error Handling

If setup fails:
- Check that you have write permissions in the project directory
- Ensure `modelchorus` package is installed
- Check that `.claude/` directory can be created/modified
- Review error output from the setup commands

## Manual Configuration

If you prefer to configure manually:

1. **Create `.modelchorusrc`:**
   ```bash
   cp .modelchorusrc.example .modelchorusrc
   ```
   Then edit as needed.

2. **Add permissions to `.claude/settings.local.json`:**
   Add `"Bash(modelchorus:*)"` to the `permissions.allow` array.

3. **Validate configuration:**
   ```bash
   modelchorus config validate
   ```

## Integration

After running this setup:
- Use `modelchorus chat` for conversations
- Use `modelchorus consensus` for multi-model responses
- Use `modelchorus thinkdeep` for deep reasoning
- Use `modelchorus argument` for dialectical analysis
- Use `modelchorus ideate` for creative brainstorming
- Use `modelchorus research` for systematic research

See the README for detailed workflow documentation.
