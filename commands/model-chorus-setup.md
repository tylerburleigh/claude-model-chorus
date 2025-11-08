---
name: model-chorus-setup
description: First-time setup for ModelChorus - configures project settings and permissions
---

# ModelChorus First-Time Setup

This command performs first-time setup for ModelChorus in your project.

## What This Does

1. Checks if the `model-chorus` Python package is installed
2. Checks version compatibility between package and plugin
3. Checks which CLI providers are available on your system
4. Offers setup modes: Express (0 questions), Quick (2), Standard (4), or Advanced (6)
5. Creates `.model-chorusrc` configuration file with:
   - Default provider models (sonnet, gemini-2.5-pro, gpt-5-codex, composer-1)
   - Automatic fallback provider configuration
   - Workflow-specific settings
6. Validates the configuration
7. Configures `.claude/settings.local.json` with necessary permissions for ModelChorus CLI

## Workflow

### Step 1: Check Package Installation

First, check if the model-chorus package is installed:

```bash
python -m model_chorus.cli.setup check-install
```

The script will return JSON with an `installed` field.

**If `installed: false`:**

Display message:
```
⚠️  ModelChorus package is not installed

The model-chorus package needs to be installed before setup can continue.
Please install it using one of these methods:

  • From PyPI: pip install model-chorus
  • Development mode: pip install -e .

After installation, run /model-chorus-setup again.
```

Then exit the setup (do not proceed).

**If `installed: true`:**

Display message:
```
✅ ModelChorus package is installed (version X.X.X)
```

Then proceed to Step 2.

### Step 2: Check Version Compatibility

Check if the installed package version matches the plugin version:

```bash
python -m model_chorus.cli.setup check-version
```

The script will return JSON with a `compatible` field and version details.

**If `compatible: true` and versions match:**

Display message:
```
✅ Version check passed: Package and plugin both at version X.X.X
```

Then proceed to Step 3.

**If `compatible: true` but package is newer than plugin:**

Display warning:
```
⚠️  Package version (X.X.X) is newer than plugin version (Y.Y.Y)

This is okay but the plugin may need updating.
You can continue setup.
```

Then proceed to Step 3.

**If `compatible: false` (package version is lower than plugin version):**

Display message:
```
⚠️  Package version mismatch detected!

  Plugin version: X.X.X
  Package version: Y.Y.Y (older)

Upgrading package to match plugin version...
```

Then automatically reinstall the package:

```bash
pip install --upgrade model-chorus
```

After reinstallation, verify the new version:

```bash
python -m model_chorus.cli.setup check-install
```

Display message:
```
✅ Package upgraded to version X.X.X
```

Then proceed to Step 3.

**If version check fails with error:**

Display message:
```
⚠️  Could not verify version compatibility

Error: <error_message>

Continuing with setup anyway...
```

Then proceed to Step 3.

### Step 3: Check Available Providers

Check which CLI providers are available on the system:

```bash
python -m model_chorus.cli.setup check-available-providers
```

The script will return JSON with `available` (list of available provider names) and `unavailable` (list of unavailable providers with error details).

**Display the results:**

```
✅ Found X available provider(s): <provider1>, <provider2>, ...

Unavailable providers: <provider_name> (<error>)
```

**IMPORTANT:** Store the list of available providers to use in Step 5 when asking the user which provider to select. Only show available providers as options.

Then proceed to Step 4.

### Step 4: Check Configuration File

Check if `.model-chorusrc` configuration file already exists:

```bash
python -m model_chorus.cli.setup check-config --project .
```

The script will return JSON with an `exists` field.

**If `exists: true`:**

Display message with the path:
```
✅ Configuration file already exists: <path>

You can:
  • View config: model-chorus config show
  • Validate config: model-chorus config validate
  • Edit manually: <path>
```

Then skip to Step 6.

**If `exists: false`:**

Proceed to Step 5 to create configuration by gathering user preferences.

### Step 5: Choose Setup Mode

**IMPORTANT:** This is the key decision point. The Express mode requires ZERO questions and auto-configures everything based on detected providers.

Use AskUserQuestion tool to ask about setup mode:

**Question - Setup Mode:**
- **Header**: "Setup Mode"
- **Question**: "How would you like to set up ModelChorus?"
- **Options**:
  1. **Express (Recommended)** - Auto-configure with smart defaults [0 questions]
  2. **Quick** - Choose provider only [1 question]
  3. **Standard** - Configure common workflows [3 questions]
  4. **Advanced** - Full configuration control [5 questions]

Based on the mode chosen, proceed with the appropriate setup flow below.

---

### Step 5.1: EXPRESS MODE (Zero Questions)

If user chose "Express", run the express config creation:

```bash
python -m model_chorus.cli.setup create-express-config --project .
```

The script will:
- Auto-select primary provider (claude → gemini → codex → cursor-agent)
- Configure all other available providers as fallbacks
- Set default models for each provider:
  - Claude: sonnet
  - Gemini: gemini-2.5-pro
  - Codex: gpt-5-codex
  - Cursor-agent: composer-1
- Use standard timeout (600s / 10 minutes)
- Configure all workflows with balanced defaults

**On success:**
```
✅ Express setup complete!

Configuration:
  • Default provider: <primary_provider>
  • Fallback providers: <other_providers>
  • Models: <configured_models>
  • Timeout: 600s (10 minutes)

All workflows configured with smart defaults.
```

Then proceed to Step 6 (Validate Configuration).

---

### Step 5.2: QUICK/STANDARD/ADVANCED MODES

If user chose Quick, Standard, or Advanced, gather configuration through questions.

**All provider selection questions should ONLY show providers that are available (from Step 3).** The setup script will automatically configure the other available providers as fallbacks.

---

##### QUICK MODE Questions

If user chose "Quick", ask this 1 question:

**Question 1:**
- **Header**: "Provider"
- **Question**: "Which AI provider should be the default for all workflows?"
- **Options**: ONLY include providers that are available from Step 3
  - If "claude" is available: Claude (Anthropic) - Recommended for most tasks
  - If "gemini" is available: Gemini (Google) - Good for creative tasks
  - If "codex" is available: Codex (OpenAI) - Alternative option
  - If "cursor-agent" is available: Cursor Agent - For cursor integration
- **Note**: The other available providers will be automatically configured as fallbacks
- **Note**: Default models will be configured automatically:
  - Claude: sonnet
  - Gemini: gemini-2.5-pro
  - Codex: gpt-5-codex
  - Cursor-agent: composer-1
- **Note**: Standard timeout of 600s (10 minutes) will be used

Then create config:
```bash
python -m model_chorus.cli.setup create-tiered-config --project . \
  --tier quick \
  --provider <chosen_provider>
```

---

##### STANDARD MODE Questions

If user chose "Standard", ask Quick question PLUS these 2 additional questions (3 total):

**Question 1:** Same as Quick mode (Provider)

**Question 2:**
- **Header**: "Multi-Model"
- **Question**: "Which providers should be used for multi-model workflows (consensus, ideate)?"
- **multiSelect**: true
- **Options**: ONLY include providers that are available from Step 3
  - If "claude" is available: Claude (Anthropic) - Best for reasoning and analysis
  - If "gemini" is available: Gemini (Google) - Good for creative tasks
  - If "codex" is available: Codex (OpenAI) - Alternative perspective
  - If "cursor-agent" is available: Cursor Agent - For cursor integration

**Question 4:**
- **Header**: "Consensus"
- **Question**: "How should consensus workflow combine multiple model responses?"
- **Options**:
  1. Show all responses - Display each model's response separately
  2. Synthesize - Combine responses into one unified answer
  3. Vote - Use majority opinion for binary questions

**Note**: ThinkDeep workflow will be configured with balanced defaults:
- ThinkDeep: medium reasoning mode

Then create config:
```bash
python -m model_chorus.cli.setup create-tiered-config --project . \
  --tier standard \
  --provider <chosen_provider> \
  --timeout <chosen_timeout> \
  --consensus-providers <space_separated_providers> \
  --consensus-strategy <chosen_strategy>
```

---

##### ADVANCED MODE Questions

If user chose "Advanced", ask Standard questions PLUS these 2 additional questions (6 total):

**Questions 1-4:** Same as Standard mode (Provider, Timeout, Multi-Model providers, Consensus strategy)

**Question 5:**
- **Header**: "ThinkDeep"
- **Question**: "What reasoning depth should thinkdeep workflow use?"
- **Options**:
  1. Low - Basic reasoning with fewer steps
  2. Medium - Balanced reasoning (recommended)
  3. High - Deep reasoning with extensive analysis

Then create config:
```bash
python -m model_chorus.cli.setup create-tiered-config --project . \
  --tier advanced \
  --provider <chosen_provider> \
  --timeout <chosen_timeout> \
  --consensus-providers <space_separated_providers> \
  --consensus-strategy <chosen_strategy> \
  --thinkdeep-mode <chosen_mode>
```

---

##### Value Mappings

Use these mappings when constructing the CLI command:

**Timeout mapping:**
- Fast → 60.0
- Standard → 120.0
- Extended → 180.0

**Provider mapping:**
- Claude (Anthropic) → claude
- Gemini (Google) → gemini
- Codex (OpenAI) → codex
- Cursor Agent → cursor-agent

**Consensus strategy mapping:**
- Show all responses → all_responses
- Synthesize → synthesize
- Vote → vote

**ThinkDeep mode mapping:**
- Low → low
- Medium → medium
- High → high

The script will create the `.model-chorusrc` file and return JSON with `success: true` if successful.

**On success:**
```
✅ Created configuration file: .model-chorusrc

Your settings:
  • Default provider: <provider>
  • Timeout: <timeout>s
  • Provider models: <configured_models>
  • Tier: <tier>
```

**On error:**
```
❌ Failed to create configuration file

Error: <error_message>

You can create the file manually by copying .model-chorusrc.example
```

#### Add Configuration to .gitignore

After successfully creating the config file, add it to `.gitignore`:

```bash
python -m model_chorus.cli.setup add-to-gitignore --project .
```

This will add `.model-chorusrc` and its variants to the project's `.gitignore` file.

**On success:**
```
✅ Added .model-chorusrc to .gitignore
```

If the entries are already present, the command will report that and continue gracefully.

Then proceed to Step 6.

### Step 6: Validate Configuration

Validate the configuration file:

```bash
python -m model_chorus.cli.setup validate-config --project .
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

Then proceed to Step 7.

### Step 7: Check Permissions

Check if Claude Code permissions are configured:

```bash
python -m model_chorus.cli.setup check-permissions --project .
```

The script will return JSON with a `configured` field.

**If `configured: true`:**

Display message:
```
✅ ModelChorus permissions are already configured!
```

Then proceed to Step 8.

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
python -m model_chorus.cli.setup add-permissions --project .
```

This will add the permissions `Bash(model-chorus:*)` and `Skill(model-chorus:*)` to `.claude/settings.local.json`.

**On success:**
```
✅ Added ModelChorus permissions to Claude Code

Permissions added:
  • Bash(model-chorus:*)
  • Skill(model-chorus:*)

Saved to: .claude/settings.local.json
```

**If user chooses "No, skip for now":**
```
⏭️  Skipped permissions setup

To add permissions later, run:
  python -m model_chorus.cli.setup add-permissions --project .
```

Then proceed to Step 8.

### Step 8: Show Success & Next Steps

After successful configuration, display a summary:

```
═══════════════════════════════════════════════════════════
           ModelChorus Setup Complete
═══════════════════════════════════════════════════════════

✅ Package: Installed (version X.X.X)

✅ Configuration: .model-chorusrc

✅ Permissions: Configured (.claude/settings.local.json)

═══════════════════════════════════════════════════════════

Next steps:
• Try chatting: model-chorus chat "Hello, how are you?"
• View config: model-chorus config show
• See all commands: model-chorus --help
• List providers: model-chorus list-providers

For more info, see the ModelChorus README.
═══════════════════════════════════════════════════════════
```

## What Gets Configured

### Configuration File (`.model-chorusrc`)

The setup creates a `.model-chorusrc` file in YAML format with:

**Global Settings:**
- `default_provider` - Default AI provider (claude, gemini, codex, cursor-agent)
- `providers.<name>.model` - Default model for each provider
- `generation.timeout` - Default timeout in seconds
- `workflows.<name>.*` - Workflow-specific configurations

**Example configuration (Express mode):**
```yaml
# ModelChorus Configuration
default_provider: claude

providers:
  claude:
    model: sonnet
  gemini:
    model: gemini-2.5-pro
  codex:
    model: gpt-5-codex

generation:
  timeout: 120.0

workflows:
  chat:
    fallback_providers:
      - gemini
      - codex
  consensus:
    providers:
      - claude
      - gemini
      - codex
    strategy: all_responses
```

You can manually edit this file later or view it with `model-chorus config show`.

### Permissions (`.claude/settings.local.json`)

The setup adds these permissions to the `allow` list:

- `Bash(model-chorus:*)` - Allow all ModelChorus CLI commands
- `Skill(model-chorus:*)` - Allow all ModelChorus skill invocations

These permissions are project-specific and non-destructive - they only allow running the model-chorus CLI tool and invoking ModelChorus skills.

## Important Notes

**Package Installation:**
- Setup will NOT install the package for you
- You must install `model-chorus` before running setup
- Use `pip install model-chorus` or `pip install -e .` for development

**Configuration:**
- Setup is **non-destructive** - won't overwrite existing config files
- Can be run multiple times safely (skips already-configured items)
- Configuration is project-specific (stored in project's `.model-chorusrc`)

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
- Ensure `model-chorus` package is installed
- Check that `.claude/` directory can be created/modified
- Review error output from the setup commands

## Manual Configuration

If you prefer to configure manually:

1. **Create `.model-chorusrc`:**
   ```bash
   cp .model-chorusrc.example .model-chorusrc
   ```
   Then edit as needed.

2. **Add permissions to `.claude/settings.local.json`:**
   Add `"Bash(model-chorus:*)"` and `"Skill(model-chorus:*)"` to the `permissions.allow` array.

3. **Validate configuration:**
   ```bash
   model-chorus config validate
   ```

## Integration

After running this setup:
- Use `model-chorus chat` for conversations
- Use `model-chorus consensus` for multi-model responses
- Use `model-chorus thinkdeep` for deep reasoning
- Use `model-chorus argument` for dialectical analysis
- Use `model-chorus ideate` for creative brainstorming

See the README for detailed workflow documentation.
