---
name: ideate
description: Creative brainstorming and idea generation with configurable creativity levels and structured output
---

# IDEATE

## Overview

The IDEATE workflow generates creative ideas through structured brainstorming with fine-grained control over creativity and quantity. Unlike simple prompts, IDEATE systematically generates multiple distinct ideas with configurable parameters that let you tune the creative output to your specific needs.

**Key Capabilities:**
- Configurable idea quantity (num_ideas parameter)
- Adjustable creativity via temperature control
- Structured generation with synthesis
- Conversation threading for iterative refinement
- Constraint-based ideation with system prompts

**Use Cases:**
- Product and feature brainstorming with creativity control
- Problem-solving with practical or innovative approaches
- Marketing campaigns from conservative to bold
- Technical innovation with tunable risk levels
- Process improvement ideas at different creativity levels

## When to Use

Use the IDEATE workflow when you need to:

- **Creative exploration** - Generate multiple distinct ideas with control over creativity level
- **Quantity control** - Specify exactly how many ideas you need (3 for quick wins, 10 for comprehensive)
- **Iterative brainstorming** - Build on previous ideas through conversation threading
- **Constrained ideation** - Generate ideas within specific constraints or criteria
- **Tiered creativity** - Explore ideas at different creativity levels (practical to bold)

## When NOT to Use

Avoid the IDEATE workflow when:

| Situation | Use Instead |
|-----------|-------------|
| Need to analyze existing argument | **ARGUMENT** - IDEATE generates new, doesn't evaluate existing |
| Want multiple model perspectives | **CONSENSUS** - IDEATE uses single model |
| Need systematic investigation | **THINKDEEP** - IDEATE is creative, not investigative |
| Simple question or fact | **CHAT** - IDEATE adds overhead for idea generation |
| Already know what you want | Direct implementation - No need for brainstorming |

## Creative Control: Temperature Parameter

The temperature parameter is your primary tool for controlling creativity, risk, and innovation level.

### Temperature Scale Overview

| Temperature | Range | Style | Characteristics |
|-------------|-------|-------|-----------------|
| **0.0-0.4** | Conservative | Practical & Proven | Low risk, optimizations, incremental improvements |
| **0.5-0.6** | Moderate | Balanced Practical | Feasible ideas with some innovation |
| **0.7-0.8** | Balanced | Mix of Safe & Innovative | Recommended for most use cases |
| **0.9-1.0** | Bold | Disruptive & Experimental | High innovation, unconventional approaches |

### Temperature Ranges Explained

### Temperature Ranges Explained

#### Low Creativity (0.0-0.4): Conservative & Practical

**Characteristics:**
- Proven, well-established approaches
- Incremental improvements on existing solutions
- Low-risk ideas focused on optimization
- Practical, implementable concepts
- Conservative thinking

**When to use:**
- Database optimization techniques
- Performance improvements
- Infrastructure upgrades
- Process refinements
- Technical debt reduction

**Example:**
```bash
model-chorus ideate "Database optimization techniques" --temperature 0.4 --num-ideas 5
```

**Expected ideas:**
- Index optimization strategies
- Query performance tuning
- Connection pooling improvements
- Caching layer additions
- Partitioning strategies

---

#### Medium Creativity (0.5-0.7): Balanced Approach

**Characteristics:**
- Mix of practical and creative solutions
- Some innovation with manageable risk
- Balanced between proven and experimental
- Good feasibility-to-innovation ratio
- **Recommended for most use cases**

**When to use:**
- General feature brainstorming
- User engagement strategies
- API design improvements
- Product roadmap planning
- Team productivity enhancements

**Example:**
```bash
model-chorus ideate "User engagement strategies for mobile app" --temperature 0.7
```

**Expected ideas:**
- Push notification personalization
- Gamification elements
- Social sharing features
- Progress tracking systems
- Community-driven content

---

#### High Creativity (0.8-0.9): Innovative & Bold

**Characteristics:**
- Bold, unconventional ideas
- Higher innovation, higher risk
- Challenges existing assumptions
- Explores emerging technologies
- Pushes boundaries

**When to use:**
- Product differentiation strategies
- Disruptive feature concepts
- Marketing campaign creativity
- Competitive advantage ideation
- Innovation workshops

**Example:**
```bash
model-chorus ideate "Disruptive features for task management" --temperature 0.9 --num-ideas 7
```

**Expected ideas:**
- AI that predicts task priorities
- Emotion-aware task scheduling
- VR/AR task visualization
- Brain-computer interface integration
- Quantum-inspired task optimization
- Blockchain-based task verification
- Biometric stress-triggered breaks

---

#### Maximum Creativity (0.95-1.0): Experimental & Disruptive

**Characteristics:**
- Highly creative, experimental concepts
- Maximum innovation exploration
- "Blue sky" thinking
- Disruptive, paradigm-shifting ideas
- High risk, high potential reward

**When to use:**
- Breakthrough innovation sessions
- Future-focused R&D ideation
- Moonshot project brainstorming
- Startup pivot exploration
- Industry disruption thinking

**Example:**
```bash
model-chorus ideate "Revolutionary approaches to software development" --temperature 1.0 --num-ideas 10
```

**Expected ideas:**
- Quantum computing for code compilation
- Biological neural networks for testing
- Holographic collaborative coding
- AI-generated entire applications from thoughts
- Self-evolving codebases
- Consciousness-driven programming languages

---

### Temperature Selection Guide

| Your Goal | Recommended Temperature | Example Use Case |
|-----------|------------------------|------------------|
| Optimize existing system | 0.3-0.5 | "Ways to speed up our database queries" |
| Improve current features | 0.5-0.7 | "Enhancements for user onboarding flow" |
| Add new features | 0.6-0.8 | "New features for productivity app" |
| Innovate in product | 0.7-0.9 | "Innovative ways to handle notifications" |
| Disrupt industry | 0.9-1.0 | "Revolutionary approaches to task management" |

### Temperature Best Practices

**DO:**
-  Start with 0.7 (balanced) for most use cases
-  Lower temperature (0.4-0.6) for technical/infrastructure work
-  Raise temperature (0.8-1.0) for marketing/creative work
-  Match temperature to your risk tolerance
-  Use continuation to refine with lower temperature

**DON'T:**
- L Use 1.0 for production-critical systems
- L Use 0.3 for creative marketing campaigns
- L Expect practical ideas at 0.95+
- L Expect disruptive ideas at 0.4-

## Quantity Control: num_ideas Parameter

The num_ideas parameter controls how many distinct ideas are generated.

### Idea Count Guidelines

| Ideas | Category | Speed | Coverage | When to Use |
|-------|----------|-------|----------|-------------|
| **3-4** | Quick Exploration | Fast | Focused/Targeted | Time-constrained, simple problems, quick wins |
| **5-7** | Standard **(RECOMMENDED)** | Balanced | Good coverage | Most use cases, general brainstorming |
| **8-10** | Comprehensive | Thorough | Extensive/Deep | Complex problems, need multiple options |
| **10+** | Exhaustive | Slower | Maximum diversity | All possibilities, very comprehensive exploration |

### Idea Count Ranges Explained

### Idea Count Ranges Explained

#### Quick Exploration (3-4 ideas)

**Characteristics:**
- Fast generation time
- Focused, targeted ideas
- Quick brainstorming sessions
- Good for time constraints
- Concise output

**When to use:**
- Quick wins needed
- Time-constrained decisions
- Simple problem domains
- Focused exploration
- Initial reconnaissance

**Example:**
```bash
model-chorus ideate "Quick wins for page load time" --num-ideas 3 --temperature 0.6
```

**Output:** 3 high-quality, implementable ideas

---

#### Standard Brainstorming (5-7 ideas) **[RECOMMENDED]**

**Characteristics:**
- Balanced coverage
- Good diversity
- Reasonable generation time
- Adequate exploration
- **Best for most use cases**

**When to use:**
- General brainstorming sessions
- Feature planning
- Problem-solving
- Standard ideation needs
- Most use cases

**Example:**
```bash
model-chorus ideate "Feature ideas for mobile app" --num-ideas 6
```

**Output:** 6 distinct ideas with good diversity and depth

---

#### Comprehensive Exploration (8-10 ideas)

**Characteristics:**
- Thorough coverage
- High diversity
- Longer generation time
- Extensive exploration
- Multiple angles covered

**When to use:**
- Important decisions
- Strategic planning
- Competitive analysis
- Innovation sessions
- Thorough exploration needed

**Example:**
```bash
model-chorus ideate "Marketing strategies for product launch" --num-ideas 9 --temperature 0.8
```

**Output:** 9 diverse ideas covering various approaches and channels

---

#### Exhaustive Ideation (10+ ideas)

**Characteristics:**
- Maximum diversity
- All angles explored
- Longest generation time
- Very comprehensive
- Diminishing returns possible

**When to use:**
- Critical strategic decisions
- Major product pivots
- Comprehensive innovation workshops
- Exploring entire solution space
- Maximum idea diversity needed

**Example:**
```bash
model-chorus ideate "Complete redesign approaches for our platform" --num-ideas 15 --temperature 0.85
```

**Output:** 15+ ideas with extensive coverage (watch for diminishing quality)

---

### Idea Count Selection Guide

| Situation | Recommended Count | Rationale |
|-----------|------------------|-----------|
| Quick decision needed | 3-4 | Fast, focused |
| Standard brainstorming | 5-7 | Balanced coverage |
| Important feature planning | 6-8 | Good exploration |
| Strategic initiative | 8-10 | Comprehensive |
| Innovation workshop | 10-15 | Maximum diversity |
| Simple problem domain | 3-5 | Fewer options needed |
| Complex problem domain | 7-10 | More angles to explore |

### Quantity Best Practices

**DO:**
- Start with 5-7 for most cases
- Use 3-4 when time is constrained
- Use 8-10 for important decisions
- Consider problem complexity when choosing count
- Use continuation to refine specific ideas

**DON'T:**
- Generate 15+ ideas without good reason (diminishing returns)
- Use 3 ideas for complex strategic decisions
- Request more ideas than you'll actually evaluate
- Conflate quantity with quality

## Combining Temperature and num_ideas

The real power comes from combining these parameters strategically.

### Strategic Combinations

#### Quick Practical Wins (Low temp + Few ideas)
```bash
model-chorus ideate "Performance optimizations for API" --temperature 0.5 --num-ideas 4
```
**Result:** 4 practical, implementable optimization ideas

---

#### Standard Innovation (Medium temp + Medium ideas)
```bash
model-chorus ideate "New features for productivity tool" --temperature 0.7 --num-ideas 6
```
**Result:** 6 balanced ideas mixing practical and innovative

---

#### Creative Exploration (High temp + Many ideas)
```bash
model-chorus ideate "Disruptive marketing campaigns" --temperature 0.95 --num-ideas 10
```
**Result:** 10 bold, creative campaign concepts

---

#### Comprehensive Strategic Planning (Varied)
```bash
# Step 1: Generate many creative ideas
model-chorus ideate "Product differentiation strategies" --temperature 0.85 --num-ideas 12

# Step 2: Refine top 3 with practical details
model-chorus ideate "Expand on ideas #2, #5, and #9 with implementation details" --continue [session-id] --temperature 0.6 --num-ideas 3
```

---

### Combination Decision Matrix

| Goal | Temperature | num_ideas | Example |
|------|-------------|-----------|---------|
| Quick fixes | 0.4-0.5 | 3-4 | "Database query optimizations" |
| Feature additions | 0.6-0.7 | 5-7 | "New app features" |
| Innovation session | 0.8-0.9 | 8-10 | "Innovative product concepts" |
| Marketing creativity | 0.9-1.0 | 6-10 | "Bold marketing campaigns" |
| Strategic planning | 0.7-0.8 | 8-12 | "5-year product roadmap ideas" |
| Process improvement | 0.5-0.6 | 5-6 | "Team workflow enhancements" |

## CLI Usage

### Basic Invocation

**Simple ideation:**
```bash
model-chorus ideate "New features for task management app"
```

**Expected output:**
```
Generating ideas for: New features for task management app

--- Idea 1: Smart Task Prioritization ---
[Detailed description]

--- Idea 2: Collaborative Workspaces ---
[Detailed description]

--- Idea 3: AI-Powered Time Estimates ---
[Detailed description]

--- Idea 4: Focus Mode with Pomodoro ---
[Detailed description]

--- Idea 5: Task Dependencies Visualization ---
[Detailed description]

--- Summary & Recommendations ---
[Synthesis with top recommendations]

Session ID: ideate-thread-abc123
```

### With Creative Control

**Control both creativity and quantity:**
```bash
# Practical ideas (low creativity, few ideas)
model-chorus ideate "Database optimization strategies" --temperature 0.5 --num-ideas 4

# Balanced brainstorming (default)
model-chorus ideate "User engagement strategies" --temperature 0.7 --num-ideas 6

# Bold, creative ideas (high creativity, many ideas)
model-chorus ideate "Disruptive product features" --temperature 0.95 --num-ideas 10
```

### With Provider Selection

```bash
# Use Claude (excellent for creative ideation)
model-chorus ideate "Marketing campaign ideas" --provider claude

# Use Gemini (strong analytical creativity)
model-chorus ideate "Technical architecture ideas" --provider gemini

# Use Codex (best for code-related ideas)
model-chorus ideate "Code refactoring approaches" --provider codex
```

### With Constraints

```bash
# Add specific constraints
model-chorus ideate "Revenue streams for startup" --system "Must be implementable within 6 months, budget under $50k, target individual developers" --temperature 0.75 --num-ideas 7
```

### With Continuation

```bash
# Initial brainstorming
model-chorus ideate "Gamification features for learning platform" --num-ideas 7

# Drill into specific idea
model-chorus ideate "Expand on idea #2 with implementation details" --continue [session-id] --temperature 0.6 --num-ideas 1
```

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | Topic/area for idea generation |
| `num_ideas` | `int` | `5` | Number of ideas to generate (1-20) |
| `temperature` | `float` | `0.7` | Creativity level (0.0-1.0) |
| `max_tokens` | `int` | `None` | Maximum response length |
| `system_prompt` | `str` | `None` | Constraints, context, or criteria |
| `continuation_id` (aliases: `--continue`, `-c`, `--session-id`) | `str` | `None` | Thread ID to continue session (all aliases work identically) |
| `files` | `List[str]` | `[]` | File paths for context |

### Parameter Defaults Rationale

- **num_ideas=5**: Balanced coverage without overwhelming output
- **temperature=0.7**: Mix of practical and creative ideas
- **These defaults work well for 80% of use cases**

## Real-World Examples

### Example 1: Product Features (Balanced)

```bash
model-chorus ideate "New features for developer productivity tool" --num-ideas 7 --temperature 0.75 --system "Target: Individual developers and small teams. Must integrate with VS Code and GitHub."
```

**Expected output:**
- 7 balanced ideas mixing practical and innovative features
- Synthesis recommending top 3 with implementation priorities

### Example 2: Marketing Campaigns (High Creativity)

```bash
model-chorus ideate "Creative marketing campaigns for sustainable fashion brand" --num-ideas 8 --temperature 0.95
```

**Expected output:**
- 8 bold, unconventional marketing concepts
- High creativity with unique angles

### Example 3: Performance Optimization (Low Creativity)

```bash
model-chorus ideate "Ways to reduce API latency by 50%" --file docs/current_architecture.md --num-ideas 6 --temperature 0.5
```

**Expected output:**
- 6 practical, proven optimization techniques
- Focus on implementable solutions

## Common Brainstorming Patterns

### Pattern 1: Feature Brainstorming

**Use Case:** Generate new features for existing product

**Pattern:**
```bash
# Step 1: Generate broad feature ideas
model-chorus ideate "New features for [product name]" --num-ideas 8 --temperature 0.75 --system "Target users: [user persona]. Current features: [list]. Goals: [objectives]"

# Step 2: Refine top 3 ideas with details
model-chorus ideate "Expand on ideas #2, #5, and #7 with implementation details, user value, and effort estimates" --continue [session-id] --temperature 0.6 --num-ideas 3
```

**Why this works:**
- Broad exploration first (higher temp, more ideas)
- Refinement second (lower temp, fewer ideas, more detail)
- Provides both creativity and practicality

---

### Pattern 2: Problem-Solution Ideation

**Use Case:** Find creative solutions to specific problem

**Pattern:**
```bash
# Generate solutions with constraints
model-chorus ideate "Creative solutions for: [problem statement]" --num-ideas 7 --temperature 0.8 --system "Constraints: [list constraints]. Must be [criteria]"
```

**Example:**
```bash
model-chorus ideate "Creative solutions for: Reducing customer support tickets by 40%" --num-ideas 7 --temperature 0.8 --system "Constraints: Budget $50k, 3-month timeline, no additional headcount. Must improve user experience."
```

---

### Pattern 3: Tiered Exploration

**Use Case:** Explore ideas at multiple creativity levels

**Pattern:**
```bash
# Practical tier
model-chorus ideate "Proven, practical approaches for [topic]" --num-ideas 5 --temperature 0.5

# Innovative tier
model-chorus ideate "Innovative but feasible ideas for [topic]" --num-ideas 6 --temperature 0.75

# Bold tier
model-chorus ideate "Bold, unconventional ideas for [topic]" --num-ideas 5 --temperature 0.95
```

**Why this works:**
- Provides range of options from safe to risky
- Helps stakeholders choose risk tolerance
- Comprehensive coverage of solution space

---

### Pattern 4: Iterative Refinement

**Use Case:** Start broad, progressively narrow focus

**Pattern:**
```bash
# Round 1: Wide exploration
model-chorus ideate "Ways to improve [area]" --num-ideas 10 --temperature 0.8

# Round 2: Focus on promising direction
model-chorus ideate "Drill deeper into idea #4 (mention specific aspect). Generate variations and improvements." --continue [session-id] --num-ideas 6 --temperature 0.75

# Round 3: Implementation details
model-chorus ideate "For the most promising variation, provide: implementation steps, required resources, timeline, risks." --continue [session-id] --temperature 0.6 --num-ideas 1
```

---

### Pattern 5: Constrained Brainstorming

**Use Case:** Generate ideas within specific constraints

**Pattern:**
```bash
model-chorus ideate "[topic] ideas" --num-ideas 7 --temperature 0.7 --system "MUST meet these constraints:\n  - Timeline: [timeframe]\n  - Budget: [amount]\n  - Team size: [number]\n  - Tech stack: [technologies]\n  - Must integrate with: [systems]\n  - Success metrics: [KPIs]"
```

**Example:**
```bash
model-chorus ideate "Revenue stream ideas for developer tools startup" --num-ideas 7 --temperature 0.75 --system "MUST meet these constraints:\n  - Timeline: Launch within 6 months\n  - Budget: Under $75k\n  - Team: 3 developers\n  - Tech stack: Python, React\n  - Must integrate with: GitHub, VS Code\n  - Success metrics: $10k MRR in first quarter"
```

---

### Pattern 6: Competitive Analysis Ideation

**Use Case:** Differentiate from competitors

**Pattern:**
```bash
model-chorus ideate "Unique features that differentiate us from [competitors]" --num-ideas 8 --temperature 0.85 --system "Competitors: [list competitors and their key features]. Our strengths: [list]. Target users: [persona]" --file competitor_analysis.md --file our_product_strategy.md
```

---

## Feature Ideation Examples

### Example 1: Mobile App Features

**Scenario:** Task management mobile app needs new features

```bash
model-chorus ideate "Innovative features for personal task management mobile app" --num-ideas 8 --temperature 0.8 --system "Target users: Busy professionals, ages 25-45. Current features: Basic task lists, due dates, notifications. Goals: Increase daily active usage, improve task completion rates."
```

**Expected Ideas:**
1. **Smart Task Prioritization with AI**
   - Learns from your habits to suggest which tasks to do first
   - Considers deadlines, effort estimates, and your energy levels
   - Adapts to your productivity patterns

2. **Collaborative Workspaces**
   - Shared task lists with team members
   - Real-time sync and notifications
   - Role-based permissions and task assignment

3. **AI-Powered Time Estimates**
   - Predicts how long tasks will take based on similar past tasks
   - Helps with realistic scheduling
   - Improves over time with your data

4. **Focus Mode with Pomodoro Integration**
   - Blocks distracting apps during focus sessions
   - Integrates with calendar to find focus time
   - Tracks productivity streaks

5. **Task Dependencies Visualization**
   - Visual graph showing task relationships
   - Identifies blockers automatically
   - Suggests optimal task order

6. **Voice Task Entry with Context**
   - Add tasks via voice with natural language
   - Auto-extracts due dates, priorities from speech
   - Works with smart assistants

7. **Habit Tracking Integration**
   - Combines tasks with recurring habits
   - Streak tracking and rewards
   - Insights on consistency

8. **Energy-Level Based Scheduling**
   - User indicates energy levels throughout day
   - System suggests tasks matching current energy
   - Adapts schedule to your rhythm

**Synthesis:** Top 3 recommendations are Smart Prioritization (high value, feasible), Focus Mode (user requested feature), and Voice Entry (competitive differentiator).

---

### Example 2: Developer Tools Feature

**Scenario:** Code editor needs productivity-boosting features

```bash
model-chorus ideate "Innovative productivity features for code editor targeting individual developers" --num-ideas 10 --temperature 0.85 --system "Target: Individual developers and small teams. Must integrate with Git. Current features: Syntax highlighting, IntelliSense, debugging. Budget: $100k. Timeline: 6 months."
```

**Expected Ideas:**
1. **AI-Powered Code Completion**
2. **Automated Test Generation from Code**
3. **Smart Code Review Suggestions**
4. **Project Dependency Visualizer**
5. **Focus Mode with Distraction Blocking**
6. **Collaborative Debugging Sessions**
7. **Performance Profiling Integration**
8. **Automated Documentation Generation**
9. **Code Quality Metrics Dashboard**
10. **Context-Aware Refactoring Suggestions**

---

### Example 3: SaaS Platform Features

**Scenario:** B2B SaaS platform needs engagement features

```bash
model-chorus ideate "User engagement and retention features for B2B project management SaaS" --num-ideas 7 --temperature 0.75 --system "Target: Small to medium businesses (10-100 employees). Current MAU: 5k. Goal: Reduce churn from 8% to 4% monthly. Budget: $150k."
```

**Expected Ideas:**
1. **Personalized Onboarding Paths** - Adaptive tutorials based on role and experience
2. **Team Health Scorecard** - Metrics showing project health, team velocity, blockers
3. **Integration Marketplace** - Connect with 100+ business tools users already use
4. **Smart Notifications** - AI determines what's urgent vs what can wait
5. **Achievement System** - Gamify project milestones and team collaboration
6. **Weekly Insights Reports** - Auto-generated summaries of progress and trends
7. **In-App Video Tutorials** - Contextual help exactly when users need it

---

### Example 4: Marketing Campaign Ideas

**Scenario:** Sustainable fashion brand needs creative campaigns

```bash
model-chorus ideate "Bold, unconventional marketing campaigns for sustainable fashion brand" --num-ideas 8 --temperature 0.95 --system "Brand values: Sustainability, transparency, quality. Target: Millennials and Gen Z concerned about environment. Budget: $200k for 6-month campaign."
```

**Expected Ideas (High Creativity):**
1. **"Wear Your Values" Influencer Partnership**
   - Partner with micro-influencers who embody sustainability
   - Each piece tells the story of its environmental impact
   - Transparency report with every garment

2. **Virtual Fashion Shows in the Metaverse**
   - 3D fashion shows in virtual worlds
   - Digital clothing NFTs with physical redemption
   - Zero-waste fashion events

3. **Clothing Lifecycle Transparency App**
   - Scan QR code to see garment's full journey
   - Shows carbon footprint, water usage, worker conditions
   - Blockchain-verified authenticity

4. **Trade-In Program with Gamification**
   - Bring old clothes, get credits for new sustainable pieces
   - Leaderboards for most clothes recycled
   - Community challenges

5. **Sustainable Fashion Education Series**
   - TikTok/YouTube series on fast fashion impact
   - Partner with environmental organizations
   - User-generated content challenges

6. **Community-Driven Design Contests**
   - Users submit sustainable design ideas
   - Winning designs produced in limited runs
   - Designers get portion of sales

7. **"True Cost" Calculator**
   - Tool showing environmental cost of fast fashion vs sustainable
   - Shareable infographics for social media
   - Viral potential through shock value

8. **Pop-Up Repair Cafés**
   - Free clothing repair events in major cities
   - Teaches mending skills
   - Builds community around sustainability

---

## Technical Contract

### Parameters

**Required:**
- `prompt` (string): The topic or problem for which to generate creative ideas

**Optional:**
- `--provider, -p` (string): AI provider to use for idea generation - Valid values: `claude`, `gemini`, `codex`, `cursor-agent` - Default: `claude`
- `--num-ideas, -n` (integer): Number of distinct ideas to generate - Range: 1-20 - Default: 5 - Recommended: 5-7 for balanced exploration
- `--continue` / `-c` / `--session-id` (string): Session ID to continue previous ideation session - All aliases work identically - Format: `ideate-thread-{uuid}` - Maintains conversation history for iterative refinement
- `--file, -f` (string, repeatable): File paths to include as context for ideation - Can be specified multiple times - Files must exist before execution - Helps ground ideas in existing constraints or context
- `--system` (string): Additional system prompt to set constraints or criteria - Useful for specifying budget, timeline, technical constraints, or target audience - Applied to all generated ideas
- `--temperature, -t` (float): Creativity level for idea generation - Range: 0.0-1.0 - Default: 0.7 - Lower (0.3-0.5) for practical/proven ideas - Medium (0.6-0.8) for balanced innovation - Higher (0.8-1.0) for bold/disruptive ideas
- `--max-tokens` (integer): Maximum response length in tokens - Provider-specific limits apply - Affects detail level of generated ideas
- `--output, -o` (string): Path to save JSON output file - Creates or overwrites file at specified path
- `--verbose, -v` (boolean): Enable detailed execution information - Default: false - Shows provider details and timing

### Return Format

The IDEATE workflow returns a JSON object with the following structure:

```json
{
  "result": "=== Idea 1: [Title] ===\n[Description]\n\n=== Idea 2: [Title] ===\n[Description]\n\n... [num_ideas total] ...\n\n=== Summary & Recommendations ===\n[Synthesis of top ideas]",
  "session_id": "ideate-thread-abc-123-def-456",
  "metadata": {
    "provider": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "num_ideas_requested": 5,
    "num_ideas_generated": 5,
    "temperature": 0.7,
    "prompt_tokens": 200,
    "completion_tokens": 800,
    "total_tokens": 1000,
    "timestamp": "2025-11-07T10:30:00Z"
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | All generated ideas with clear headers, followed by a summary and recommendations section that synthesizes the ideas |
| `session_id` | string | Session ID for continuing this ideation session (format: `ideate-thread-{uuid}`) |
| `metadata.provider` | string | The AI provider used for idea generation |
| `metadata.model` | string | Specific model version used by the provider |
| `metadata.num_ideas_requested` | integer | Number of ideas requested via `--num-ideas` parameter |
| `metadata.num_ideas_generated` | integer | Number of ideas actually generated (typically matches requested) |
| `metadata.temperature` | float | Temperature setting used for this ideation (0.0-1.0) - Higher values produce more creative/bold ideas |
| `metadata.prompt_tokens` | integer | Number of tokens in the input (prompt + context + history) |
| `metadata.completion_tokens` | integer | Number of tokens in the generated ideas and synthesis |
| `metadata.total_tokens` | integer | Total tokens consumed (prompt_tokens + completion_tokens) |
| `metadata.timestamp` | string | ISO 8601 timestamp of when the ideation was completed |

**Usage Notes:**
- Save the `session_id` to continue ideation with refinements, expansions, or follow-up questions
- The `result` includes both individual ideas and a synthesis/recommendation section
- Token costs scale with `num_ideas` - more ideas means higher token usage
- Temperature directly impacts idea creativity: 0.4 (practical) → 0.7 (balanced) → 0.9 (bold)
- The synthesis section at the end provides a curated view of the strongest ideas
- Use `--system` to provide constraints that shape all generated ideas consistently

## Best Practices for Brainstorming

### 1. Start Broad, Then Narrow

**Pattern:**
```bash
# ✅ Good: Start with many ideas at medium-high creativity
model-chorus ideate "Product improvement ideas" --num-ideas 10 --temperature 0.8

# Then refine best ideas with more detail
model-chorus ideate "Expand on ideas #2 and #7" --continue [session-id] --temperature 0.6
```

**Avoid:**
```bash
# ❌ Less effective: Going straight to detailed implementation
model-chorus ideate "Detailed implementation plan for specific feature" --temperature 0.5
```

---

### 2. Match Parameters to Purpose

| Purpose | Temperature | num_ideas | Example |
|---------|-------------|-----------|---------|
| Quick wins | 0.4-0.6 | 3-5 | Infrastructure improvements |
| Standard features | 0.6-0.8 | 5-7 | Product roadmap planning |
| Innovation session | 0.8-0.95 | 8-12 | Competitive differentiation |
| Breakthrough thinking | 0.95-1.0 | 10-15 | Industry disruption ideas |

---

### 3. Use Constraints for Focus

**With constraints (focused, actionable):**
```bash
model-chorus ideate "Feature ideas" --system "Timeline: 3 months. Budget: $50k. Team: 2 developers. Tech: React, Node.js"
```

**Without constraints (too broad):**
```bash
model-chorus ideate "Feature ideas"
```

---

### 4. Save and Document Ideas

**Pattern:**
```bash
# Generate ideas and save output
model-chorus ideate "Revenue stream ideas" --num-ideas 8 > brainstorm_2025-11-07.txt

# Reference in follow-up
model-chorus ideate "Combine ideas #3 and #6 from previous session" --continue [session-id]
```

---

### 5. Combine with Other Workflows

**IDEATE → ARGUMENT Pattern:**
```bash
# Step 1: Generate ideas with IDEATE
model-chorus ideate "API architecture approaches" --num-ideas 6 --temperature 0.75

# Step 2: Evaluate top idea with ARGUMENT
model-chorus argument "GraphQL approach (idea #3) is the best choice for our use case"
```

**IDEATE → CONSENSUS Pattern:**
```bash
# Step 1: Generate ideas with single model
model-chorus ideate "Database migration strategies" --num-ideas 5

# Step 2: Validate top approach with multiple models
model-chorus consensus "Evaluate zero-downtime migration approach" -p claude -p gemini -s synthesize
```

---

## Troubleshooting Brainstorming Sessions

### Issue: Ideas Too Similar or Repetitive

**Symptoms:** All ideas feel like variations of the same concept

**Solutions:**
```bash
# Increase temperature
model-chorus ideate "..." --temperature 0.9  # Was: 0.6

# Request explicit diversity
model-chorus ideate "..." --system "Generate highly diverse ideas exploring completely different approaches and angles"

# Generate more ideas (more attempts = more variety)
model-chorus ideate "..." --num-ideas 12  # Was: 5
```

---

### Issue: Ideas Too Generic or Vague

**Symptoms:** Ideas lack specificity or actionable details

**Solutions:**
```bash
# Provide more context
model-chorus ideate "Feature ideas" --system "Context: Healthcare app for elderly users, iOS only, accessibility focus, 10k users" --file user_research.md

# Request specific details in prompt
model-chorus ideate "Detailed feature ideas with implementation approach, user value, and success metrics"

# Use continuation to refine
model-chorus ideate "Expand idea #4 with specific user stories, technical requirements, and UI mockup descriptions" --continue [session-id] --temperature 0.6
```

---

### Issue: Ideas Too Impractical

**Symptoms:** Creative but not feasible with current resources

**Solutions:**
```bash
# Lower temperature
model-chorus ideate "..." --temperature 0.6  # Was: 0.9

# Add practical constraints
model-chorus ideate "..." --system "Must be implementable within 3 months with 2 developers and $30k budget. Use existing tech stack."

# Request explicitly practical ideas
model-chorus ideate "Practical, immediately implementable ideas for..."
```

---

### Issue: Not Enough Creative Ideas

**Symptoms:** All ideas are safe, incremental improvements

**Solutions:**
```bash
# Increase temperature significantly
model-chorus ideate "..." --temperature 0.95  # Was: 0.6

# Request bold thinking
model-chorus ideate "Bold, unconventional, disruptive ideas for..." --temperature 0.95

# Remove constraints temporarily
model-chorus ideate "Blue sky thinking for ... (no constraints, maximum creativity)"
```

---

### Issue: Too Many Ideas to Evaluate

**Symptoms:** Overwhelmed with options, can't decide

**Solutions:**
```bash
# Use continuation to synthesize and prioritize
model-chorus ideate "From the 12 ideas generated, identify the top 3 based on: user value, implementation feasibility, and competitive advantage. Explain reasoning." --continue [session-id] --num-ideas 1 --temperature 0.6

# Or generate fewer ideas initially
model-chorus ideate "..." --num-ideas 5  # Was: 12

# Use ARGUMENT to evaluate top ideas
model-chorus argument "Idea #3 (specific feature) should be our top priority"
```

---

## Comparison with Other Workflows

### IDEATE vs CHAT

| Aspect | IDEATE | CHAT |
|--------|--------|------|
| **Purpose** | Generate multiple creative ideas | Simple conversation |
| **Output** | Multiple distinct ideas + synthesis | Single response |
| **Creativity Control** | Temperature + num_ideas | Temperature only |
| **Use when** | Need multiple options | Need single answer |

### IDEATE vs ARGUMENT

| Aspect | IDEATE | ARGUMENT |
|--------|--------|----------|
| **Purpose** | Generate new ideas | Analyze existing arguments |
| **Process** | Creative generation | Dialectical reasoning |
| **Creativity** | Configurable via temperature | Fixed analytical approach |
| **Use when** | Need creative options | Need to evaluate a claim |

## Progress Reporting

The IDEATE workflow automatically displays progress updates to stderr as it executes. You will see messages like:

```
Starting ideate workflow (estimated: 20-45s)...
✓ ideate workflow complete
```

**Important for Claude Code:** Progress updates are emitted automatically - do NOT use `BashOutput` to poll for progress. Simply invoke the command and wait for completion. All progress information streams automatically to stderr without interfering with stdout.

## See Also

**Related Workflows:**
- **CHAT** - Simple conversation for quick questions
- **ARGUMENT** - Evaluate generated ideas dialectically
- **CONSENSUS** - Get multi-model perspectives on ideas
- **THINKDEEP** - Systematic investigation with evidence
