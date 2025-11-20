#!/usr/bin/env python3
"""
Script to fix ThinkDeepWorkflow.run() calls to use the explicit API.

Converts old style:
    workflow.run(prompt="...", ...)

To new style:
    workflow.run(
        step="...",
        step_number=1,
        total_steps=1,
        next_step_required=False,
        findings="...",
        ...
    )
"""

import re
from pathlib import Path


def fix_simple_call(match):
    """Fix a simple single-line prompt call."""
    indent = match.group(1)
    prompt_text = match.group(2)
    rest = match.group(3) if match.group(3) else ""

    # Determine if there are additional parameters
    has_continuation = "continuation_id" in rest

    # Build the new call
    new_call = f"""{indent}result = await workflow.run(
{indent}    step={prompt_text},
{indent}    step_number=1,
{indent}    total_steps=1,
{indent}    next_step_required=False,
{indent}    findings="Initial investigation",{rest}
{indent})"""

    return new_call


def fix_multiline_call(content):
    """Fix multi-line workflow.run calls."""
    # Pattern for multi-line calls starting with prompt=
    pattern = (
        r'(\s+)(result\d*) = await workflow\.run\(\s*\n\s+prompt="([^"]+)"([^)]*)\)'
    )

    def replace_multiline(match):
        indent = match.group(1)
        result_var = match.group(2)
        prompt_text = match.group(3)
        rest = match.group(4)

        # Check if this is a continuation
        has_continuation = "continuation_id" in rest
        step_num = 1
        if has_continuation:
            # Try to extract step number from result variable (e.g., result2 -> step 2)
            if result_var[-1].isdigit():
                step_num = int(result_var[-1])

        next_step = has_continuation or "files" in rest

        # Build new call
        new_call = f"""{indent}{result_var} = await workflow.run(
{indent}    step="{prompt_text}",
{indent}    step_number={step_num},
{indent}    total_steps={step_num if not next_step else step_num + 1},
{indent}    next_step_required={str(next_step)},
{indent}    findings="Investigation step {step_num}",{rest}
{indent})"""

        return new_call

    return re.sub(pattern, replace_multiline, content, flags=re.MULTILINE | re.DOTALL)


def process_file(filepath):
    """Process a single test file."""
    print(f"Processing {filepath}...")

    with open(filepath) as f:
        content = f.read()

    original = content

    # Fix single-line calls
    content = re.sub(
        r'(\s+)result\d* = await workflow\.run\(prompt="([^"]+)"([^)]*)\)',
        fix_simple_call,
        content,
    )

    # Fix multi-line calls
    content = fix_multiline_call(content)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False


def main():
    """Main function."""
    files_to_fix = [
        "tests/test_thinkdeep_workflow.py",
        "tests/test_thinkdeep_complex.py",
        "tests/test_thinkdeep_expert_validation.py",
        "tests/test_workflow_integration_chaining.py",
        "examples/thinkdeep_example.py",
    ]

    updated = 0
    for filepath in files_to_fix:
        path = Path(filepath)
        if path.exists():
            if process_file(path):
                updated += 1
        else:
            print(f"  ✗ File not found: {filepath}")

    print(f"\nUpdated {updated} files")


if __name__ == "__main__":
    main()
