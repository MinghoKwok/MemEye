# MemEye Prompt Templates

This directory stores prompt templates for MemEye benchmark runs.

The purpose of these templates is not to hard-code one dataset. The purpose is to capture the benchmark-wide instruction patterns that repeatedly matter across MemEye tasks:

- answer only from the provided multimodal history
- prefer the most recent information when the history changes
- do not invent hidden labels or metadata
- obey the required answer format exactly
- keep answers concise unless the task explicitly asks for explanation

## Files

- `sys_prompt.txt`: default benchmark-wide system prompt
- `identity_prompt.txt`: identity and instance-binding tasks
- `spatial_prompt.txt`: spatial grounding and layout tasks
- `state_update_prompt.txt`: evolving-state and contradiction-resolution tasks
- `pattern_prompt.txt`: inter-image visual pattern induction tasks
- `refusal_prompt.txt`: unsupported or no-evidence answer refusal tasks

## Usage

These files are intended as reusable prompt building blocks for future routing and benchmark orchestration changes.

They document the current MemEye prompt philosophy even when a specific runner does not yet load every task-family template directly.
