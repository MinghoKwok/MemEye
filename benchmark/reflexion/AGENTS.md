# AGENTS.md

## Scope
This file contains Reflexion-specific instructions for adapting Mem-Gallery’s built-in Reflexion core to this benchmark’s Mem-Gallery-style dataset.

Follow this file for all work in this subtree.

## Project Goal
Implement Reflexion on this project’s dataset, which follows the same general format as Mem-Gallery.

The target is a faithful benchmark adaptation of Mem-Gallery’s built-in Reflexion core, integrated into this repo’s local benchmark framework.

Do not adopt Mem-Gallery’s full benchmark scaffold as the main implementation base.

## Source of Truth

### Mem-Gallery Reflexion core is the source of truth for:
- Reflexion memory behavior
- reflection-related storage / recall / optimization behavior
- Reflexion-specific internal logic in the downloaded `memengine` code

Use the downloaded local Reflexion core under this subtree as the primary reference for core Reflexion behavior.

### This repo’s local benchmark code is the source of truth for:
- dataset parsing
- dialogue-to-memory conversion
- how image captions are injected into text-only memory
- how question images are represented at recall time
- the final QA loop structure
- prompt usage
- config style
- logging style
- output and runner conventions

Use:
- `Benchmark_Pipeline/benchmark/a_mem.py`
- `Benchmark_Pipeline/benchmark/methods.py`
- `Benchmark_Pipeline/benchmark/runner.py`
- local prompts already used in this repo
- local config conventions already used in this repo

Notes:
- In this setup, the downloaded Mem-Gallery code should provide the Reflexion core only.
- Do not rebuild Mem-Gallery’s full benchmark framework locally.
- There is no requirement to use Mem-Gallery’s `run_bench.py` as the main scaffold.
- Use `Benchmark_Pipeline/benchmark/a_mem.py` as the main local adapter reference for structure, benchmark integration pattern, and logging style.
- Reuse `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` if they are useful.
- Do not duplicate existing benchmark utilities if they already solve the needed problem cleanly.

## Non-Negotiable Rules
- Keep the adaptation thin.
- Build a local benchmark adapter around the downloaded Reflexion core.
- Do not rewrite Reflexion into MemEngine form beyond what has already been downloaded.
- Do not reimplement unrelated Mem-Gallery baselines.
- Do not redesign Reflexion core behavior.
- Do not silently substitute models, embedders, APIs, or retrieval components.
- If a required API, model, or config is missing, stop and ask.

## Critical Rule: Treat Reflexion as Text-Only for This Adaptation
For this benchmark adaptation, treat Reflexion as a text-only memory system unless the user explicitly asks for a new multimodal Reflexion extension.

Therefore:
- Do not add a native visual memory module.
- Do not add a separate visual embedding store.
- Do not add cross-modal retrieval.
- Do not redesign Reflexion into a new multimodal memory architecture.

For this benchmark, images must be incorporated using the existing text-only adaptation strategy already used in this repo:
- use the provided image caption from the dataset
- append image information into the stored text
- append question-image caption text into the recall query when needed

This is a benchmark adaptation layer. Do not present Reflexion itself as natively multimodal unless explicitly implementing a new multimodal extension.

## How to Represent Images for Reflexion
When a dialogue turn contains an image, store it as text in the adapter layer, not as a native image memory object.

Append:
- `image_caption`

Recommended stored-text pattern:

    <dialogue text>
    image:
    image_caption: <caption>

At recall time, if a question includes an image, append the question image caption to the recall query in the same textual style.

Do this in the adapter layer. Do not push this logic deep into the downloaded Reflexion core unless there is no clean wrapper alternative.

## Wrapper Boundary
The wrapper is responsible for:
- loading this repo’s Mem-Gallery-style dialogue data
- converting each dialogue turn into the text form expected by Reflexion
- injecting image captions into stored text
- injecting question-image captions into recall queries
- passing retrieved Reflexion context into the local benchmark QA loop
- reusing local benchmark utilities when useful
- saving outputs in the local benchmark format

The downloaded Reflexion core remains responsible for:
- core Reflexion memory behavior
- recall / store / optimize logic already implemented in the downloaded Mem-Gallery core
- Reflexion-specific data flow inside that core

## Code Organization
Prefer clean separation between:
- downloaded Reflexion core code
- local benchmark adapter code
- local experiment utilities

Preferred approach:
- keep the downloaded Reflexion core minimally modified
- place dataset parsing and benchmark glue outside Reflexion core files
- use wrapper or adapter code whenever possible
- reuse `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` where practical
- use `Benchmark_Pipeline/benchmark/a_mem.py` as the main local reference for the expected adapter pattern
- make it easy to trace which files are downloaded core code and which are local additions

A good outcome is a local adapter file such as:
- `Benchmark_Pipeline/benchmark/reflexion.py`

## Dataset
Dataset root:
- `Benchmark_Pipeline/data`

Current priority files:
- `Benchmark_Pipeline/data/dialog/Home_Renovation_Interior_Design.json`
- `Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant.json`

Image root:
- `Benchmark_Pipeline/data/image`

Assume Mem-Gallery-style format unless local files show otherwise.

## Prompts
Use this repo’s existing local prompt conventions.

Do not import Mem-Gallery prompt files as the main prompt system unless explicitly needed.
Only add new prompts if necessary.
Do not silently replace existing local benchmark prompts.

## Models
- Always prefer the model choices already used by this repo’s local benchmark conventions unless the user instructs otherwise.
- If a required API or model is unavailable, ask the user.
- Do not switch to alternative models without notification.

## Outputs
Write outputs and run artifacts to:
- `Benchmark_Pipeline/output/{task_name}`
- `Benchmark_Pipeline/runs/{task_name}`

Save enough information to debug and reproduce runs, including when available:
- config used
- dataset path
- model names
- prompt names
- stored memory text
- recall query text
- retrieved memories
- final predictions
- run logs
- error logs

## Modification Policy
Reuse the downloaded Reflexion core whenever possible.

Before making substantive changes to Reflexion core behavior, ask for permission.

Substantive changes include:
- changing memory schema
- changing recall logic
- changing store logic
- changing optimization logic
- changing reflection behavior
- changing embeddings
- changing core prompts inside the downloaded Reflexion core
- changing Reflexion-specific module interfaces

When in doubt:
- keep Reflexion core unchanged
- implement the behavior in the adapter layer
- ask before making deeper changes

## Practical Implementation Priority
Implement in this order:
1. inspect local dataset files directly
2. inspect the downloaded Reflexion core files in this subtree
3. inspect `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` for reusable utilities
4. inspect `Benchmark_Pipeline/benchmark/a_mem.py` for local adapter structure reference
5. load Mem-Gallery-format dialogue data
6. convert each dialogue turn into Reflexion-storable text
7. inject image captions into stored text
8. inject question-image captions into recall queries
9. connect Reflexion retrieval to the local benchmark QA loop
10. save outputs in the project’s output structure
11. only then do cleanup or refactoring

## Files to Read First
Read these files before coding:
1. downloaded Reflexion core files in this subtree
2. `Benchmark_Pipeline/benchmark/methods.py`
3. `Benchmark_Pipeline/benchmark/runner.py`
4. `Benchmark_Pipeline/benchmark/a_mem.py`
5. `Benchmark_Pipeline/data/dialog/Home_Renovation_Interior_Design.json`
6. `Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant.json`

## Human Approval Required
Do not do any of the following without explicit permission:
- change downloaded Reflexion core behavior
- add a new multimodal retrieval architecture
- install major new dependencies
- substantially restructure the repository
- switch models or APIs from the local benchmark’s intended choice
- push to GitHub
- push to Hugging Face