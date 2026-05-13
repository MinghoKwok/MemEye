# MemEye Benchmark Methods

13 methods across 4 categories. Each method is evaluated in both MCQ and Open modes.

## Method Taxonomy

### Full Context (2)

No compression — the entire dialogue history is fed to the LLM.

| Method | Config | Modality | Description |
|--------|--------|----------|-------------|
| FC-Text | `full_context_text_only` | Text | Full history with image captions replacing images |
| FC-Multimodal | `full_context_multimodal` | Visual | Full history with native images |

### Retrieval / RAG (2)

Semantic retrieval selects top-k relevant rounds from history.

| Method | Config | Modality | Description |
|--------|--------|----------|-------------|
| SRAG-Text | `semantic_rag_text_only` | Text | Dense text retrieval (all-MiniLM-L6-v2) |
| SRAG-Multimodal | `semantic_rag_multimodal` | Visual | Dense text + cross-modal image retrieval (SigLIP2) |

### Summarization (2)

LLM compresses dialogue history into summaries before answering.

| Method | Config | Modality | Description |
|--------|--------|----------|-------------|
| SimpleMem | `simplemem` | Text | Omni-SimpleMem: intent-aware summarization (text-only) |
| SimpleMem-MM | `simplemem_multimodal` | Visual | Omni-SimpleMem with native images |

### Agentic Memory (7)

Autonomous agents that construct, update, and query their own memory.

| Method | Config | Modality | Memory Mechanism |
|--------|--------|----------|------------------|
| A-MEM | `a_mem` | Text | Autonomous memory management (add/delete/update) |
| Reflexion | `reflexion` | Text | Self-reflection loop |
| Gen. Agents | `gen_agents` | Text | Reflection + planning (Park et al. 2023) |
| MemoryOS | `memoryos` | Text | Hierarchical memory scheduling |
| M2A | `m2a` | Visual | Multimodal ReAct loop + semantic memory bank (SigLIP2) |
| MMA | `mma` | Visual | Confidence-weighted multimodal retrieval (SigLIP v1 so400m) |
| MIRIX | `mirix` | Visual | Multi-layer memory agent system |

## Multimodal Embedding Models

| Method | Embedder | Dimensions |
|--------|----------|------------|
| M2A, SRAG-MM | google/siglip2-base-patch16-384 | 768 |
| MMA | google/siglip-so400m-patch14-384 | 1152 |
| MIRIX | all-MiniLM-L6-v2 (local) | 384 |
| Gen. Agents | all-MiniLM-L6-v2 (local) | 384 |
| SimpleMem-MM, FC-MM | Native images (no embedding) | — |

## Oracle Methods (Ablation)

| Method | Config | Description |
|--------|--------|-------------|
| Clue Only (T) | `clue_only_context` | Only clue rounds, captions replace images |
| Clue Only (V) | `clue_only_context_multimodal` | Only clue rounds, with native images |
| Session Only (T) | `target_session_context_text_only` | All rounds from clue sessions, text only |
| Session Only (V) | `target_session_context` | All rounds from clue sessions, with images |

## Caption-Proof Gap Pairs

Three method pairs provide Text vs Visual modality comparison:

| Text-only | Visual | Category |
|-----------|--------|----------|
| FC-Text | FC-Multimodal | Full Context |
| SRAG-Text | SRAG-Multimodal | Retrieval |
| SimpleMem | SimpleMem-MM | Summarization |
