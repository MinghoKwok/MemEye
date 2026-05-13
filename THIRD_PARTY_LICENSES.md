# Third-Party Vendored Code

MemEye includes small vendored baseline implementations under `benchmark/`.
Those subtrees keep their own license files where the license could be
verified.

| Vendored path | Upstream/source | License status |
| --- | --- | --- |
| `benchmark/a-mem/A-mem/` | A-MEM, https://github.com/WujiangXu/A-mem | MIT; see `benchmark/a-mem/A-mem/LICENSE` |
| `benchmark/evermemos/upstream/` | EverMemOS, https://github.com/EverMind-AI/EverMemOS | Apache-2.0; see `benchmark/evermemos/upstream/LICENSE` |
| `benchmark/memgpt/upstream/` | MemGPT/Letta, https://github.com/letta-ai/letta | Apache-2.0; see `benchmark/memgpt/upstream/LICENSE` |
| `benchmark/memoryos/MemoryOS/` | MemoryOS vendored code | Existing upstream license retained at `benchmark/memoryos/MemoryOS/LICENSE` |
| `benchmark/mirix/upstream/` | MIRIX, https://github.com/Mirix-AI/MIRIX | Apache-2.0; see `benchmark/mirix/upstream/LICENSE` |
| `benchmark/simplemem/upstream/` | SimpleMem vendored code | Existing upstream licenses retained under `benchmark/simplemem/upstream/` |
| `benchmark/gen_agents/upstream/` | MemEngine (Generative Agents), https://github.com/nuster1128/MemEngine | Not bundled — user must clone upstream; see `benchmark/gen_agents/SETUP.md` |

The top-level MemEye license does not override third-party license terms.
