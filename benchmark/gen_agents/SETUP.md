# Generative Agents Baseline — Setup

The Generative Agents method depends on [MemEngine](https://github.com/nuster1128/MemEngine), which is not bundled in this repository because its license could not be verified at release time.

## Installation

Clone the upstream MemEngine repository into this directory:

```bash
cd benchmark/gen_agents
git clone https://github.com/nuster1128/MemEngine.git upstream
```

The expected layout after cloning:

```
benchmark/gen_agents/
├── SETUP.md
└── upstream/
    └── memengine/
        ├── __init__.py
        ├── config/
        ├── function/
        ├── memory/
        └── ...
```

Once cloned, the `gen_agents` method will work automatically — no further configuration needed.
