"""The reader panel, in table order. Standard library only, so the analysis scripts that import
it also run under a bare python3.

Four families served as generators and jurors during construction; three took no part in it and
act as controls. Every script that groups readers into participating families and controls takes
the groups from here, so the tables, the paired tests and the scale fits cannot disagree about
who is in the pool. The reader YAML's `pool:` flags must say the same.
"""

PARTICIPATING = (("qwen", "Qwen"), ("llama", "Llama"), ("gemma", "Gemma"), ("olmo31", "OLMo 3.1"))
CONTROL = (("glm", "GLM"), ("phi", "Phi-4"), ("mistral", "Mistral"))
PANEL = PARTICIPATING + CONTROL

PANEL_NAMES = tuple(key for key, _ in PANEL)
POOL_NAMES = frozenset(key for key, _ in PARTICIPATING)
DISPLAY = dict(PANEL)
