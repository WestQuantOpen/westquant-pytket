# westquant-pytket

Sequential representation search over pytket compiler choices.

Current alpha searches the ordered decision chain:

```text
optimization -> placement -> routing -> rebase
```

Built-in action families include `RemoveRedundancies`,
`FullPeepholeOptimise`, `SynthesiseTket`, naive/graph placement,
`RoutingPass`, `AASRouting` and `RebaseTket` through pytket's public pass API.

Every rollout produces a `wqt-policy-v0.1` record. Compilation failures are
retained. Small circuits attempt exact unitary verification; larger or
unsupported circuits are explicitly `unknown` rather than silently trusted.

## Native smoke

```bash
pip install -e ../westquant-core -e .[test]
pytest -q
python integration/smoke.py
```

## CLI

```bash
westquant-pytket-search --qasm circuit.qasm --edges 0-1,1-2,2-3 --beam-width 3
```
