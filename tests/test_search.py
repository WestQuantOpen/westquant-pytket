from westquant_core import Evaluation
from westquant_pytket.search import PytketSequentialSearch


class FakeCircuit:
    n_qubits = 3
    n_bits = 0
    def get_commands(self): return []
    def depth(self): return 0


class FakeCompiler:
    def compile(self, circuit, prefix):
        score = sum({"none": 5, "redundancies": 3, "peephole": 1, "synthesise": 2, "naive": 2, "graph": 1, "routing": 1, "aas": 2, "tket": 1}.get(a.name, 0) for a in prefix)
        out = FakeCircuit()
        out._score = score
        return out


class FakeVerifier:
    def verify(self, original, candidate): return {"equivalence": "exact", "verified": True}


def test_sequential_runs(monkeypatch):
    import westquant_pytket.search as mod
    monkeypatch.setattr(mod, "circuit_metrics", lambda c: {"two_qubit_gates": c._score, "depth": c._score, "n_gates": c._score})
    result = PytketSequentialSearch(beam_width=2, compiler=FakeCompiler(), verifier=FakeVerifier()).run(FakeCircuit())
    assert result.best is not None
    assert len(result.records(framework="pytket")) > 0
