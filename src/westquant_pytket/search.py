from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from westquant_core import Action, DeterministicBeamSearch, Evaluation, Objective
from .adapter import circuit_metrics


STAGES = ("optimization", "placement", "routing", "rebase")


@dataclass(frozen=True)
class PytketSearchSpace:
    optimization: tuple[str, ...] = ("none", "redundancies", "peephole", "synthesise")
    placement: tuple[str, ...] = ("none", "naive", "graph")
    routing: tuple[str, ...] = ("none", "routing", "aas")
    rebase: tuple[str, ...] = ("none", "tket", "synthesise")

    def choices(self, stage: str) -> tuple[str, ...]:
        return tuple(getattr(self, stage))


class PytketCompiler:
    def __init__(self, *, architecture_edges: Sequence[tuple[int, int]] | None = None) -> None:
        self.architecture_edges = tuple(architecture_edges or ())

    def compile(self, circuit: Any, prefix: Sequence[Action]) -> Any:
        from pytket.architecture import Architecture
        from pytket.passes import (
            AASRouting,
            FullPeepholeOptimise,
            NaivePlacementPass,
            PlacementPass,
            RebaseTket,
            RemoveRedundancies,
            RoutingPass,
            SynthesiseTket,
        )
        from pytket.placement import GraphPlacement

        out = circuit.copy()
        arc = Architecture(list(self.architecture_edges)) if self.architecture_edges else None
        selected = {a.stage: a.name for a in prefix}

        name = selected.get("optimization", "none")
        if name == "redundancies":
            RemoveRedundancies().apply(out)
        elif name == "peephole":
            FullPeepholeOptimise().apply(out)
        elif name == "synthesise":
            SynthesiseTket().apply(out)

        name = selected.get("placement", "none")
        if name != "none":
            if arc is None:
                raise ValueError("placement requires architecture_edges")
            if name == "naive":
                NaivePlacementPass(arc).apply(out)
            elif name == "graph":
                PlacementPass(GraphPlacement(arc)).apply(out)

        name = selected.get("routing", "none")
        if name != "none":
            if arc is None:
                raise ValueError("routing requires architecture_edges")
            if name == "routing":
                RoutingPass(arc).apply(out)
            elif name == "aas":
                AASRouting(arc).apply(out)

        name = selected.get("rebase", "none")
        if name == "tket":
            RebaseTket().apply(out)
        elif name == "synthesise":
            SynthesiseTket().apply(out)
        return out


class PytketVerifier:
    def __init__(self, *, max_unitary_qubits: int = 7) -> None:
        self.max_unitary_qubits = max_unitary_qubits

    def verify(self, original: Any, candidate: Any) -> dict[str, Any]:
        n = int(getattr(original, "n_qubits", 0))
        if n != int(getattr(candidate, "n_qubits", -1)):
            return {"equivalence": "unknown", "verified": False, "reason": "width_changed"}
        if n > self.max_unitary_qubits:
            return {"equivalence": "unknown", "verified": False, "reason": "unitary_limit"}
        try:
            import numpy as np
            u = np.asarray(original.get_unitary(), dtype=complex)
            v = np.asarray(candidate.get_unitary(), dtype=complex)
            overlap = np.vdot(u.ravel(), v.ravel())
            if abs(overlap) < 1e-15:
                ok = bool(np.allclose(u, v, atol=1e-8, rtol=1e-8))
            else:
                phase = overlap / abs(overlap)
                ok = bool(np.allclose(u, phase * v, atol=1e-8, rtol=1e-8))
            return {"equivalence": "exact" if ok else "invalid", "verified": True, "method": "get_unitary"}
        except Exception as exc:
            return {"equivalence": "unknown", "verified": False, "reason": type(exc).__name__, "message": str(exc)}


class PytketSequentialSearch:
    def __init__(self, *, search_space: PytketSearchSpace | None = None, beam_width: int = 3,
                 compiler: Any | None = None, verifier: Any | None = None) -> None:
        self.search_space = search_space or PytketSearchSpace()
        self.compiler = compiler or PytketCompiler()
        self.verifier = verifier or PytketVerifier()
        self.engine = DeterministicBeamSearch(
            stages=STAGES,
            beam_width=beam_width,
            objectives=(Objective("two_qubit_gates"), Objective("depth"), Objective("n_gates")),
        )

    def run(self, circuit: Any, *, challenge_id: str = "pytket-challenge", context: dict[str, Any] | None = None):
        def actions(stage, prefix):
            return [Action(stage, name) for name in self.search_space.choices(stage)]

        def evaluate(prefix):
            start = time.perf_counter()
            try:
                compiled = self.compiler.compile(circuit, prefix)
                metrics = circuit_metrics(compiled)
                verification = self.verifier.verify(circuit, compiled)
                return Evaluation(
                    success=True,
                    metrics=metrics,
                    verification=verification,
                    artifacts={"prefix": [a.to_dict() for a in prefix]},
                    cost={"compile_seconds": time.perf_counter() - start},
                )
            except Exception as exc:
                return Evaluation(
                    success=False,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    cost={"compile_seconds": time.perf_counter() - start},
                )

        return self.engine.run(challenge_id=challenge_id, actions=actions, evaluate=evaluate)
