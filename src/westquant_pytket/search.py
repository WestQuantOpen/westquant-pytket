from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import permutations
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
        warnings: list[str] = []
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
                try:
                    AASRouting(arc).apply(out)
                except Exception as exc:
                    warnings.append(f"aas_routing_failed:{type(exc).__name__}")
                    RoutingPass(arc).apply(out)

        name = selected.get("rebase", "none")
        if name == "tket":
            RebaseTket().apply(out)
        elif name == "synthesise":
            SynthesiseTket().apply(out)

        # Attach compile warnings to the circuit so the evaluator can record them.
        try:
            out._compile_warnings = warnings  # type: ignore[attr-defined]
        except Exception:
            pass
        return out


class PytketVerifier:
    def __init__(self, *, max_unitary_qubits: int = 7) -> None:
        self.max_unitary_qubits = max_unitary_qubits

    @staticmethod
    def _permutation_matrix(n: int, perm: tuple[int, ...]) -> Any:
        """Build the qubit-permutation matrix for *n* qubits."""
        import numpy as np

        dim = 1 << n
        p = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            bits = [(i >> (n - 1 - j)) & 1 for j in range(n)]
            perm_bits = [bits[perm[j]] for j in range(n)]
            perm_i = sum(perm_bits[j] << (n - 1 - j) for j in range(n))
            p[perm_i, i] = 1.0
        return p

    @staticmethod
    def _equivalent_up_to_phase(u: Any, v: Any, atol: float = 1e-8) -> bool:
        """Return True when *u* and *v* are equal up to a global phase."""
        import numpy as np

        overlap = np.vdot(u.ravel(), v.ravel())
        if abs(overlap) < 1e-15:
            return bool(np.allclose(u, v, atol=atol, rtol=atol))
        phase = np.conj(overlap) / abs(overlap)
        return bool(np.allclose(u, phase * v, atol=atol, rtol=atol))

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

            # Try every permutation of the qubit ordering.  Routing passes may
            # permute the logical-to-physical qubit mapping, which changes the
            # raw unitary even though the circuit is logically equivalent.
            for perm in permutations(range(n)):
                if perm == tuple(range(n)):
                    if self._equivalent_up_to_phase(u, v):
                        return {"equivalence": "exact", "verified": True, "method": "get_unitary"}
                else:
                    p = self._permutation_matrix(n, perm)
                    v_perm = p @ v @ p.T.conj()
                    if self._equivalent_up_to_phase(u, v_perm):
                        return {"equivalence": "exact", "verified": True, "method": "get_unitary"}

            return {"equivalence": "invalid", "verified": True, "method": "get_unitary"}
        except Exception as exc:
            return {"equivalence": "unknown", "verified": False, "reason": type(exc).__name__, "message": str(exc)}


class PytketSequentialSearch:
    def __init__(self, *, search_space: PytketSearchSpace | None = None, beam_width: int = 3,
                 compiler: Any | None = None, verifier: Any | None = None, seed: int | None = None) -> None:
        self.search_space = search_space or PytketSearchSpace()
        self.compiler = compiler or PytketCompiler()
        self.verifier = verifier or PytketVerifier()
        self.seed = seed
        self.engine = DeterministicBeamSearch(
            stages=STAGES,
            beam_width=beam_width,
            objectives=(Objective("two_qubit_gates"), Objective("depth"), Objective("n_gates")),
        )

    def _bind_parameters(self, circuit: Any, parameter_values: dict | None) -> Any:
        """Bind free symbols to concrete values when the circuit is parameterised."""
        is_symbolic = getattr(circuit, "is_symbolic", lambda: False)
        if not is_symbolic():
            return circuit
        free_symbols = getattr(circuit, "free_symbols", lambda: set())()
        if not free_symbols:
            return circuit
        import numpy as np

        rng = np.random.default_rng(self.seed)
        symbol_map = dict(parameter_values or {})
        for sym in free_symbols:
            if sym not in symbol_map:
                symbol_map[sym] = float(rng.random())
        out = circuit.copy()
        out.symbol_substitution(symbol_map)
        return out

    def run(self, circuit: Any, *, challenge_id: str = "pytket-challenge", context: dict[str, Any] | None = None,
            bind_parameters: bool = True, parameter_values: dict | None = None):
        # Bind parameters before the search so that downstream passes (which
        # cannot handle symbolic circuits) work correctly.
        search_circuit = circuit
        if bind_parameters:
            try:
                search_circuit = self._bind_parameters(circuit, parameter_values)
            except Exception:
                search_circuit = circuit

        def actions(stage, prefix):
            return [Action(stage, name) for name in self.search_space.choices(stage)]

        def evaluate(prefix):
            start = time.perf_counter()
            try:
                compiled = self.compiler.compile(search_circuit, prefix)
                metrics = circuit_metrics(compiled)
                # Surface any compile-time warnings (e.g. AAS fallback).
                warnings = getattr(compiled, "_compile_warnings", None)
                if warnings:
                    metrics["warnings"] = list(warnings)
                verification = self.verifier.verify(search_circuit, compiled)
                return Evaluation(
                    success=True,
                    metrics=metrics,
                    verification=verification,
                    artifacts={
                        "prefix": [a.to_dict() for a in prefix],
                        "compiled_circuit": compiled,
                    },
                    cost={"compile_seconds": time.perf_counter() - start},
                )
            except Exception as exc:
                return Evaluation(
                    success=False,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    cost={"compile_seconds": time.perf_counter() - start},
                )

        return self.engine.run(challenge_id=challenge_id, actions=actions, evaluate=evaluate)
