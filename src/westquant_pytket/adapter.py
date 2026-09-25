from __future__ import annotations

from collections import Counter
from typing import Any
from westquant_core import Representation, RepresentationKind


def circuit_metrics(circuit: Any) -> dict[str, Any]:
    commands = list(circuit.get_commands())
    counts = Counter(str(cmd.op.type) for cmd in commands)
    two_qubit = sum(1 for cmd in commands if len(cmd.qubits) == 2)
    depth = None
    try:
        depth = int(circuit.depth())
    except Exception:
        pass
    return {
        "n_qubits": int(circuit.n_qubits),
        "n_bits": int(circuit.n_bits),
        "n_gates": len(commands),
        "depth": depth,
        "two_qubit_gates": two_qubit,
        "operations": dict(counts),
    }


class PytketAdapter:
    framework = "pytket"

    def import_native(self, circuit: Any, *, representation_id: str = "pytket:circuit") -> Representation:
        return Representation(
            id=representation_id,
            kind=RepresentationKind.CIRCUIT,
            framework=self.framework,
            payload=circuit_metrics(circuit),
            metadata={"native_type": type(circuit).__name__},
        )
