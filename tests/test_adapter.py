from westquant_pytket.adapter import circuit_metrics


class Op:
    def __init__(self, typ): self.type = typ
class Cmd:
    def __init__(self, typ, n):
        self.op = Op(typ)
        self.qubits = [object() for _ in range(n)]
class Circuit:
    n_qubits = 3
    n_bits = 1
    def get_commands(self): return [Cmd("H", 1), Cmd("CX", 2), Cmd("CX", 2)]


def test_metrics():
    m = circuit_metrics(Circuit())
    assert m["two_qubit_gates"] == 2
    assert m["n_gates"] == 3
