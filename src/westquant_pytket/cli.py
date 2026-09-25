from __future__ import annotations
import argparse, json
from pathlib import Path
from westquant_core import write_jsonl
from .search import PytketCompiler, PytketSequentialSearch


def _parse_edges(text: str):
    if not text:
        return []
    return [tuple(map(int, item.split("-"))) for item in text.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description="WestQuant sequential representation search for pytket")
    p.add_argument("--qasm", required=True, help="Input OpenQASM file")
    p.add_argument("--edges", default="", help="Architecture edges, e.g. 0-1,1-2,2-3")
    p.add_argument("--beam-width", type=int, default=3)
    p.add_argument("--output", default="results/westquant-pytket")
    args = p.parse_args()

    from pytket.qasm import circuit_from_qasm
    circuit = circuit_from_qasm(args.qasm)
    search = PytketSequentialSearch(beam_width=args.beam_width, compiler=PytketCompiler(architecture_edges=_parse_edges(args.edges)))
    result = search.run(circuit, challenge_id=Path(args.qasm).stem, context={"architecture_edges": _parse_edges(args.edges)})
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"best": result.best.to_dict() if result.best else None, "n_states": len(result.states)}, indent=2), encoding="utf-8")
    write_jsonl(out / "trajectory.jsonl", result.records(framework="pytket", context={"architecture_edges": _parse_edges(args.edges)}))

if __name__ == "__main__": main()
