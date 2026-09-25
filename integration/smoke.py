from pytket import Circuit
from westquant_pytket import PytketCompiler, PytketSequentialSearch

c=Circuit(4).H(0).CX(0,1).CX(1,2).CX(2,3)
search=PytketSequentialSearch(beam_width=2,compiler=PytketCompiler(architecture_edges=[(0,1),(1,2),(2,3)]))
r=search.run(c,challenge_id='smoke-pytket')
assert r.best is not None
print(r.best.to_dict())
