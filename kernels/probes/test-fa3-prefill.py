"""CPU checks for causal work accounting and static SASS region parsing."""
import ast
import contextlib
import io
import runpy
from pathlib import Path

root = Path(__file__).parent
source = ast.parse((root/'fa3-prefill.py').read_text())
node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'work_counts')
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'work_counts', 'exec'), namespace)  # noqa: S102 (extract pure function without CUDA imports)
counts = namespace['work_counts']
assert counts([4096]) == (19760, 2048)
assert counts([1024]*4) == (5760, 2048)
assert counts([4027]) == (19152, 2016)
assert counts([1]) == (4,4)
i1,t1 = counts([4096])
i2,t2 = counts([1024]*4)
x,y = .01,.1
a,b = i1*x+t1*y,i2*x+t2*y
det = i1*t2-i2*t1
assert abs((a*t2-b*t1)/det-x)<1e-12
assert abs((i1*b-i2*a)/det-y)<1e-12
fixture = '''Function : fixture
 /*0000*/ MOV R1, c[0x0][0x28];
 /*0010*/ SETMAXNREG.DEC 0x18;
 /*0020*/ STL [R1], R2;
 /*0030*/ SETMAXNREG.INC 0xf0;
 /*0040*/ WARPGROUP.ARRIVE;
 /*0050*/ MUFU.EX2 R2, R3;
 /*0060*/ F2FP.F8.PACK_AB R4, R5, R6;
 /*0070*/ LDL R2, [R1];
 /*0080*/ WARPGROUP.DEPBAR.LE SB0, 0x0;
 /*0090*/ WARPGROUP.ARRIVE;
 /*00a0*/ @P0 BRA 0x40;
'''
report = runpy.run_path(str(root/'fa3-prefill-sass.py'))['report']
output = io.StringIO()
with contextlib.redirect_stdout(output):
    report(fixture)
text = output.getvalue()
assert "producer 0x10..0x20" in text and "'STL': 1" in text
assert "consumer 0x30..0xa0" in text
window = next(line for line in text.splitlines() if 'ARRIVE-window' in line)
assert "'MUFU.EX2': 1" in window and "'F2FP': 1" in window and "'LDL': 1" in window
assert 'backedge 0xa0 -> 0x40' in text
assert 'arithmetic-backedge-loop 0x40..0xa0' in text
roles = runpy.run_path(str(root/'fa3-prefill-sass.py'))['register_role']
assert roles('USETMAXREG.TRY_ALLOC.CTAPOOL', 'UP0, 0xe8') == 'consumer'
assert roles('USETMAXREG.DEALLOC.CTAPOOL', '0x28') == 'producer'
print('PASS CPU causal work accounting and SASS register/loop census')
