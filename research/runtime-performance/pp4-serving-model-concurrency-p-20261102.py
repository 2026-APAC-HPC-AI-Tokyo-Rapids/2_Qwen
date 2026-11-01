import importlib.util
from multiprocessing import Pool
spec = importlib.util.spec_from_file_location("m", str(__import__("pathlib").Path(__file__).with_name("pp4-serving-model.py"))); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P_NOW = (0.1132, 0.1138, 0.1166)
def ev(args):
    tag, kw = args
    o = m.avg(seeds=12, **kw)
    return tag, kw["C"], {k: round(o[k], 2) for k in m.KEYS}
items = []
for ds in (0.985, 0.98, 0.975):
    for C in (216, 220, 224, 228, 232, 236):
        items.append((f"ds{ds}", dict(m.NOSTALL, P=P_NOW, C=C, step_scale=ds)))
with Pool() as pool:
    for tag, C, o in pool.map(ev, items):
        print(f"{tag} C{C} score {o['score']:8.1f} sd {o['score_sd']:5.1f} TTFTfail {o['ttft_fail']:5.1f} ITLfail {o['itl_fail']:4.1f} ITL {o['itl_mean']:5.2f}/{o['itl_p99']:5.2f} ssTTFT {o['ss_ttft']:.2f} rate {o['ss_rate']:.2f} decn {o['dec_n']:.1f}")
