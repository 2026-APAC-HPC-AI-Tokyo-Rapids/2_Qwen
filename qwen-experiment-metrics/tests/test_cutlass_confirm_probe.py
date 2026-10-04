"""Predeclared confirmation gate is testable without CUDA."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('confirm_probe',Path(__file__).parents[2]/'scripts/benchmark-prefill-cutlass-confirm.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture(balanced=.99,skew=.998,heldout=.999,quality=True):
    def case(pattern,ratio,tokens=8192):
        variants={'wheel':{'rounds_ms':[1]*20}}
        for name in probe.CANDIDATE_IDS:
            variants[name]={'status':'measured','rounds_ms':[ratio]*20,'quality_eligible':quality}
        return {'tokens':tokens,'distribution':pattern,'variants':variants}
    return [case('balanced',balanced),case('skewed',skew)],[case('skewed',heldout,4096),case('moderate',heldout)]


def test_gate_requires_both_predeclared_effects_quality_and_heldout_nonregression():
    assert all(x['e2e_evidence_gate_pass'] for x in probe.evaluate(*fixture()).values())
    for kwargs in [{'balanced':.997},{'skew':.9995},{'heldout':1.001},{'quality':False}]:
        assert not any(x['e2e_evidence_gate_pass'] for x in probe.evaluate(*fixture(**kwargs)).values())


def test_gate_accepts_small_number_of_slow_rounds_but_not_more_than_five():
    training,heldout=fixture()
    for case in training:
        for name in probe.CANDIDATE_IDS:
            case['variants'][name]['rounds_ms'][:5]=[1.001]*5
    assert all(x['e2e_evidence_gate_pass'] for x in probe.evaluate(training,heldout).values())
    for case in training:
        for name in probe.CANDIDATE_IDS:
            case['variants'][name]['rounds_ms'][5]=1.001
    assert not any(x['e2e_evidence_gate_pass'] for x in probe.evaluate(training,heldout).values())


def test_bootstrap_is_deterministic_diagnostic_not_gate():
    result=probe.summarize([.99]*20,[1]*20)
    assert result['faster_rounds']==20
    assert result['diagnostic_bootstrap95']==[.99,.99]
    assert probe.GATE['automatic_adoption'] is False
