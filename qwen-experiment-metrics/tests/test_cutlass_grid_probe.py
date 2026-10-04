"""CPU contracts of the finite offline grid, without importing CUDA packages."""
import importlib.util
from pathlib import Path

PATH=Path(__file__).parents[2]/'scripts/benchmark-prefill-cutlass-grid.py'
spec=importlib.util.spec_from_file_location('grid_probe',PATH)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_resolved_stage_duplicates_are_not_rebenchmarked():
    base={'tile_mnk': [128,128,128],'cluster_mnk': [1,2,1],'schedule': 'Cooperative'}
    entries=[dict(base,mode=0,resolved_stages=4),dict(base,mode=1,resolved_stages=2),dict(base,mode=2,resolved_stages=4)]
    unique,duplicate=probe.unique_configurations(entries)
    assert [x['mode'] for x in unique]==[0,1]
    assert duplicate==[2]


def test_rotation_reverse_keeps_each_candidate_exactly_once():
    names=['wheel','stock','up0','down0','up1']
    orders=[probe.rotated_order(names,index) for index in range(5)]
    assert all(sorted(order)==sorted(names) for order in orders)
    assert len({tuple(order) for order in orders})==5


def test_shortlist_rejects_regression_missing_pattern_and_quality_failure():
    records=[]
    for case in range(2):
        records.append({'variants':{'stock':{'rounds_ms':[1]*5},'winner':{'role':'up','mode':1,'status':'measured','quality_eligible':True,'rounds_ms':[.95]*5},'regression':{'role':'up','mode':2,'status':'measured','quality_eligible':True,'rounds_ms':[.9 if case==0 else 1.01]*5},'bad':{'role':'up','mode':3,'status':'measured','quality_eligible':False,'rounds_ms':[.8]*5}}})
    assert probe.shortlist(records,'up')==[1]
    assert probe.shortlist(records,'down')==[]


def test_manifest_rejects_stale_source_and_old_mode_map(tmp_path):
    import hashlib

    import pytest

    source=tmp_path/'kernel.cu'
    source.write_text('fixed kernel\n')
    entry={'id':'coop128_c1x2_sauto','mode':0,'resolved_stages':5}
    manifest={'schema':'prefill-moe-grid/v2','source_hashes':{'kernel.cu':hashlib.sha256(source.read_bytes()).hexdigest()},'configurations':[entry]}
    condition="def _prefill_tile_modes(up,down,*args):\n    names={'coop128_c1x2_sauto':0}\n    return names[up],names[down]\n"
    probe.validate_manifest(manifest,[{'mode':0,'resolved_stages':5}],tmp_path,condition)
    with pytest.raises(AssertionError,match='production name/ID mismatch'):
        probe.validate_manifest(manifest,[{'mode':0,'resolved_stages':5}],tmp_path,condition.replace("sauto':0","sauto':3"))
    source.write_text('old kernel\n')
    with pytest.raises(AssertionError,match='source hash mismatch'):
        probe.validate_manifest(manifest,[{'mode':0,'resolved_stages':5}],tmp_path,condition)
