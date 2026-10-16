"""Freeze a host-submission diagnostic, without altering the primary plan."""
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c16 import common as c


def main():
    c.protect_prior()
    protocol=c.load_protocol()
    names=list(protocol['worker_source_SHA256'])
    names += [(c.SOURCE/name).relative_to(c.ROOT).as_posix() for name in ('submission.py','run_submission.py')]
    target=Path('/tmp/tma-phase11-c16-submission-snapshot')
    hashes={}
    for name in sorted(set(names)):
        data=(c.ROOT/name).read_bytes()
        path=target/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        hashes[name]=c.sha(data)
    diagnostic={'scope':'Host-submission attribution only; separate from frozen primary model check',
        'reason':'Direct single-kernel CUDA events can include CPU submission gaps; the 26.6us residual is not yet an isolated device cost',
        'frozen_before_counter_values_and_new_timing_analysis':True,
        'primary_protocol_SHA256':c.sha(c.encode(protocol)),
        'worker_source_SHA256':hashes,'new_programs':0,'workers':1,'binaries':10,
        'cache_modes':['warm','cold'],'methods':['direct','graph'],
        'graph':'exactly one original kernel node between two external event record nodes',
        'samples_per_cell':20,'event_samples':800,'batch_divide':False,
        'model_parameters_changed':False,'outcome_based_repeats':False,
        'documentation':['https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__EVENT.html',
                         'https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__GRAPH.html']}
    c.write_once(c.OUT/'preregistration/submission_protocol.json',diagnostic)
    c.write_once(c.OUT/'preregistration/submission_source_snapshot.zip',c.archive(target/'experiments'))
    print('Submission diagnostic frozen: one kernel per graph sample; 800 paired event samples; no new programs')


if __name__=='__main__':
    main()
