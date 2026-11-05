"""Replay the complete frozen non-reduction matrix with shared cadence control.

The target kernels and their artifacts are unchanged. MRoPE's in-place Q/K
storages are restored from immutable GPU snapshots before every untimed prime
and measured target, so repeated priming does not change the mathematical input.
"""
import ctypes as C
from pathlib import Path
import traceback

from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c
from experiments.tma_reduction_layout.phase11.nonreduction_survey.fixtures import DriverFixture
from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device, Loaded
from .nonreduction_audit import physical_identity


def groups(programs):
    """Deduplicate only valid same-case aliases, retaining independent U/G0."""
    result = {}
    for meta in programs:
        key = (meta['geometry'], physical_identity(meta))
        if key not in result:
            result[key] = dict(meta=meta, aliases=[])
        elif meta['variant'] == 'G0' and result[key]['meta']['variant'] != 'G0':
            result[key]['aliases'].append(result[key]['meta']['tag'])
            result[key]['meta'] = meta
        else:
            result[key]['aliases'].append(meta['tag'])
    return list(result.values())


class Restore:
    """Capture-safe GPU copies from initial physical storage, never lane scalars."""
    def __init__(self, fixture, device):
        self.device = device
        self.rows = []
        for group in sorted(fixture.mutated):
            blob = fixture.groups[group]['raw'].tobytes()
            source = device.allocate(len(blob))
            device.upload(source, blob)
            self.rows.append(dict(group=group, source=source,
                destination=fixture.bases[group] + 256,
                size_bytes=len(blob), SHA256=c.sha(blob)))

    def __call__(self):
        for row in self.rows:
            self.device.driver.call('cuMemcpyDtoDAsync_v2',
                [C.c_uint64, C.c_uint64, C.c_size_t, C.c_void_p],
                row['destination'], row['source'], row['size_bytes'], self.device.stream)

    def check(self):
        rows = []
        for row in self.rows:
            actual = c.sha(self.device.download(row['source'], row['size_bytes']))
            rows.append(dict(group=row['group'], bytes=row['size_bytes'],
                initial_SHA256=row['SHA256'], actual_SHA256=actual,
                pass_=actual == row['SHA256']))
        return rows


def run(protocol, payload, root, invocation):
    from .engine import Session

    root = Path(root)
    device = session = None
    loaded = {}
    try:
        suite = protocol['suites']['nonreduction']
        assert c.sha(payload) == suite['payload_SHA256']
        for name, digest in protocol['source_SHA256'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
        artifact = root/'program_files'
        c.extract(payload, artifact)
        parents = suite['parent_protocols']
        programs_by_source = suite['programs_by_source']
        assert set(parents) == set(programs_by_source) == {'n2', 'n5'}
        device = Device()
        session = Session(device, protocol, root, invocation, 'nonreduction')
        logical_count = physical_count = context_count = 0
        for source in ('n2', 'n5'):
            parent = parents[source]
            programs = programs_by_source[source]
            assert programs == parent['programs'], 'Original program metadata changed'
            prefix = suite.get('artifact_prefixes', {}).get(source, source).rstrip('/')
            for case in parent['cases']:
                metas = [meta for meta in programs if meta['geometry'] == case['id']]
                assert metas and {m['variant'] for m in metas} >= {'U', 'G0'}
                expected = next(m['fixture'] for m in metas
                    if m['variant'] in ('U', 'G0') and 'fixture' in m)
                fixture = DriverFixture(c.adapter(case).fixture(case, 'finite'), device)
                assert c.encode(fixture.metadata()) == c.encode(expected), case['id']
                restore = Restore(fixture, device)
                assert bool(restore.rows) == bool(expected['inplace'])
                items, owners = [], []
                for group in groups(metas):
                    original = group['meta']
                    meta = {**original,
                        'artifact_directory': prefix + '/' + original['artifact_directory']}
                    assert meta['tag'] not in loaded, meta['tag']
                    kernel = Loaded(device, artifact, meta)
                    loaded[meta['tag']] = kernel
                    owner, params = kernel.parameters(fixture.arguments(original))
                    owners.append(owner)

                    def launch(kernel=kernel, params=params, grid=fixture.grid):
                        kernel.launch(grid, params, device.stream)

                    def readback(fixture=fixture, restore=restore):
                        # No reset or target launch: this validates the actual
                        # captured graph's final target, including all restores.
                        device.synchronize()
                        result = fixture.check()
                        snapshots = restore.check()
                        result['immutable_restore_snapshots'] = snapshots
                        result['pass'] = bool(result['pass'] and all(row['pass_'] for row in snapshots))
                        result['pattern'] = 'finite'
                        return result

                    def check(kernel=kernel, params=params, fixture=fixture, readback=readback):
                        # These CPU copies/checks happen only at engine block
                        # boundaries. Captured sample paths use GPU Restore.
                        fixture.reset()
                        kernel.launch(fixture.grid, params, device.stream)
                        device.synchronize()
                        return readback()

                    items.append(dict(tag=meta['tag'], aliases=group['aliases'],
                        role=original['variant'] if original['variant'] in ('U', 'G0') else 'GL',
                        kernels=[kernel], launch=launch,
                        prepare=restore if restore.rows else None,
                        expected_nodes=dict(kernels=1, memsets=0, copies=len(restore.rows)),
                        check=check, check_graph=readback))
                context = dict(id=case['id'], case=case['id'], geometry=case['id'],
                    family=case['family'], source=source, case_metadata=case,
                    grid=list(fixture.grid), expected_fixture=expected,
                    logical_programs=len(metas), physical_programs=len(items),
                    inplace_groups=len(restore.rows),
                    inplace_restore='Immutable GPU snapshot copied before every prime and target; copies precede measured start event.',
                    original_three_pattern_eligibility=suite.get('eligibility_by_source', {}).get(source),
                    fresh_check_patterns=['finite'],
                    graph_oracle_readback_required=True)
                session.measure(context, items)
                logical_count += len(metas)
                physical_count += len(items)
                context_count += 1
                device.free_buffers()
                del items, owners, fixture, restore
        assert (context_count, logical_count, physical_count) == (16, 144, 134)
        return session.finish({tag: kernel.binding() for tag, kernel in loaded.items()})
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        bindings = {tag: kernel.binding() for tag, kernel in loaded.items()}
        if session is not None:
            return session.finish(bindings, error=error)
        cleanup_error = None
        if device:
            try:
                device.close()
            except Exception:
                cleanup_error = traceback.format_exc()
        failure = dict(status='FAILED', suite='nonreduction', invocation=invocation,
            protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
            error=error, cleanup_error=cleanup_error, bindings=bindings,
            compiler_calls=0, native_rebuilds=0)
        c.write_once(root/'result.json', failure)
        return failure
