"""Fresh natural FULL/T_control curves for the complete seen C2d/D1 domains."""
from pathlib import Path
import traceback

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device
from .anchors_audit import groups,selected_cells,default_tag
from .legacy import OracleSession


def run(protocol,payload,root,invocation):
    from .engine import Session as Measurement
    root = Path(root)
    suite = protocol['suites']['anchors']
    assert c.sha(payload)==suite['payload_SHA256']
    artifact = root/'program_files'
    c.extract(payload,artifact)
    panels = suite['panels']
    assert set(panels)=={'c2d','d1'}
    assert sum(len(selected_cells(p)) for p in panels.values())==417
    device = measurement = oracle = None
    error = None
    bindings = {}
    try:
        device = Device()
        measurement = Measurement(device,protocol,root,invocation,'anchors')
        measurement.result.update(
            source_control_scope='Explicit frozen Gluon upstream-layout reference and independent T_control. No Triton U/G0 pair is claimed.',
            canonical_physical_conditions=417,contexts_expected=57,
            D1_new_blind_claim=False,selector_refit=False,
            execution_bindings={},readonly_input_checks=[],complete_eligibility={})
        for panel_name in ('c2d','d1'):
            panel = panels[panel_name]
            rows = selected_cells(panel)
            metadata = {tag:panel['gate']['metadata'][tag] for tag in sorted({row['tag'] for row in rows})}
            namespace = panel.get('namespace',panel_name)
            assert namespace==panel_name
            oracle = OracleSession(device,artifact/namespace,metadata,rows)
            for tag,kernel in oracle.kernels.items():
                expected = panel['gate']['resources'][tag]
                actual = kernel.binding()
                assert actual['ABI']==expected['ABI']
                assert actual['attributes']==expected['attributes'],(panel_name,tag)
                assert actual['reservations']==expected['reservations'],(panel_name,tag)
            if invocation==0:
                measurement.result['complete_eligibility'][panel_name] = oracle.eligibility(
                    panel['parent_protocol']['correctness']['patterns'])
            for (m,n,w,grid),cells in groups(panel):
                fullgrid = (8192//m)*(8192//n)
                factor = grid//fullgrid
                context = dict(id=f'{panel_name}|M{m}|N{n}|W{w}|qnatural|G{grid}',
                    parent_stage=panel_name.upper(),panel=panel_name,M=m,N=n,W=w,
                    geometry=f'M{m}_N{n}_w{w}',
                    q='natural',grid=grid,K=1,footprint_factor=factor,
                    footprint_bytes=grid*m*n*2,
                    source_protocol_SHA256=panel['source']['protocol_SHA256'],
                    reference_scope='Frozen upstream-default explicit Gluon layout, not source-aligned Triton U/G0.')
                items=[]
                for cell in sorted(cells,key=lambda row:row['tag']):
                    tag=cell['tag'];meta=metadata[tag];kernel=oracle.kernels[tag]
                    owners,params,serialized=oracle.params(tag,1)
                    role=('control' if meta['family']=='ONE_TRANSFER' else
                          'default' if tag==default_tag(meta) else 'candidate')
                    itemtag=panel_name+'|'+cell['cell_id']
                    aliases=[panel_name+'|'+alias for alias in cell['aliases']]
                    item=dict(tag=itemtag,aliases=aliases,role=role,kernels=[kernel],prepare=None,
                        expected_nodes=dict(kernels=1,memsets=0,copies=0),
                        launch=lambda kernel=kernel,params=params,grid=grid:kernel.launch(grid,params,device.stream,'natural'),
                        check=lambda tag=tag,grid=grid,params=params:oracle.full_check(tag,grid,'natural',params),
                        check_graph=lambda tag=tag,grid=grid:oracle.read_check(tag,grid,'natural'),
                        parameter_owners=owners,parameters=params,serialized_parameters=serialized,
                        original_tag=tag,panel=panel_name,q='natural',grid=grid,K=1)
                    items.append(item)
                    measurement.result['execution_bindings'][itemtag]=dict(
                        original_tag=tag,original_cell_id=cell['cell_id'],panel=panel_name,aliases=aliases,
                        function_handle=kernel.function.value,serialized_parameters=serialized,
                        q='natural',grid=grid,K=1,reservation=kernel.reservations['natural'],
                        tensordesc_meta=meta['tensordesc_meta'],target_CUBIN_SHA256=kernel.digest)
                measurement.measure(context,items)
            checked=oracle.readonly_input_check()
            measurement.result['readonly_input_checks'].append(dict(panel=panel_name,**checked))
            assert checked['pass'],checked
            bindings.update({panel_name+'|'+tag:kernel.binding() for tag,kernel in oracle.kernels.items()})
            # All graphs have been destroyed; retain modules but release the
            # panel's buffers before allocating the next independent matrix.
            device.free_buffers()
            oracle=None
    except Exception:
        error=traceback.format_exc()
        print(error,flush=True)
        if oracle:
            bindings.update({'partial|'+tag:kernel.binding() for tag,kernel in oracle.kernels.items()})
    if measurement:
        return measurement.finish(bindings,error=error)
    if device:
        device.close()
    result=dict(status='FAILED',suite='anchors',invocation=invocation,error=error)
    c.write_once(root/'result.json',result)
    return result
