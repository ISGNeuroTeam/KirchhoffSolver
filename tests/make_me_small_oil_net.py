import pandas as pd
import numpy as np
from ksolver.io.calculate_DF import calculate_DF, calculate_PPD_from_nodes_edges_df, calculate_PPD_from_dfs
from ksolver.tools.HE2_schema_maker import make_oilpipe_schema_from_OT_dataset
from ksolver.tools.HE2_tools import dump_graph_results_to_dicts, dump_graph_results_to_json_str
from ksolver.tools.HE2_ABC import Root
from ksolver.solver.HE2_Solver import HE2_Solver
from ksolver.graph.nodes.HE2_Vertices import is_source, HE2_Source_Vertex, HE2_ABC_GraphVertex, HE2_Boundary_Vertex
from ksolver.graph.edges.HE2_WellPump import HE2_WellPump, create_HE2_WellPump_instance_from_dataframe
import ksolver.graph.edges.HE2_WellPump as wp
from ksolver.graph.edges.HE2_Pipe import HE2_OilPipe
from ksolver.graph.edges.HE2_Plast import HE2_Plast
from ksolver.fluids.HE2_Fluid import HE2_BlackOil, gimme_dummy_BlackOil
from ksolver.tools.HE2_ABC import oil_params
import networkx as nx
from numpy.random import Generator, PCG64

Tailaki_oil_params = oil_params(
    sat_P_bar=67,
    plastT_C=84,
    gasFactor=36,
    oildensity_kg_m3=826,
    waterdensity_kg_m3=1015,
    gasdensity_kg_m3=1,
    oilviscosity_Pa_s=35e-3,
    volumewater_percent=50,
    volumeoilcoeff=1.017,
)

pump_curves = pd.read_csv('C:\work\IsNeugro\TajlakiData\PumpChart.csv')
_ = create_HE2_WellPump_instance_from_dataframe(pump_curves)
pump_models = list(wp.pumps_cache.keys())
pump_models = pump_models[:14] + pump_models[16:]

def make_small_oil_tree_net(pads_cnt, wells_cnt, rand_seed=None):

    # Создаем рандомом все параметры дуг графа
    rng = Generator(PCG64(rand_seed))
    PPlast = rng.uniform(230, 270, wells_cnt)
    KProd = rng.uniform(0.2, 1.5, wells_cnt)
    WC = rng.uniform(0.40, 0.98, wells_cnt)
    CasingH = rng.uniform(100, 500, wells_cnt)
    global pump_models
    Pumps = rng.integers(0, len(pump_models), wells_cnt)
    Pump_H_noise = rng.uniform(-150, 150, wells_cnt)
    Freqs = rng.uniform(40, 60, wells_cnt)
    OG_pipes_dx = rng.uniform(300, 1500, 2*pads_cnt)
    OG_pipes_dy = rng.uniform(-15, 15, 2*pads_cnt)

    rez = nx.MultiDiGraph()

    pds = list(range(pads_cnt))
    wls = list(range(wells_cnt))
    wells = dict()
    for well in wls:
        pad = rng.choice(pds)
        name = f"PAD_{pad}_well_{well}"
        wells[name] = (well, pad)

    # В эти словари сложим объекты узлов и дуг, потом прицепим их к графу
    edge_objs = {}
    node_objs = {}

    # Узлы для кустов
    rez.add_nodes_from([f'PAD{n}' for n in pds])
    node_objs.update({f'PAD{n}': HE2_ABC_GraphVertex() for n in pds})

    # Создаем подграф скважины и сразу приецпляем его к узлу куста
    for well, (well_num, pad) in wells.items():
        nodes=[well, f"{well}_zaboi", f"{well}_pump_intake", f"{well}_pump_outlet", f"{well}_wellhead", f"PAD{pad}"]
        edgelist = []
        for i in range(len(nodes) - 1):
            edgelist += [(nodes[i], nodes[i + 1], 0)]
        rez.add_nodes_from(nodes[:-1])
        rez.add_edges_from(edgelist)

        op_dict = Tailaki_oil_params._asdict()
        op_dict.update(volumewater_percent=WC[well_num])
        op = oil_params(**op_dict)
        fluid = HE2_BlackOil(op)
        src_obj = HE2_Source_Vertex("P", PPlast[well_num], fluid, 20)
        node_objs.update({n: HE2_ABC_GraphVertex() for n in nodes})
        node_objs[nodes[0]] = src_obj

        pump = pump_models[Pumps[well_num]]
        pump_H = float(pump[0].split('-')[2])
        pump_H += Pump_H_noise[well_num]
        edge_objs[edgelist[0]] = HE2_Plast(KProd[well_num])
        edge_objs[edgelist[1]] = HE2_OilPipe([0], [CasingH[well_num]], [0.01], [1e-5])
        edge_objs[edgelist[2]] = create_HE2_WellPump_instance_from_dataframe(pump_curves, pump[0], None, Freqs[well_num], pump[1])
        edge_objs[edgelist[3]] = HE2_OilPipe([0], [pump_H], [0.057], [1e-5])
        edge_objs[edgelist[4]] = HE2_OilPipe([10], [0], [0.2], [1e-5])

    # Теперь часть графа между кустами и ДНС
    rez.add_nodes_from(map(str, range(pads_cnt-1)))
    node_objs.update({str(i): HE2_ABC_GraphVertex() for i in range(pads_cnt-1)})
    edgelist = [(str(i), str(i+1), 0) for i in range(pads_cnt-2)]
    edgelist += [('PAD0', '0', 0)]
    edgelist += [(f'{pads_cnt-2}', 'DNS', 0)]
    edgelist += [(f'PAD{i+1}', f'{i}', 0) for i in range(pads_cnt-1)]
    rez.add_edges_from(edgelist)
    edge_objs.update({e:HE2_OilPipe([OG_pipes_dx[i]], [OG_pipes_dy[i]], [0.2], [1e-5]) for i, e in enumerate(edgelist)})
    dns_obj = HE2_Boundary_Vertex('P', 7)
    dns_obj.fluid = gimme_dummy_BlackOil()
    node_objs['DNS'] = dns_obj
    for e in edge_objs:
        edge_objs[e].fluid = gimme_dummy_BlackOil()

    # Прицепляем объекты узлов и дуг к графу
    nx.set_node_attributes(rez, name="obj", values=node_objs)
    nx.set_edge_attributes(rez, name="obj", values=edge_objs)
    return rez


if __name__ == '__main__':
    net = make_small_oil_tree_net(2, 4, 42)
