from os import path
import pandas as pd
import numpy as np
from ksolver.io.calculate_DF import calculate_DF, calculate_PPD_from_nodes_edges_df, calculate_PPD_from_dfs
from ksolver.tools.viz import draw_result_graph
# import tests.config as config
import config
import otp
from ksolver.tools.HE2_schema_maker import make_oilpipe_schema_from_OT_dataset
from ksolver.solver.HE2_Solver import HE2_Solver
from ksolver.graph.nodes.HE2_Vertices import is_source, HE2_Source_Vertex
from itertools import product
from datetime import datetime
from multiprocessing import Pool
import random
from numpy.random import Generator, PCG64


def test4():
    data_folder = config.get_data_folder()
    dataframe = pd.read_csv(path.join(data_folder, "dns_2_pk (1).csv"))
    G, calc_df, df_to_graph_edges_mapping = make_oilpipe_schema_from_OT_dataset(dataframe, folder=data_folder)
    inlets = [n for n in G.nodes if is_source(G, n)]
    wells_count = len(inlets)
    # P_intk_vec = np.ones(wells_count) * 35
    # freqs = np.ones(wells_count) * 47
    P_intk_vec = np.random.uniform(20, 50, wells_count)
    freqs = np.random.uniform(40, 60, wells_count)
    for i, n in enumerate(inlets):
        intake = n + '_pump_intake'
        outlet = n + '_pump_outlet'
        fluid = G.nodes[n]['obj'].fluid
        G.nodes[intake]['obj'] = HE2_Source_Vertex('P', P_intk_vec[i], fluid, 20)
        pump_obj = G[intake][outlet][0]['obj']
        pump_obj.changeFrequency(freqs[i])

    solver = HE2_Solver(G)
    solver.solve(mix_fluids=False, it_limit=150, threshold=5)
    if solver.op_result.success:
        print('Solved!')

def generate_meta_params():
    metas = []
    wides = [3, 4]
    mids = [0.6, 0.7, 0.8, 0.9, 1.0]
    slopes = [1.2, 1.5, 2, 3]
    borders = [3, 3.5, 4, 4.5, 5, 5.5]
    steps = [[], [1], []]
    for (w, m, s, b) in product(wides, mids, slopes, borders):
        if w == 1 and s != slopes[0]:
            continue
        mid = m if m > 0 else 1
        sqs = s ** 0.5
        s0 = mid * sqs ** (1 - w)
        steps = np.array([np.round(s0 * sqs ** (2*i), 4) for i in range(w)])
        meta = dict()
        meta['steps'] = steps
        meta['border_factor'] = b
        metas += [meta]
    return metas

def run_one_net(args):
    P_intk_vec, freq_vec, meta = args
    data_folder = config.get_data_folder()
    dataframe = pd.read_csv(path.join(data_folder, "dns_2_pk (1).csv"))
    G, calc_df, df_to_graph_edges_mapping = make_oilpipe_schema_from_OT_dataset(dataframe, folder=data_folder)
    inlets = [n for n in G.nodes if is_source(G, n)]
    wells_count = len(inlets)
    for i, n in enumerate(inlets):
        intake = n + '_pump_intake'
        outlet = n + '_pump_outlet'
        fluid = G.nodes[n]['obj'].fluid
        G.nodes[intake]['obj'] = HE2_Source_Vertex('P', P_intk_vec[i], fluid, 20)
        pump_obj = G[intake][outlet][0]['obj']
        pump_obj.changeFrequency(freq_vec[i])
    solver = HE2_Solver(G)

    t = datetime.now()
    solver.solve(mix_fluids=False, it_limit=1000, threshold=5, steps=meta['steps'], cut_factor=meta['border_factor'])
    dt = datetime.now() - t
    return dt, solver.op_result


def test5():
    metas = generate_meta_params()
    random.shuffle(metas)
    net_cnt, prcs_cnt = 32, 4
    P_vecs = [np.random.uniform(20, 50, 300) for i in range(net_cnt)]
    f_vecs = [np.random.uniform(40, 60, 300) for i in range(net_cnt)]
    rez_file = open('meta_results.txt', 'r')
    checked_meta_strs = [s[:-1] for s in rez_file]

    with Pool(processes=4) as pool:
        metalist = metas
        for i, meta in enumerate(metalist):
            rez_file = open('meta_results.txt', 'a')
            sss = meta.__str__()
            if sss in checked_meta_strs:
                print('{EQ~!!!')
                continue
            print(meta, file=rez_file)
            print(datetime.now(), f'                   {i}/{len(metalist)}', file=rez_file)
            print(meta)
            print(datetime.now(), f'                   {i}/{len(metalist)}')
            args = list(zip(P_vecs, f_vecs, [meta]*net_cnt))
            rez = []
            while args:
                piece, args = args[:prcs_cnt], args[prcs_cnt:]
                rez += list(pool.map(run_one_net, piece))
                fail_cnt = 0
                for dt, opr in rez:
                    fail_cnt += not opr.success
                if fail_cnt >= 1 + net_cnt // 2:
                    break

            avg_time, avg_fun, avg_its, succ_cnt = 0, 0, 0, 0
            for dt, opr in rez:
                avg_time += dt.seconds
                succ_cnt += opr.success
                avg_fun += opr.fun
                avg_its += opr.nfev
            avg_time /= net_cnt
            avg_fun /= net_cnt
            avg_its /= net_cnt
            msg = f'avg_time={avg_time:.3f}, avg_fun={avg_fun:.3f},  avg_its={avg_its:.3f}, succ_cnt={succ_cnt}/{net_cnt}'
            print(msg)
            print(msg, file=rez_file)

            for dt, opr in rez:
                msg = f'     scnds={dt.seconds}, fun={opr.fun}, nfev={opr.nfev}, succ={opr.success}'
                print(msg)
                print(msg, file=rez_file)
            rez_file.close()


def test6():
    # metas = generate_meta_params()
    metas = [{'steps': np.array([0.2828, 0.5657, 1.1314, 2.2627]), 'border_factor': 5}]
    # metas += [{'steps': np.array([0.1, 0.2, 0.4, 0.8, 1.6]), 'border_factor': 3}]
    random.shuffle(metas)
    rng = Generator(PCG64(42))
    net_cnt = 32
    P_vecs = [rng.uniform(20, 50, 300) for i in range(net_cnt)]
    f_vecs = [rng.uniform(40, 60, 300) for i in range(net_cnt)]

    # with Pool(processes=4) as pool:
    metalist = metas
    for i, meta in enumerate(metalist):
        print(datetime.now(), f'{i}/{len(metalist)}')
        rez = []
        # for r in pool.map(run_one_net, list(zip(P_vecs, f_vecs, [meta]*net_cnt))):
        for i, args in enumerate(zip(P_vecs, f_vecs, [meta]*net_cnt)):
            # if not i in (0, 3, 5, 7, 10, 13, 15, 18, 21, 23, 27, 29):
            if i != 2:
                continue
            r = run_one_net(args)
            dt, opr = r
            msg = f'{i}     scnds={dt.seconds}, fun={opr.fun}, nfev={opr.nfev}, succ={opr.success}'
            print(msg)
            rez += [r]

        avg_time, avg_fun, avg_its, succ_cnt = 0, 0, 0, 0
        for dt, opr in rez:
            avg_time += dt.seconds
            succ_cnt += opr.success
            avg_fun += opr.fun
            avg_its += opr.nfev
        avg_time /= net_cnt
        avg_fun /= net_cnt
        avg_its /= net_cnt
        msg = f'avg_time={avg_time:.3f}, avg_fun={avg_fun:.3f},  avg_its={avg_its:.3f}, succ_cnt={succ_cnt}/{net_cnt}'
        print(msg)

        for dt, opr in rez:
            msg = f'     scnds={dt.seconds}, fun={opr.fun}, nfev={opr.nfev}, succ={opr.success}'
            print(msg)


def define_mae_for_flows():
    rng = Generator(PCG64(424242))
    P_vec = rng.uniform(20, 50, 300)
    f_vec = rng.uniform(40, 60, 300)
    data_folder = config.get_data_folder()
    dataframe = pd.read_csv(path.join(data_folder, "dns_2_pk (1).csv"))
    G, calc_df, df_to_graph_edges_mapping = make_oilpipe_schema_from_OT_dataset(dataframe, folder=data_folder)
    inlets = [n for n in G.nodes if is_source(G, n)]
    for i, n in enumerate(inlets):
        intake = n + '_pump_intake'
        outlet = n + '_pump_outlet'
        fluid = G.nodes[n]['obj'].fluid
        G.nodes[intake]['obj'] = HE2_Source_Vertex('P', P_vec[i], fluid, 20)
        pump_obj = G[intake][outlet][0]['obj']
        pump_obj.changeFrequency(f_vec[i])
    solver = HE2_Solver(G)
    solver.push_result_to_log = False
    solver.solve(mix_fluids=False, it_limit=1000, threshold=1)

    Qs, Xs = [], []
    for i, n in enumerate(inlets):
        intake = n + '_pump_intake'
        outlet = n + '_pump_outlet'
        Q = G.nodes[intake]['obj'].result['Q']
        pump_obj = G[intake][outlet][0]['obj']
        x = pump_obj.result['x']
        Qs += [Q]
        Xs += [x]
    q1 = np.array(Qs)
    x1 = np.array(Xs)

    solver = HE2_Solver(G)
    solver.push_result_to_log = False
    solver.solve(mix_fluids=False, it_limit=1000, threshold=1000)

    Qs, Xs = [], []
    for i, n in enumerate(inlets):
        intake = n + '_pump_intake'
        outlet = n + '_pump_outlet'
        Q = G.nodes[intake]['obj'].result['Q']
        pump_obj = G[intake][outlet][0]['obj']
        x = pump_obj.result['x']
        Qs += [Q]
        Xs += [x]
    q2 = np.array(Qs)
    x2 = np.array(Xs)
    a = np.concatenate([q1, x1])
    b = np.concatenate([q2, x2])

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    print(mean_absolute_error(a, b))

def generate_tilt_list():
    tilts = []
    for b in list(range(4, 11)) + [12, 14, 17, 20]:
        for d in range(b//4, 3*b//4+1):
            tilts += [dict(tilt_blocked_its_count=b, degradation_rate=d/b)]
    return tilts

def search_for_best_tilt_params():
    rng = Generator(PCG64(42))
    net_cnt, prcs_cnt = 32, 4
    meta = {'steps': np.array([0.2828, 0.5657, 1.1314, 2.2627]), 'border_factor': 5}
    tilt_list = generate_tilt_list()
    P_vecs = [rng.uniform(20, 50, 300) for i in range(net_cnt)]
    f_vecs = [rng.uniform(40, 60, 300) for i in range(net_cnt)]
    rez_file = open('tilt_results.txt', 'r')
    checked_tilt_strs = [s[:-1] for s in rez_file]

    with Pool(processes=4) as pool:
        for i, tilt in enumerate(tilt_list):
            rez_file = open('meta_results.txt', 'a')
            sss = tilt.__str__()
            if sss in checked_tilt_strs:
                continue
            print(tilt, file=rez_file)
            print(datetime.now(), f'                   {i}/{len(tilt_list)}', file=rez_file)
            print(tilt)
            print(datetime.now(), f'                   {i}/{len(tilt_list)}')
            args = list(zip(P_vecs, f_vecs, [meta]*net_cnt))
            rez = []
            while args:
                piece, args = args[:prcs_cnt], args[prcs_cnt:]
                rez += list(pool.map(run_one_net, piece))
                fail_cnt = 0
                for dt, opr in rez:
                    fail_cnt += not opr.success
                if fail_cnt >= 1 + net_cnt // 2:
                    break

            avg_time, avg_fun, avg_its, succ_cnt = 0, 0, 0, 0
            for dt, opr in rez:
                avg_time += dt.seconds
                succ_cnt += opr.success
                avg_fun += opr.fun
                avg_its += opr.nfev
            avg_time /= net_cnt
            avg_fun /= net_cnt
            avg_its /= net_cnt
            msg = f'avg_time={avg_time:.3f}, avg_fun={avg_fun:.3f},  avg_its={avg_its:.3f}, succ_cnt={succ_cnt}/{net_cnt}'
            print(msg)
            print(msg, file=rez_file)

            for dt, opr in rez:
                msg = f'     scnds={dt.seconds}, fun={opr.fun}, nfev={opr.nfev}, succ={opr.success}'
                print(msg)
                print(msg, file=rez_file)
            rez_file.close()

if __name__ == '__main__':
    # define_mae_for_flows()
    test6()