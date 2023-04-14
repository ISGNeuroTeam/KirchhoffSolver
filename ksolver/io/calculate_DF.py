from ksolver.solver.HE2_Solver import HE2_Solver
from ksolver.tools.HE2_schema_maker import make_oilpipe_schema_from_OT_dataset, make_waterpipe_schema_from_OT_dataset
from ksolver.tools.HE2_schema_maker import make_waterpipe_schema_from_edges_and_nodes_df, make_waterpipe_schema_from_dfs
from ksolver.tools.HE2_tools import check_solution, print_solution
from logging import getLogger
import pandas as pd
import numpy as np
from pandasql import sqldf

logger = getLogger(__name__)


def ad_hoc_convert_ppd_two_df_to_single_df(df_dict):

    def drop_not_used_columns(n_df, e_df):
        e_cols = ['node_name_start', 'node_name_end', '__pad_num', 'adku_pad_name', 'X_end', 'X_start', 'Y_end', 'Y_start', 'condition']
        e_cols += ['external_coating_type_name', 'internal_coating_type_name', 'part_id', 'pipe_material_name', 'pipeline_id', 'rs_schema_name']
        e_cols += ['simple_part_creation_date', 'simple_part_id']
        n_cols = ['rs_schema_name']

        n_cols += ['__pad_num']
        e_cols += ['altitude_diff', 'p_coll']

        n_df = n_df.drop(columns = n_cols)
        e_df = e_df.drop(columns = e_cols)
        return n_df, e_df

    def add_some_columns(n_df, e_df):
        n_df['IsOutlet'] = np.nan
        return n_df, e_df

    def rename_columns(n_df, e_df):
        n_df = n_df.rename(columns={'kind':'Kind', 'value':'Value', 'is_source':'IsSource'})
        return n_df, e_df

    def join_dfs(n_df, e_df):
        ncols = set(list(n_df.columns)) - set(['node_id'])
        n_pref_cols = ncols and set(['Kind', 'Value', 'T', 'IsSource', 'IsOutlet'])
        n_suff_cols = ncols - n_pref_cols

        query = '''
            Select E.*, N.* 
            from E left join N on E.node_id_start = N.node_id
        '''
        df = sqldf(query, dict(N=n_df, E=e_df))
        df = df.drop(columns=['node_id'])
        df = df.rename(columns={k:f'start{k}' for k in n_pref_cols})
        df = df.rename(columns={k:f'{k}_start' for k in n_suff_cols})

        query = '''
            Select E.*, N.* 
            from E left join N on E.node_id_end = N.node_id
        '''
        df = sqldf(query, dict(N=n_df, E=df))
        df = df.drop(columns=['node_id'])
        df = df.rename(columns={k:f'end{k}' for k in n_pref_cols})
        df = df.rename(columns={k:f'{k}_end' for k in n_suff_cols})

        return df

    n_df = df_dict['nodes']
    e_df = df_dict['edges']

    n_df, e_df = drop_not_used_columns(n_df, e_df)
    n_df, e_df = add_some_columns(n_df, e_df)
    n_df, e_df = rename_columns(n_df, e_df)
    rez_df = join_dfs(n_df, e_df)
    return rez_df


def ad_hoc_join_wells_to_ppd_single_df(wells_df, net_df):

    def drop_not_used_columns(w_df, n_df):
        w_cols = ['rs_schema_name']
        n_cols = []
        n_df = n_df.drop(columns=n_cols)
        w_df = w_df.drop(columns=w_cols)
        return w_df, n_df

    wells_df, net_df = drop_not_used_columns(wells_df, net_df)
    # в ppd_df node_id и node_name - одно и тоже. В скважинах отличаются.
    wells_df["node_id_end"] = wells_df["node_name_end"]
    wells_df["node_id_start"] = wells_df["node_name_start"]
    wells_df["endKind"] = 'Q'
    wells_df = wells_df.rename(columns={'rowType':'row_type'})
    wells_df['row_type'] = 'injection_well'

# 1. Построить множество кустов скважин.
    pads_set = set(list(wells_df["node_name_start"].unique()))
# 2. На всех этих кустах сбросить граничное условие
    pads_mask = net_df['node_name_end'].isin(pads_set)
    net_df.loc[pads_mask, "endKind"] = np.nan
    net_df.loc[pads_mask, "endValue"] = np.nan
    net_df.loc[pads_mask, "endIsOutlet"] = np.nan

    pads_mask = net_df['node_name_start'].isin(pads_set)
    net_df.loc[pads_mask, "startKind"] = np.nan
    net_df.loc[pads_mask, "startValue"] = np.nan
    net_df.loc[pads_mask, "startIsSource"] = np.nan

    columns_to_add = set(net_df.columns) - set(wells_df.columns)
    for col in columns_to_add:
        wells_df[col] = np.nan

    columns_to_add = set(wells_df.columns) - set(net_df.columns)
    for col in columns_to_add:
        net_df[col] = np.nan

    nwo = ['padNum', 'wellNum', 'row_type']
    nwo += ['node_id_start', 'node_name_start', 'X_start', 'Y_start', 'altitude_start']
    nwo += ['node_id_end', 'node_name_end', 'X_end', 'Y_end', 'altitude_end']
    nwo += ['startKind', 'startValue', 'startT', 'startIsSource', 'startIsOutlet']
    nwo += ['endKind', 'endValue', 'endT',  'endIsSource', 'endIsOutlet']
    nwo += ['L', 'd', 's','intD', 'effectiveD']
    nwo += ['choke_diam']
    others = set(wells_df.columns) - set(nwo)
    nwo += list(others)

    net_df = net_df[nwo]
    wells_df = wells_df[nwo]

    query = '''
        Select * from N 
        Union all
        Select * from W
    '''

    rez_df = df = sqldf(query, dict(N=net_df, W=wells_df))

    return rez_df


def calculate_PPD_from_dfs(df_dict, solver_params = None):
    df_nodes = df_dict['nodes']
    df_edges = df_dict['edges']
    G, df_to_graph_edges_mapping = make_waterpipe_schema_from_dfs(df_dict)
    solver = HE2_Solver(G)
    if solver_params is None:
        solver_params = dict(threshold=0.5)
    solver.solve(mix_fluids=False, **solver_params)
    df_nodes_rez, df_edges_rez = None, None
    if solver.op_result.success:
        df_nodes_rez = put_results_to_nodes_dataframe(G, df_nodes, df_to_graph_edges_mapping)
        df_edges_rez = put_results_to_edges_dataframe(G, df_edges, df_to_graph_edges_mapping)
    return df_nodes_rez, df_edges_rez, G



def calculate_PPD_from_nodes_edges_df(df_nodes, df_edges):
    G, df_to_graph_edges_mapping = make_waterpipe_schema_from_edges_and_nodes_df(df_nodes, df_edges)
    solver = HE2_Solver(G)
    solver.solve(threshold=0.5, mix_fluids=False)
    df_nodes_rez, df_edges_rez = None, None
    if solver.op_result.success:
        df_nodes_rez = put_results_to_nodes_dataframe(G, df_nodes, df_to_graph_edges_mapping)
        df_edges_rez = put_results_to_edges_dataframe(G, df_edges, df_to_graph_edges_mapping)
    return df_nodes_rez, df_edges_rez, G


def calculate_DF(dataframe, data_folder="./data", return_graph=False, check_results=False, network_kind = 'oil', solver_params = None, row_type_col='juncType', check_input_df=True):
    # G, calc_df, df_to_graph_edges_mapping = None, None, None
    if network_kind == 'oil':
        G, calc_df, df_to_graph_edges_mapping = make_oilpipe_schema_from_OT_dataset(dataframe, folder=data_folder, row_type_col=row_type_col, need_check=check_input_df)
    elif network_kind == 'water':
        G, calc_df, df_to_graph_edges_mapping = make_waterpipe_schema_from_OT_dataset(dataframe, folder=data_folder)
    else:
        raise IndexError

    cols_to_drop = ['index', 'start_id_count', 'end_id_count', 'sourceByCount', 'outletByCount', 'sourceMistakes', 'outletMistakes', 'sourceValueIsFilled']
    cols_to_drop += ['outletValueIsFilled', 'sourceKindIsFilled', 'outletKindIsFilled', 'inletBoundaryMistakes', 'outletBoundaryMistakes', 'sumOfCounts']
    cols_to_drop = set(cols_to_drop) & set(calc_df.columns)
    calc_df = calc_df.drop(columns=cols_to_drop)

    need_mix_fluids = network_kind == 'oil'
    if solver_params is None:
        solver_params = dict(threshold=0.5)

    solver = HE2_Solver(G)
    solver.solve(mix_fluids=need_mix_fluids, **solver_params)
    if solver.op_result.success:
        if check_results:
            validity = check_solution(G)
            #        print(validity)
            #        print_solution(G)

        calc_df = put_result_to_dataframe(G, calc_df, df_to_graph_edges_mapping, network_kind)

    if return_graph:
        return (calc_df, G)
    else:
        return calc_df


def put_results_to_nodes_dataframe(G, df_nodes, df_to_graph_edges_mapping, network_kind = 'water'):
    df = df_nodes.copy()
    df['result_P'] = None
    df['result_Q'] = None
    for n in G.nodes:
        rez = G.nodes[n]["obj"].result
        df.loc[df["node_name"] == n, "result_P_bar"] = rez["P_bar"]
        if 'Q_m3_day' in rez:
            df.loc[df["node_name"] == n, "result_Q_m3_day"] = rez["Q_m3_day"]

    return df

def put_results_to_edges_dataframe(G, df_edges, df_to_graph_edges_mapping, network_kind = 'water'):
    df = df_edges.copy()
    df["res_X_kg_sec"] = None
    df["velocity_m_sec"] = None
    df["res_pump_power_watt"] = None
    df["res_wellpump_outlet_pressure_bar"] = None

    for row in df.itertuples():
        start = row.node_name_start
        end = row.node_name_end
        rec = df_to_graph_edges_mapping[row.Index]
        u, v, k = rec[0]

        if start != u:
            logger.error("broken df_to_graph_edges_mapping, or calc_df.index")
            logger.error(f"{start}, {end}, {u}, {v}, {k}")
            raise IndexError

        rez = G[u][v][k]["obj"].result

        df.loc[row.Index, "res_X_kg_sec"] = rez["x"]
        df.loc[row.Index, "res_liquid_density_kg_m3"] = rez["liquid_density"]
        Q = rez["x"] / rez["liquid_density"]
        Area = 3.1415926 * row.intD ** 2 / 4
        V = Q / Area
        df.loc[row.Index, "velocity_m_sec"] = V
        if 'power' in rez:
            df.loc[row.Index, "res_pump_power_watt"] = rez['power']
        if row.row_type == 'water_production_well':
            node_obj = G.nodes[v]['obj']
            df.loc[row.Index, "res_wellpump_outlet_pressure_bar"] = node_obj.result['P_bar']

    return df

def put_result_to_dataframe(G, calc_df, df_to_graph_edges_mapping, network_kind = 'oil'):
    for n in G.nodes:
        rez = G.nodes[n]["obj"].result
        calc_df.loc[calc_df["node_id_start"] == n, "startP"] = rez["P_bar"]
        calc_df.loc[calc_df["node_id_start"] == n, "startT"] = rez["T_C"]
        calc_df.loc[calc_df["node_id_end"] == n, "endP"] = rez["P_bar"]
        calc_df.loc[calc_df["node_id_end"] == n, "endT"] = rez["T_C"]
        if 'Q_m3_day' in rez:
            calc_df.loc[calc_df["node_id_start"] == n, "start_Q_m3_day"] = rez["Q_m3_day"]
            calc_df.loc[calc_df["node_id_end"] == n, "end_Q_m3_day"] = rez["Q_m3_day"]

    calc_df["res_watercut_percent"] = None
    calc_df["res_liquid_density_kg_m3"] = None
    calc_df["res_pump_power_watt"] = None
    for index, row in calc_df.iterrows():
        start = row["node_id_start"]
        end = row["node_id_end"]
        u, v, k = df_to_graph_edges_mapping[index]
        if start != u or end != v:
            logger.error("broken df_to_graph_edges_mapping, or calc_df.index")
            logger.error(f"{start}, {end}, {u}, {v}, {k}")
            raise IndexError

        rez = G[u][v][k]["obj"].result
        calc_df.loc[index, "X_kg_sec"] = rez["x"]
        calc_df.loc[index, "res_watercut_percent"] = rez["WC"]

        calc_df.loc[index, "res_liquid_density_kg_m3"] = rez["liquid_density"]
        Q = rez["x"] / rez["liquid_density"]
        Area = 3.1415926 * row["intD"] ** 2 / 4
        V = Q / Area
        calc_df.loc[index, "velocity_m_sec"] = V
        if 'power' in rez:
            calc_df.loc[index, "res_pump_power_watt"] = rez['power']

    return calc_df
