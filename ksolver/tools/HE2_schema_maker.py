import networkx as nx
import pandas as pd
import numpy as np
from os import path
from ksolver.graph.edges.HE2_Pipe import HE2_WaterPipe, HE2_OilPipe
from ksolver.graph.edges.HE2_Choke import HE2_Choke
from ksolver.graph.edges.HE2_Plast import HE2_Plast
from ksolver.graph.edges.HE2_WellPump import create_HE2_WellPump_instance_from_dataframe
from ksolver.graph.nodes import HE2_Vertices as vrtxs
from ksolver.fluids.HE2_Fluid import gimme_BlackOil, gimme_dummy_BlackOil, HE2_DummyWater, gimme_dummy_oil_params, HE2_BlackOil
from logging import getLogger
from ksolver.tools.HE2_ABC import oil_params

logger = getLogger(__name__)

pump_curves = None
inclination = None
HKT = None


def handle_error(exception_class, msg, df=None):
    # TODO: log'em all
    raise exception_class(msg)


def make_waterpipe_schema_from_edges_and_nodes_df(df_nodes, df_edges):
    dfs = dict(nodes=df_nodes, edges=df_edges)
    rez = make_waterpipe_schema_from_dfs(dfs, row_type_col='juncType')
    return rez


def get_pad_well_name(name):
    try:
        rez = str(int(name))
    except:
        rez = name
    return rez


def make_waterpipe_schema_from_dfs(dfs, row_type_col="row_type"):
    df_nodes = dfs['nodes']
    df_edges = dfs['edges']
    df_kns_pumps = dfs.get('kns_pumps', None)
    df_well_pumps = dfs.get('well_pumps', None)
    df_inclination = dfs.get('inclination', None)
    df_HKT = dfs.get('HKT', None)

    outlets = {}
    inlets = {}
    juncs = {}

    inlets_df, outletsdf, juncs_df = split_nodes_df_to_groups(df_nodes)

    for row in inlets_df.itertuples():
        fluid = HE2_DummyWater()
        obj = vrtxs.HE2_Source_Vertex(row.kind, row.value, fluid, row.T)
        inlets.update({row.node_name: obj})

    for row in outletsdf.itertuples():
        obj = vrtxs.HE2_Boundary_Vertex(row.kind, row.value)
        obj.fluid = HE2_DummyWater()
        outlets.update({row.node_name: obj})

    for row in juncs_df.itertuples():
        obj = vrtxs.HE2_ABC_GraphVertex()
        juncs.update({row.node_name: obj})

    flag1 = len(set(inlets.keys()) & set(outlets.keys())) != 0
    flag2 = len(set(juncs.keys()) & set(outlets.keys())) != 0
    flag3 = len(set(inlets.keys()) & set(juncs.keys())) != 0
    if flag1 or flag2 or flag3:
        handle_error(ValueError, f'ambiguous kind of node')

    G = nx.MultiDiGraph()  # Di = directed
    df_to_graph_edges_mapping = dict()

    for k, v in {**inlets, **outlets, **juncs}.items():
        G.add_node(k, obj=v)

    read_roughness = 'roughness' in df_edges
    if not 'serial_number' in df_edges:
        df_edges['serial_number'] = -1

    for row in df_edges.itertuples():
        start = row.node_name_start
        end = row.node_name_end
        row_type = row.row_type
        if row_type in ("pipe", 'kns_pump'):
            fluid = HE2_DummyWater()
            if row_type == "pipe":
                L = row.L
                uphill = row.altitude_end - row.altitude_start
                diam_coef = row.effectiveD
                D = row.intD
                roughness = 1e-5
                if read_roughness:
                    roughness = row.roughness
                obj = HE2_WaterPipe([L], [uphill], [D * diam_coef], [roughness])
                obj.fluid = fluid
            else:  # row_type == 'kns_pump'
                model = row.pump_model
                frequency = row.pump_frequency
                sn = row.serial_number
                pump = create_HE2_WellPump_instance_from_dataframe(full_HPX=df_kns_pumps, model=model, fluid=fluid, frequency=frequency, sn=sn)
                obj = pump
            k = G.add_edge(start, end, obj=obj)  # edge index in MultiDiGraph. k in G[u][v][k] index
            df_to_graph_edges_mapping[row.Index] = [(start, end, k)]

        elif row_type == 'water_production_well':
            wellNum = get_pad_well_name(row.well_num)
            padNum = get_pad_well_name(row.pad_num)
            pump_outlet_node = f"PAD_{padNum}_WELL_{wellNum}_pump_outlet"
            model = row.pump_model
            frequency = row.pump_frequency
            fluid = HE2_DummyWater()
            pump = create_HE2_WellPump_instance_from_dataframe(full_HPX=df_well_pumps, model=model, fluid=fluid, frequency=frequency)

            pumpdepth = row.pump_depth
            D, L, roughness, uphill = get_pipe_from_pump_to_wellhead(df_HKT, df_inclination, pumpdepth, wellNum)

            nkt_pipe = HE2_WaterPipe([L], [uphill], [D], [roughness])

            obj = vrtxs.HE2_ABC_GraphVertex()
            G.add_node(pump_outlet_node, obj=obj)
            k = G.add_edge(start, pump_outlet_node, obj=pump)  # edge index in MultiDiGraph. k in G[u][v][k] index
            df_to_graph_edges_mapping[row.Index] = [(start, pump_outlet_node, k)]
            k = G.add_edge(pump_outlet_node, end, obj=nkt_pipe)  # edge index in MultiDiGraph. k in G[u][v][k] index
            df_to_graph_edges_mapping[row.Index] += [(pump_outlet_node, end, k)]
        elif row_type == 'water_production_well':
            pass
        else:
            msg = f"unknown type of graph edge in dataset. Row_type is {row_type}. Start, end id is {start} {end}. "
            msg += "Cannot add this edge to graph, skip"
            logger.warning(msg)
            continue

    cmpnts = nx.algorithms.components.number_weakly_connected_components(G)
    if cmpnts != 1:
        logger.error(f"Not single componented graph!")
        raise ValueError

    return G, df_to_graph_edges_mapping


def get_pipe_from_pump_to_wellhead(df_HKT, df_inclination, pumpdepth, wellNum):
    tubing = df_inclination[df_inclination["wellNum"] == wellNum]
    tubing = tubing.sort_values(by="depth")
    local_HKT = df_HKT[df_HKT["wellNum"] == wellNum]
    if tubing.empty or local_HKT.empty:
        uphill = pumpdepth
        L = pumpdepth
        D = 0.057
        roughness = 1e-5
        return D, L, roughness, uphill

    fulldepth = 0
    stages = local_HKT["stageNum"].unique()
    for stageNum in stages:
        stage_HKT = local_HKT[local_HKT["stageNum"] == stageNum]
        stage_HKT = stage_HKT[stage_HKT["_time"] == stage_HKT["_time"].max()]
        fulldepth += stage_HKT["stageLength"].iloc[0]
        tubing.loc[tubing["depth"] <= fulldepth, "IntDiameter"] = (stage_HKT["stageDiameter"].iloc[0] - 16) / 1000
    tubing.loc[tubing["IntDiameter"].isna(), "IntDiameter"] = 0.057
    pump_place = tubing[abs(tubing["depth"] - pumpdepth) == min(abs(tubing["depth"] - pumpdepth))].iloc[0]
    uphill = pump_place["absMark"]
    L = pump_place["prolongation"]
    D = pump_place["IntDiameter"]
    roughness = 1e-5
    return D, L, roughness, uphill


def gimme_nodes_columns_in_dataframe(df, columns):
    start_cols, end_cols = [], []
    df_cols = set(df.columns)
    for suff, lst in [('start', start_cols), ('end', end_cols)]:
        for col in columns:
            v1 = f'{col}_{suff}'
            v2 = f'{suff}_{col}'
            if len({v1, v2} & df_cols) > 1:
                handle_error(IndexError, f'ambigious column names: {v1} and {v2}')
            if v1 in df_cols:
                lst += [v1]
            elif v2 in df_cols:
                lst += [v2]
            else:
                handle_error(IndexError, f'Columns {v1} or {v2} are expected but not found')

    return start_cols, end_cols


def collapse_multiple_node_rows(nodes_df):
    cnts = nodes_df.value_counts(subset=['node_name'], sort=False)
    cnts = cnts[cnts > 1]
    bad_names = [t[0] for t in cnts.index.values]

    cols = list(set(nodes_df.columns) - {'node_name'})
    for name in bad_names:
        for col in cols:
            mask = nodes_df.node_name == name
            values = set(nodes_df[mask][col].dropna().unique())
            values = list(values - {'', None, np.nan})
            if len(values) > 1:
                msg = f'Cannot collapse multiple node in one. Node_name {name}, column {col}, ambigous values {values}'
                handle_error(NotImplementedError, msg, cnts)

            value = np.nan if len(values) == 0 else values[0]
            nodes_df.loc[mask, col] = value

    return nodes_df


def gimme_nodes_dataframe(full_df):
    # node_columns_full = ['node_id', 'node_type', 'X', 'Y', 'node_name', 'altitude']
    node_columns_full = ['node_id', 'node_name', 'altitude', 'kind', 'q', 'p', 'is_source', 'T', 'value']
    # columns_short = ['rs_schema_id', "pipeline_id", "part_id", "thread_number"]
    # columns_short += [col + suff for col in node_columns_full for suff in ['_start', '_end']]
    start_cols, end_cols = gimme_nodes_columns_in_dataframe(full_df, node_columns_full)

    dfs = []
    for cols in [start_cols, end_cols]:
        _df = full_df[cols]
        # _df = _df.drop_duplicates()
        # if len(_df.node_name_start.unique()) != len(_df):
        #     cnts = _df.value_counts(subset=['node_name_start'], sort=False)
        #     cnts = cnts[cnts > 1]
        #     pass
        _df.columns = node_columns_full
        dfs += [_df]
    nodes_df = pd.concat(dfs).drop_duplicates()

    # Так, тут проблема. Мне нужно чтобы node_name было PK. Но если на один node_name есть несколько записей с разными значениями тупла, то drop_duplicates() не спасает
    # И нужно эти несколько записей как-то превращать в одну
    # Еще нужно воткнуть строгую проверку количества узлов в исходном и в конечном датасетах

    if len(nodes_df.node_name.unique()) != len(nodes_df):
        nodes_df = collapse_multiple_node_rows(nodes_df)

    mask = nodes_df.kind == ''
    nodes_df.kind[mask] = np.nan

    return nodes_df


def split_nodes_df_to_groups(nodes_df):
    src_df, snk_df, jnc_df = None, None, None
    # Check some conditions
    # 1. All source nodes are bound nodes
    # 2.All bound nodes must have one of P or Q

    src_mask = nodes_df['is_source'] == True
    jncs_mask = nodes_df['kind'].isna()
    bnds_mask = ~jncs_mask
    if 'p' in nodes_df and 'q' in nodes_df:
        p_mask = ~nodes_df['p'].isna()
        q_mask = ~nodes_df['q'].isna()

        pq_mask = p_mask | q_mask
        bad_bnds_mask = bnds_mask & ~pq_mask
        if bad_bnds_mask.any():
            handle_error(ValueError, 'Boundary nodes must have value for P or Q', nodes_df[bad_bnds_mask])

        bad_sources_mask = src_mask & ~bnds_mask
        if bad_sources_mask.any():
            handle_error(ValueError, 'All source nodes must be boundary nodes', nodes_df[bad_bnds_mask])

        if not 'value' in nodes_df:
            kind_p_mask = nodes_df['kind'] == 'P'
            p_mask = ~nodes_df['p'].isna()
            p_mask = p_mask & kind_p_mask

            kind_q_mask = nodes_df['kind'] == 'Q'
            q_mask = ~nodes_df['q'].isna()
            q_mask = q_mask & kind_q_mask

            nodes_df['value'] = np.nan
            nodes_df.loc[p_mask, 'value'] = nodes_df.loc[p_mask, 'p']
            nodes_df.loc[q_mask, 'value'] = nodes_df.loc[q_mask, 'q']

    snk_mask = bnds_mask & ~src_mask

    src_df = nodes_df[src_mask]
    snk_df = nodes_df[snk_mask]
    jncs_df = nodes_df[jncs_mask]
    if len(src_df) + len(snk_df) + len(jncs_df) != len(nodes_df):
        handle_error(ValueError, 'Something wrong with dataset splitting')

    return src_df, snk_df, jncs_df


def preprocess(df):
    '''Add 'value column if not exists, renaming columns etc'''
    for suff in ['start', 'end']:
        for col, col_replace in {'Value': 'value', 'Kind': 'kind', 'T': 'T', 'IsSource': 'is_source'}.items():
            if f'{suff}{col}' in df.columns:
                df = df.rename(columns={f'{suff}{col}': f'{suff}_{col_replace}'})
        if not f'{suff}_value' in df.columns:
            p_mask = ~df[f'p_{suff}'].isna()
            q_mask = ~df[f'q_{suff}'].isna()
            df[f'{suff}_value'] = np.nan
            df.loc[p_mask, f'{suff}_value'] = df.loc[p_mask, f'p_{suff}']
            df.loc[q_mask, f'{suff}_value'] = df.loc[q_mask, f'q_{suff}']
        else:
            if not f'p_{suff}' in df.columns:
                df[f'p_{suff}'] = np.nan
                kind_mask = df[f'{suff}_kind'] == 'P'
                p_mask = ~df[f'{suff}_value'].isna()
                df.loc[p_mask & kind_mask, f'p_{suff}'] = df.loc[p_mask & kind_mask, f'{suff}_value']
            if not f'q_{suff}' in df.columns:
                df[f'q_{suff}'] = np.nan
                kind_mask = df[f'{suff}_kind'] == 'Q'
                q_mask = ~df[f'{suff}_value'].isna()
                df.loc[q_mask & kind_mask, f'q_{suff}'] = df.loc[q_mask & kind_mask, f'{suff}_value']

        if not f'{suff}_T' in df.columns:
            df[f'{suff}_T'] = 20

    if not 'effectiveD' in df.columns:
        df['effectiveD'] = 0.85
    if not 'roughness' in df.columns:
        df['roughness'] = 1e-5
    if not 'node_id_start' in df.columns:
        df['node_id_start'] = df['node_name_start']
    if not 'node_id_end' in df.columns:
        df['node_id_end'] = df['node_name_end']


    return df


def make_waterpipe_schema_from_OT_dataset(dataset, folder="./data", calc_df=None):
    outlets = {}
    inlets = {}
    juncs = {}

    if calc_df is None:
        calc_df = dataset

    calc_df = preprocess(calc_df)

    nodes_df = gimme_nodes_dataframe(calc_df)
    inlets_df, outletsdf, juncs_df = split_nodes_df_to_groups(nodes_df)

    for row in inlets_df.itertuples():
        fluid = HE2_DummyWater()
        obj = vrtxs.HE2_Source_Vertex(row.kind, row.value, fluid, row.T)
        inlets.update({row.node_name: obj})

    for row in outletsdf.itertuples():
        obj = vrtxs.HE2_Boundary_Vertex(row.kind, row.value)
        outlets.update({row.node_name: obj})

    for row in juncs_df.itertuples():
        obj = vrtxs.HE2_ABC_GraphVertex()
        juncs.update({row.node_name: obj})

    flag1 = len(set(inlets.keys()) & set(outlets.keys())) != 0
    flag2 = len(set(juncs.keys()) & set(outlets.keys())) != 0
    flag3 = len(set(inlets.keys()) & set(juncs.keys())) != 0
    if flag1 or flag2 or flag3:
        handle_error(ValueError, f'ambiguous kind of node')

    G = nx.MultiDiGraph()  # Di = directed
    df_to_graph_edges_mapping = dict()

    for k, v in {**inlets, **outlets, **juncs}.items():
        G.add_node(k, obj=v)

    for row in calc_df.itertuples():
        start = row.node_name_start
        end = row.node_name_end
        row_type = row.row_type
        if row_type == "pipe":
            L = row.L
            uphill = row.altitude_end - row.altitude_start
            if uphill * 0 != 0:  # checking for float nan
                uphill = 0
            diam_coef = row.effectiveD
            D = row.intD
            roughness = row.roughness
            obj = HE2_WaterPipe([L], [uphill], [D * diam_coef], [roughness])
            obj.fluid = HE2_DummyWater()
        elif row_type == "injection_well":
            L = 30
            uphill = 0
            diam_coef = 1
            D = 0.1
            roughness = 1e-5
            obj = HE2_WaterPipe([L], [uphill], [D * diam_coef], [roughness])
            obj.fluid = HE2_DummyWater()
        elif row_type == "choke":
            d_choke = row.choke_diam
            d_pipe = row.intD
            obj = HE2_Choke(diam=d_choke, d_pipe=d_pipe, fluid=HE2_DummyWater())
        else:
            msg = f"unknown type of graph edge in dataset. Row_type is {row_type}. Start, end id is {start} {end}. "
            msg += "Cannot add this edge to graph, skip"
            logger.warning(msg)
            continue

        k = G.add_edge(start, end, obj=obj)  # edge index in MultiDiGraph. k in G[u][v][k] index
        df_to_graph_edges_mapping[row.Index] = (start, end, k)

    cmpnts = nx.algorithms.components.number_weakly_connected_components(G)
    if cmpnts != 1:
        logger.error(f"Not single componented graph!")
        raise ValueError

    return G, calc_df, df_to_graph_edges_mapping


def get_base_oil_params_from_df(dataset: pd.DataFrame):
    op = gimme_dummy_oil_params()
    fluid_columns = ['Oil_density_kg_m3', 'Water_density_kg_m3', 'Gas_factor_m3_m3', 'Formation_pressure_bar']
    fl_set = set(fluid_columns) & set(dataset.columns)
    if len(fl_set) != len(fluid_columns):
        return op
    fluids_df = dataset[fluid_columns].drop_duplicates().dropna()
    if len(fluids_df) != 1:
        return op
    od, wd, gf, pp = list(fluids_df.itertuples(index=False))[0]
    op_dict = op._asdict()
    op_dict.update(sat_P_bar=pp, oildensity_kg_m3=od, waterdensity_kg_m3=wd, gasFactor=gf)
    rez = oil_params(**op_dict)
    return rez


def make_oilpipe_schema_from_OT_dataset(dataset, folder="./data", calc_df=None, ignore_Watercut=False, row_type_col='juncType', need_check=True):
    global pump_curves
    if pump_curves is None:
        pump_curves = pd.read_csv(path.join(folder, "PumpChart.csv"))

    dataset_cols = set(dataset.columns)
    well_cols = ["perforation", "frequency", "wellNum", "padNum"]
    well_cols += ["pumpDepth", "productivity", "model"]
    for col in well_cols:
        if not col in dataset_cols:
            dataset[col] = None

    if calc_df is None:
        calc_df = make_calc_df_oil(dataset, folder, row_type_col=row_type_col, need_check=need_check)

    base_op = get_base_oil_params_from_df(calc_df)

    outlets = {}
    inlets = {}
    juncs = {}

    inlets_df = calc_df[calc_df["startIsSource"]]
    outletsdf = calc_df[calc_df["endIsOutlet"]]
    juncs_df = pd.concat((calc_df["node_id_start"], calc_df["node_id_end"])).unique()
    for row in inlets_df.itertuples():
        if ignore_Watercut:
            volumewater = 50
        elif pd.notna(row.VolumeWater):
            volumewater = row.VolumeWater
        else:
            logger.warning(f'Watercut should be known for source nodes: {row["node_id_start"]}')
            volumewater = 50

        if row.startKind == 'PQCurve':
            start_value = np.genfromtxt(row.startValue, delimiter=',')
            start_t = np.genfromtxt(row.startT, delimiter=',')
            volumewater, WC_Q_Curve = 50, np.genfromtxt(row.VolumeWater, delimiter=',')
        else:
            start_value = row.startValue
            start_t = row.startT
            WC_Q_Curve = None

        fluid = gimme_BlackOil(base_op, volumewater_percent=volumewater)
        source_id = row.node_id_start
        source_obj = vrtxs.HE2_Source_Vertex(row.startKind, start_value, fluid, start_t, WC_Q_Curve=WC_Q_Curve)
        inlets.update({source_id: source_obj})

    for row in outletsdf.itertuples():
        sink_id = row.node_id_end
        sink_obj = vrtxs.HE2_Boundary_Vertex(row.endKind, row.endValue)
        outlets.update({sink_id : sink_obj})

    for id in juncs_df:
        if id not in inlets and id not in outlets:
            juncs.update({id: vrtxs.HE2_ABC_GraphVertex()})

    G = nx.MultiDiGraph()  # Di = directed
    df_to_graph_edges_mapping = dict()

    for k, v in {**inlets, **outlets, **juncs}.items():
        G.add_node(k, obj=v)

    col_indexes = {col: idx for idx, col in enumerate(list(calc_df), start=1)}
    row_type_idx = col_indexes[row_type_col]

    for row in calc_df.itertuples():
        start = row.node_id_start
        end = row.node_id_end
        row_type = row[row_type_idx]
        VolumeWater = row.VolumeWater if pd.notna(row.VolumeWater) and row.startKind != 'PQCurve' else 50
        fluid = gimme_BlackOil(base_op, volumewater_percent=VolumeWater)
        if row_type == "pipe":
            L = row.L
            uphill = row.uphillM
            diam_coef = row.effectiveD
            D = row.intD
            roughness = row.roughness
            obj = HE2_OilPipe([L], [uphill], [D * diam_coef], [roughness], fluid)
        elif row_type == "plast":
            productivity = row.productivity
            obj = HE2_Plast(productivity=productivity, fluid=fluid)
        elif row_type == "wellpump":
            model = row.model
            frequency = row.frequency
            if model not in pump_curves['pumpModel']:
                model = replace_pump_model_with_closest(model, pump_curves)
            pump = create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=model, fluid=fluid, frequency=frequency)
            if "K_pump" in calc_df.columns:
                K_pump = row.K_pump
                if K_pump > 0 and K_pump < 100500:
                    pump.change_stages_ratio(K_pump)
            obj = pump
        else:
            msg = f"unknown type of graph edge in dataset. start, end id is {start} {end}. Cannot add this edge to graph, skip"
            logger.warning(msg)
            continue

        k = G.add_edge(start, end, obj=obj)  # edge index in MultiDiGraph. k in G[u][v][k] index
        df_to_graph_edges_mapping[row.Index] = (start, end, k)

    cmpnts = nx.algorithms.components.number_weakly_connected_components(G)
    if cmpnts != 1:
        logger.error(f"Not single componented graph!")
        raise ValueError

    return G, calc_df, df_to_graph_edges_mapping


def make_calc_df_oil(dataset, folder, row_type_col='juncType', need_check=True):
    global inclination, HKT
    if inclination is None:
        inclination = pd.read_parquet(
            path.join(folder, "inclination"), engine="pyarrow"
        )
    if HKT is None:
        HKT = pd.read_parquet(path.join(folder, "HKT"), engine="pyarrow")
    wells_df = dataset[(dataset[row_type_col] == "oilwell") | (dataset[row_type_col] == "projectwell")]
    dataset = dataset[(dataset[row_type_col] != "oilwell") & (dataset[row_type_col] != "projectwell")]
    dataset[["node_id_start", "node_id_end"]] = (
        dataset[["node_id_start", "node_id_end"]].astype(str)
    )
    dict_list = []
    tempdf = populate_wells_df(HKT, dict_list, inclination, wells_df, row_type_col)
    dataset = dataset.append(tempdf)
    calc_df = dataset
    calc_df["startIsSource"] = calc_df["startIsSource"].fillna(False)
    calc_df["endIsOutlet"] = calc_df["endIsOutlet"].fillna(False)
    calc_df[["node_id_start", "node_id_end"]] = calc_df[
        ["node_id_start", "node_id_end"]
    ].astype(str)

    if need_check:
        ids_count = pd.concat(
            (calc_df["node_id_start"], calc_df["node_id_end"])
        ).value_counts()
        ids_count.rename("ids_count")
        calc_df = calc_df.join(ids_count.to_frame(), on="node_id_start", how="left")
        calc_df = calc_df.rename(columns={0: "start_id_count"})

        ids_count = calc_df["node_id_start"].value_counts()
        ids_count.rename("ids_count")
        calc_df = calc_df.join(
            ids_count.to_frame().rename(columns={"node_id_start": 0}),
            on="node_id_end",
            how="left",
        )
        calc_df = calc_df.rename(columns={0: "end_id_count"})
        calc_df["end_id_count"] = calc_df["end_id_count"].fillna(0)
        calc_df["sourceByCount"] = calc_df["start_id_count"] == 1
        calc_df["outletByCount"] = calc_df["end_id_count"] == 0
        calc_df["sourceMistakes"] = calc_df["sourceByCount"] == calc_df["startIsSource"]
        calc_df["sourceMistakes"] = True
        calc_df["outletMistakes"] = calc_df["outletByCount"] == calc_df["endIsOutlet"]
        calc_df["sourceValueIsFilled"] = pd.notna(calc_df["startValue"])
        calc_df["outletValueIsFilled"] = pd.notna(calc_df["endValue"])
        calc_df["sourceKindIsFilled"] = pd.notna(calc_df["startKind"])
        calc_df["outletKindIsFilled"] = pd.notna(calc_df["endKind"])
        calc_df["inletBoundaryMistakes"] = True
        calc_df["outletBoundaryMistakes"] = True
        calc_df.loc[calc_df["startIsSource"], "inletBoundaryMistakes"] = (
                calc_df[calc_df["startIsSource"]]["sourceValueIsFilled"]
                & calc_df[calc_df["startIsSource"]]["sourceKindIsFilled"]
        )
        calc_df.loc[calc_df["endIsOutlet"], "outletBoundaryMistakes"] = (
                calc_df[calc_df["endIsOutlet"]]["outletValueIsFilled"]
                & calc_df[calc_df["endIsOutlet"]]["outletKindIsFilled"]
        )
        calc_df["sumOfCounts"] = calc_df["start_id_count"] + calc_df["end_id_count"]
        calc_df = calc_df[calc_df["sumOfCounts"] >= 2]
        mistakes_df = calc_df[
            (~calc_df["sourceMistakes"])
            | (~calc_df["outletMistakes"])
            | (~calc_df["inletBoundaryMistakes"])
            | (~calc_df["outletBoundaryMistakes"])

            ]
        if not mistakes_df.empty:
            print(
                f"Following nodes: {mistakes_df[~mistakes_df['sourceMistakes']]['node_id_start'].values} should be sources"
            )
            print(
                f"Following nodes: {mistakes_df[~mistakes_df['outletMistakes']]['node_id_start'].values} should be outlets"
            )
            print(
                f"Start kind and value for following nodes: {mistakes_df[~mistakes_df['inletBoundaryMistakes']]['node_id_start'].values} should be filled"
            )
            print(
                f"End kind and value for following nodes: {mistakes_df[~mistakes_df['outletBoundaryMistakes']]['node_id_end'].values} should be filled"
            )
            assert False

    calc_df = calc_df.reset_index()  # Damn you, pandas black emperor!
    return calc_df


def replace_pump_model_with_closest(model, pump_curves):
    nominal_params = pump_curves[pump_curves['debit'] == pump_curves['NominalQ']]
    split_model = model.split("-")
    try:
        nominal_Q = float(split_model[1])
    except:
        nominal_Q = float(split_model[1].split('/')[0])
    try:
        nominal_H = float(split_model[-1])
    except:
        try:
            nominal_H = float(split_model[-1].split('(')[0])
        except:
            nominal_H = float(split_model[-1].split(')')[-2])
    closest_Q = nominal_params[abs(nominal_params['debit'] - nominal_Q) == min(abs(nominal_params['debit'] - nominal_Q))]
    closest_H = closest_Q[abs(closest_Q['pressure'] - nominal_H) == min(abs(closest_Q['pressure'] - nominal_H))]
    replacement_model = closest_H['pumpModel'].iloc[0]
    return replacement_model



def populate_wells_df(HKT, dict_list, inclination, wells_df, row_type_col='juncType'):
    for i, row in wells_df.iterrows():
        dict_list = process_well_row(HKT, dict_list, inclination, row, row_type_col)
    tempdf = pd.DataFrame.from_dict(dict_list)
    return tempdf


def process_well_row(HKT, dict_list, inclination, row, row_type_col='juncType'):
    try:
        wellNum = str(int(row.wellNum))
    except:
        wellNum = row['wellNum']
    if pd.isna(row['wellNum']):
        wellNum = np.random.randint(15000, 1000000)
    # padNum = str(int(row["padNum"]))
    # col_indexes = {col: idx for idx, col in enumerate(list(calc_df), start=1)}
    # row_type_idx = col_indexes[row_type_col]

    if row[row_type_col] == 'projectwell':
        padNum = row['padNum']
        pump_abs_depth = row['pump_abs_depth']
        pump_prolongation = row['pump_prolongation']
        perforation_abs_depth = row['perforation_abs_depth']
        perforation_prolongation = row['perforation_prolongation']
        dict_list = make_well_parts_rows_no_tubing(
            dict_list, padNum, pump_abs_depth, pump_prolongation, perforation_abs_depth, perforation_prolongation, row,
            wellNum, row_type_col
        )
        return dict_list

    padNum = row.padNum
    pumpdepth = row.pumpDepth
    perforation = row.perforation
    tubing = inclination[inclination["wellNum"] == wellNum]
    if tubing.empty:
        dict_list = make_well_parts_rows_no_tubing(
            dict_list, padNum, pumpdepth, 10, perforation, 10, row,
            wellNum, row_type_col
        )
        return dict_list
    tubing = tubing.sort_values(by="depth")
    tubing["Roughness"] = 3e-5
    tubing["IntDiameter"] = 0.057
    tubing["NKTlength"] = 10
    local_HKT = HKT[HKT["wellNum"] == wellNum]
    fulldepth = 0
    stages = local_HKT["stageNum"].unique()
    for stageNum in stages:
        stage_HKT = local_HKT[local_HKT["stageNum"] == stageNum]
        stage_HKT = stage_HKT[stage_HKT["_time"] == stage_HKT["_time"].max()]
        fulldepth += stage_HKT["stageLength"].iloc[0]
        tubing.loc[tubing["depth"] <= fulldepth, "IntDiameter"] = (stage_HKT["stageDiameter"].iloc[0] - 16) / 1000
    pump_place = tubing[abs(tubing["depth"] - pumpdepth) == min(abs(tubing["depth"] - pumpdepth))].iloc[0]
    perforation_place = tubing[abs(tubing["depth"] - perforation) == min(abs(tubing["depth"] - perforation))].iloc[0]

    dict_list = make_well_parts_rows(dict_list, padNum, perforation_place, pump_place, row, wellNum, row_type_col)
    return dict_list


def make_well_parts_rows_no_tubing(
        dict_list, padNum, pump_abs_depth, pump_prolongation, perforation_abs_depth, perforation_prolongation, row, wellNum, row_type_col='juncType'
):
    row_d = row.to_dict()
    # plast-zaboi
    dct = row_d.copy()

    dct[row_type_col] = "plast"
    dct["effectiveD"] = 1
    dct["roughness"] = 3e-5
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_zaboi"
    dct["endIsOutlet"] = False
    dict_list += [dct]
    # zaboi-intake
    dct = dct.copy()
    dct["startIsSource"] = False
    dct[row_type_col] = "pipe"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_zaboi"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_pump_intake"
    absdiff = perforation_abs_depth - pump_abs_depth
    Ldiff = perforation_prolongation - pump_prolongation
    dct["L"] = (Ldiff ** 2 + absdiff ** 2) ** 0.5
    dct["uphillM"] = absdiff
    dct["intD"] = 0.127
    dict_list += [dct]
    # wellpump
    dct = dct.copy()
    dct[row_type_col] = "wellpump"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_pump_intake"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_pump_outlet"
    dict_list += [dct]
    # outlet-wellhead
    dct = dct.copy()
    dct[row_type_col] = "pipe"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_pump_outlet"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_wellhead"
    absdiff = pump_abs_depth
    Ldiff = pump_prolongation
    dct["L"] = (Ldiff ** 2 + absdiff ** 2) ** 0.5
    dct["uphillM"] = absdiff
    dct["intD"] = 0.057
    dict_list += [dct]
    # wellhead-pad
    dct = dct.copy()
    dct[row_type_col] = "pipe"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_wellhead"
    dct["node_id_end"] = row["node_id_end"]
    dct["endIsOutlet"] = row["endIsOutlet"]
    dct["L"] = 50
    dct["uphillM"] = 0
    dct["intD"] = 0.1
    dict_list += [dct]
    return dict_list


def make_well_parts_rows(
        dict_list, padNum, perforation_place, pump_place, row, wellNum, row_type_col='juncType'
):
    row_d = row.to_dict()
    # plast-zaboi
    dct = row_d.copy()

    dct[row_type_col] = "plast"
    dct["effectiveD"] = 1
    dct["roughness"] = 3e-5
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_zaboi"
    dct["endIsOutlet"] = False
    dict_list += [dct]
    # zaboi-intake
    dct = dct.copy()
    dct["startIsSource"] = False
    dct[row_type_col] = "pipe"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_zaboi"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_pump_intake"
    absdiff = perforation_place["absMark"] - pump_place["absMark"]
    Ldiff = perforation_place["depth"] - pump_place["depth"]
    absdiff = absdiff if absdiff != 0 else 10
    Ldiff = Ldiff if Ldiff != 0 else 10
    dct["L"] = Ldiff
    dct["uphillM"] = absdiff
    dct["intD"] = 0.127
    dict_list += [dct]
    # wellpump
    dct = dct.copy()
    dct[row_type_col] = "wellpump"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_pump_intake"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_pump_outlet"
    dict_list += [dct]
    # outlet-wellhead
    dct = dct.copy()
    dct[row_type_col] = "pipe"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_pump_outlet"
    dct["node_id_end"] = f"PAD_{padNum}_WELL_{wellNum}_wellhead"
    absdiff = pump_place["absMark"]
    Ldiff = pump_place["depth"]
    dct["L"] = Ldiff
    dct["uphillM"] = absdiff
    dct["intD"] = pump_place["IntDiameter"]
    dict_list += [dct]
    # wellhead-pad
    dct = dct.copy()
    dct[row_type_col] = "pipe"
    dct["node_id_start"] = f"PAD_{padNum}_WELL_{wellNum}_wellhead"
    dct["node_id_end"] = row["node_id_end"]
    dct["endIsOutlet"] = row["endIsOutlet"]
    dct["L"] = 50
    dct["uphillM"] = 0
    dct["intD"] = 0.1
    dict_list += [dct]
    return dict_list


def make_schema_from_OISPipe_dataframes(df_pipes, df_boundaries):
    df = df_pipes[["node_id_start", "node_id_end"]]
    df.columns = ["source", "target"]
    G = nx.from_pandas_edgelist(df, create_using=nx.DiGraph)
    edge_list = list(G.edges())
    edge_set = set(edge_list)
    rev_set = set([(v, u) for u, v in edge_set])
    fin_edge_set = edge_set - rev_set
    G = nx.DiGraph(fin_edge_set)

    cmpnts = nx.algorithms.components.number_connected_components(
        nx.Graph(fin_edge_set)
    )
    if cmpnts != 1:
        print("Not single component graph!")
        assert False

    pipes = dict()
    for u, v in G.edges():
        df = df_pipes
        df = df[(df.node_id_start == u) & (df.node_id_end == v)]
        d = df.iloc[0].to_dict()

        Ls = [d["L"]]
        Hs = [d["altitude_end"] - d["altitude_start"]]
        Ds = [d["D"] - 2 * d["S"]]
        Rs = [1e-5]

        pipe = HE2_WaterPipe(Ls, Hs, Ds, Rs)
        pipes[(u, v)] = pipe
    nx.set_edge_attributes(G, name="obj", values=pipes)

    df = df_boundaries[["Q", "P"]].fillna(-1e9)
    df_boundaries["value"] = df.max(axis=1)

    nodes = dict()
    id_list = list(df_boundaries.id.values)
    for n in G.nodes():
        df = df_boundaries
        if n in id_list:
            df = df[df.id == n]
            d = df.iloc[0].to_dict()
        else:
            obj = vrtxs.HE2_ABC_GraphVertex()
            nodes[n] = obj
            continue
        # if d['kind']=='P':
        #     print(d)

        if d["is_source"]:
            obj = vrtxs.HE2_Source_Vertex(d["kind"], d["value"], "water", 20)
        elif d["kind"] == "Q" and ((d["Q"] is None) or d["Q"] == 0):
            obj = vrtxs.HE2_ABC_GraphVertex()
        else:
            obj = vrtxs.HE2_Boundary_Vertex(d["kind"], d["value"])
        nodes[n] = obj

    nx.set_node_attributes(G, name="obj", values=nodes)

    for n in G.nodes():
        o = G.nodes[n]["obj"]
        assert o is not None

    for u, v in G.edges():
        o = G[u][v]["obj"]
        assert o is not None

    return G


def make_multigraph_schema_from_OISPipe_dataframes(df_pipes, df_boundaries):
    df = df_pipes[["node_id_start", "node_id_end"]]
    df["idx_for_result"] = df.index
    df.columns = ["source", "target", "idx_for_result"]
    G = nx.from_pandas_edgelist(
        df, create_using=nx.MultiDiGraph, edge_attr="idx_for_result"
    )

    cmpnts = nx.algorithms.components.number_connected_components(nx.Graph(G))
    if cmpnts != 1:
        print("Not single component graph!")
        assert False

    pipes = dict()
    for u, v, k in G.edges:
        df = df_pipes
        df = df[(df.node_id_start == u) & (df.node_id_end == v)]
        d = df.iloc[k].to_dict()

        Ls = [d["L"]]
        Hs = [d["altitude_end"] - d["altitude_start"]]
        Ds = [d["D"] - 2 * d["S"]]
        Rs = [1e-5]

        pipe = HE2_WaterPipe(Ls, Hs, Ds, Rs)
        pipes[(u, v, k)] = pipe
    nx.set_edge_attributes(G, name="obj", values=pipes)

    mask = df_boundaries.kind == "Q"
    df_boundaries["value"] = -1e9
    df_boundaries.loc[mask, "value"] = df_boundaries.loc[mask, "Q"]
    df_boundaries.loc[~mask, "value"] = df_boundaries.loc[~mask, "P"]

    nodes = dict()
    id_list = list(df_boundaries.id.values)
    for n in G.nodes():
        df = df_boundaries
        if n in id_list:
            df = df[df.id == n]
            d = df.iloc[0].to_dict()
        else:
            obj = vrtxs.HE2_ABC_GraphVertex()
            nodes[n] = obj
            continue
        # if d['kind']=='P':
        #     print(d)

        if d["is_source"]:
            obj = vrtxs.HE2_Source_Vertex(d["kind"], d["value"], "water", 20)
        elif d["kind"] == "Q" and ((d["Q"] is None) or d["Q"] == 0):
            obj = vrtxs.HE2_ABC_GraphVertex()
        else:
            obj = vrtxs.HE2_Boundary_Vertex(d["kind"], d["value"])
        nodes[n] = obj

    nx.set_node_attributes(G, name="obj", values=nodes)

    for n in G.nodes():
        o = G.nodes[n]["obj"]
        assert o is not None

    for u, v, k in G.edges:
        o = G[u][v][k]["obj"]
        assert o is not None

    return G
