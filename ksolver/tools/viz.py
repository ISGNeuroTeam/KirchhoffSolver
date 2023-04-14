import pandas as pd
from ksolver.graph.nodes.HE2_Vertices import HE2_Boundary_Vertex, HE2_Source_Vertex
import numpy as np
from pyvis.network import Network


def draw_result_graph(dataset, G, use_coordinates=False, coordinate_scaling=3, use_fitted_values = False):
    id_dict = pd.DataFrame()
    id_dict["ids"] = pd.concat((dataset["node_id_start"], dataset["node_id_end"]))
    id_dict["names"] = pd.concat((dataset["node_name_start"], dataset["node_name_end"]))

    if use_coordinates:
        id_dict["Xs"] = pd.concat((dataset["X_start"], dataset["X_end"]))
        id_dict["Ys"] = pd.concat((dataset["Y_start"], dataset["Y_end"]))
        id_dict["Xs"] = id_dict["Xs"] / coordinate_scaling
        id_dict["Ys"] = id_dict["Ys"] / coordinate_scaling

    id_dict = id_dict.drop_duplicates(subset="ids")
    # dataset.to_csv("../CommonData/dataset.csv")

    nt = Network("768px", "1024px", directed=True)
    nt.toggle_physics(False)
    base_size = 15
    if G is not None:
        for n in G.nodes:
            propetries = id_dict.loc[id_dict["ids"] == n]
            obj = G.nodes[n]["obj"]
            if isinstance(obj, HE2_Source_Vertex):
                color = "green" if obj.result['Q'] > 0 else "red"
                size = base_size if obj.result['Q'] > 0 else 2 * base_size
            elif isinstance(obj, HE2_Boundary_Vertex):
                color = "red"
                size = 2 * base_size
            else:
                color = "blue"
                size = base_size * 0.75
            if use_coordinates:
                nt.add_node(
                    n,
                    label=propetries["names"].iloc[0]
                    + f'\n P в узле {np.round(obj.result["P_bar"], decimals=2)}',
                    size=size,
                    title=propetries["names"].iloc[0]
                    + f'<br> P в узле {np.round(G.nodes[n]["obj"].result["P_bar"], decimals=2)}',
                    x=propetries["Xs"].iloc[0],
                    y=propetries["Ys"].iloc[0],
                    color=color,
                )
            else:
                nt.add_node(
                    n,
                    label=propetries["names"].iloc[0]
                    + f'\n P в узле {np.round(obj.result["P_bar"], decimals=2)}',
                    size=size,
                    title=propetries["names"].iloc[0]
                    + f'<br> P в узле {np.round(G.nodes[n]["obj"].result["P_bar"], decimals=2)}',
                    color=color,
                )
    else:
        for n in np.unique(dataset[['node_id_start', 'node_id_end']].to_numpy().flatten()):
            propetries = id_dict.loc[id_dict["ids"] == n]

            rows_start = dataset[dataset['node_id_start'] == n]
            rows_end = dataset[dataset['node_id_end'] == n]
            if True in rows_start['startIsSource'].values:
                color = "green" if rows_start['X_kg_sec'].sum() > 0 else "red"
                size = base_size if rows_start['X_kg_sec'].sum() else 2 * base_size
                node_P = rows_start['startP'].iloc[0]
            elif True in rows_end['endIsOutlet'].values:
                color = "red"
                size = 2 * base_size
                node_P = rows_end['endP'].iloc[0]
            else:
                color = "blue"
                size = base_size * 0.75
                node_P = rows_start['startP'].iloc[0]
            if use_coordinates:
                nt.add_node(
                    n,
                    label=propetries["names"].iloc[0]
                    + f'\n P в узле {np.round(node_P, decimals=2)}',
                    size=size,
                    title=propetries["names"].iloc[0]
                    + f'<br> P в узле {np.round(node_P, decimals=2)}',
                    x=propetries["Xs"].iloc[0],
                    y=propetries["Ys"].iloc[0],
                    color=color,
                )
            else:
                nt.add_node(
                    n,
                    label=propetries["names"].iloc[0]
                    + f'\n P в узле {np.round(node_P, decimals=2)}',
                    size=size,
                    title=propetries["names"].iloc[0]
                    + f'<br> P в узле {np.round(node_P, decimals=2)}',
                    color=color,
                )


    for i, row in dataset.iterrows():
        X_column = 'X_kg_sec' if not use_fitted_values else 'newX'
        start = row["node_id_start"] if row[X_column] > 0 else row["node_id_end"]
        end = row["node_id_end"] if row[X_column] > 0 else row["node_id_start"]
        volumewater = np.round(row["res_watercut_percent"], decimals=2)
        title = (
            f'Расход {np.round(row[X_column] * 86400 / row["res_liquid_density_kg_m3"], decimals=2)} м3/сутки '
            f"<br> Обв. {volumewater} %"
        )
        nt.add_edge(
            start, to=end, title=title, width=base_size / 2
        )  # value = abs(np.round(row["X_kg_sec"] *86400 / row["res_liquid_density_kg_m3"], decimals=2)))
    return nt