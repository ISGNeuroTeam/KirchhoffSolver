import pandas as pd

from ksolver.io.calculate_DF import calculate_DF


def get_solver_error(
    dataframe,
    data_folder,
    diams,
    verification_values: pd.DataFrame,
    use_pressures=True,
    flow_weight=1,
    pressure_weight=1,
    relative_error=True,
):
    total_error = 1000
    diams = diams if len(diams) != 1 else diams[0]
    dataframe["effectiveD"] = diams
    mdf = calculate_DF(dataframe, data_folder)

    if not "X_kg_sec" in mdf.columns:
        return total_error
    total_error = calculate_error(
        mdf,
        verification_values,
        relative_error=relative_error,
        kind="flow",
        flow_weight=flow_weight,
        pressure_weight=pressure_weight,
    )
    if use_pressures:
        total_error += calculate_error(
            mdf,
            verification_values,
            relative_error=relative_error,
            kind="pressure",
            pressure_weight=pressure_weight,
        )

    """
    for node_id in verification_start_points['node_id_start'].unique():
        calc_flow_verification_values.update({node_id: verification_start_points[verification_start_points["node_id_start"] == node_id]['X_kg_sec'].sum()})

    for node_id in verification_end_points['node_id_end'].unique():
        calc_flow_verification_values.update({node_id: verification_end_points[verification_end_points["node_id_end"] == node_id]['X_kg_sec'].sum()})

    total_error = 0
    for key in list(flow_verification_values.keys()):
        total_error+= abs(flow_verification_values[key] - flow_verification_values[key]) if not relative_error else abs(calc_verification_values[key] - real_verification_values[key]) / real_verification_values[key]
"""

    return total_error


def get_boundary_flows(dataframe):
    inlets = dataframe[dataframe["startIsSource"]][
        ["node_id_start", "X_kg_sec", "startP"]
    ]
    inlets = inlets.rename(columns={"node_id_start": "node_id", "startP": "pressure"})
    outlets = dataframe[dataframe["endIsOutlet"]][["node_id_end", "X_kg_sec", "endP"]]
    outlets = outlets.rename(columns={"node_id_end": "node_id", "endP": "pressure"})
    boundaries = pd.concat((inlets, outlets), axis=0)
    return (
        boundaries.groupby("node_id")
        .sum()
        .reset_index()
        .rename(columns={"X_kg_sec": "liquid_debit"})
    )


def calculate_error(
    mdf,
    verification_values,
    relative_error=True,
    kind="flow",
    flow_weight=1,
    pressure_weight=1,
):
    if kind == "flow":
        column = "liquid_debit" if kind == "flow" else "pressure"
        start_column = "X_kg_sec" if kind == "flow" else "startP"
        end_column = "X_kg_sec" if kind == "flow" else "endP"
        renamed_column = "X_kg_sec" if kind == "flow" else "calc_pressure"
        weight = flow_weight
    else:
        column = "pressure"
        start_column = "startP"
        end_column = "endP"
        renamed_column = "calc_pressure"
        weight = pressure_weight

    flow_verification_points = verification_values.dropna(subset=[column], axis=0)[
        ["node_id", column]
    ]
    flow_verification_nodes_list = flow_verification_points["node_id"]

    verification_start_points = mdf[
        mdf["node_id_start"].isin(flow_verification_nodes_list)
    ][["node_id_start", start_column]]
    verification_start_points = (
        verification_start_points.groupby("node_id_start")
        .sum()
        .reset_index()
        .rename(columns={"node_id_start": "node_id", start_column: renamed_column})
    )
    verification_start_points = verification_start_points.merge(
        flow_verification_points, on=["node_id"], how="left"
    )

    verification_end_points = mdf[
        mdf["node_id_end"].isin(flow_verification_nodes_list)
    ][["node_id_end", end_column]]
    verification_end_points = (
        verification_end_points.groupby("node_id_end")
        .sum()
        .reset_index()
        .rename(columns={"node_id_end": "node_id", end_column: renamed_column})
    )
    verification_end_points = verification_end_points.merge(
        flow_verification_points, on=["node_id"], how="left"
    )

    full_errors_df = pd.concat(
        (verification_start_points, verification_end_points), axis=0
    ).drop_duplicates(subset=["node_id"])
    if not full_errors_df.empty:
        if relative_error:
            total_error = (
                abs(full_errors_df[column] - full_errors_df[renamed_column])
                / full_errors_df[renamed_column]
            ).sum()
        else:
            total_error = abs(
                full_errors_df[column] - full_errors_df[renamed_column]
            ).sum()
    else:
        total_error = 0
    return total_error


def fix_column_types(dataset):
    dataset[
        [
            "startValue",
            "VolumeWater",
            "D",
            "L",
            "S",
            "X_end",
            "X_start",
            "Y_end",
            "Y_start",
            "altitude_end",
            "altitude_start",
            "effectiveD",
            "intD",
            "part_L",
            "roughness",
            "uphillM",
            "endValue",
            "startT",
        ]
    ] = dataset[
        [
            "startValue",
            "VolumeWater",
            "D",
            "L",
            "S",
            "X_end",
            "X_start",
            "Y_end",
            "Y_start",
            "altitude_end",
            "altitude_start",
            "effectiveD",
            "intD",
            "part_L",
            "roughness",
            "uphillM",
            "endValue",
            "startT",
        ]
    ].astype(
        float
    )
    return dataset
