import pandas as pd
from ksolver.solver.HE2_Solver import HE2_Solver
from ksolver.tools.HE2_schema_maker import make_oilpipe_schema_from_OT_dataset
from logging import getLogger

logger = getLogger(__name__)
from ksolver.graph.nodes.HE2_Vertices import HE2_Source_Vertex
from ksolver.tools.HE2_ABC import Root

# dataframe = pd.read_csv("data/DNS2_with_wells.csv")
dataframe = pd.read_csv("data/DNS_wells.csv")
G1, calc_df, df_to_graph_edges_mapping = make_oilpipe_schema_from_OT_dataset(dataframe)
for node in G1.nodes:
    if "_pump_intake" in node:
        well_num = int(node.split("_")[3])
        input_pressure = dataframe[dataframe["wellNum"] == well_num]["input_pressure"].values[0]
        obj = HE2_Source_Vertex(
            kind="P",
            value=input_pressure,
            fluid=fluid,
            T=20,
        )
        G1.nodes[node]["obj"] = obj

solver = HE2_Solver(G1)
solver.solve(threshold=0.5)
