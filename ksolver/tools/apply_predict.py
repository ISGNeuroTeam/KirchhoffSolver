import numpy as np
from ksolver.fluids.HE2_Fluid import HE2_BlackOil
from ksolver.tools.HE2_ABC import oil_params

def apply_predict_2(G, inlets, X, Y):
    residual_vec = np.zeros(len(inlets))
    up_fluid = estimate_mean_fluid(G, inlets, X, Y)
    for i, well_name in enumerate(inlets):
        freq = X[2 * i]
        dyn_height = X[2 * i + 1]
        total_Q = Y[2 * i]
        Q_from_zatrub = Y[2 * i + 1]

        zaboi = f"{well_name}_zaboi"
        intake = f"{well_name}_pump_intake"
        outlet = f"{well_name}_pump_outlet"
        wellhead = f"{well_name}_wellhead"

        down_fluid = G.nodes[well_name]['obj'].fluid
        plast_obj = G[well_name][zaboi][0]['obj']
        x = total_Q - Q_from_zatrub
        fluid = down_fluid if x > 0 else up_fluid
        plast_obj.fluid = fluid
        P_plast = G.nodes[well_name]['obj'].P
        P_zab, T = plast_obj.perform_calc_forward(P_plast, 20.0, x)
        dens = fluid.CurrentLiquidDensity_kg_m3
        WC = fluid.oil_params.volumewater_percent

        result = dict(x=x, liquid_density=dens, WC=WC)
        plast_obj.result = result
        result = dict(P_bar=P_plast, T_C=20.0, Q=x, Q_m3_day=x / dens * 86400)
        G.nodes[well_name]['obj'].result = result
        result = dict(P_bar=P_zab, T_C=20.0)
        G.nodes[zaboi]['obj'].result = result

        # Casing
        casing_obj = G[zaboi][intake][0]['obj']
        casing_obj.fluid = fluid
        P_intake_1, T = casing_obj.perform_calc_forward(P_zab, 20.0, x)
        dens = fluid.CurrentLiquidDensity_kg_m3
        WC = fluid.oil_params.volumewater_percent

        result = dict(x=x, liquid_density=dens, WC=WC)
        casing_obj.result = result
        result = dict(P_bar=P_intake_1, T_C=20.0, Q=Q_from_zatrub, Q_m3_day=Q_from_zatrub / dens * 86400)
        G.nodes[intake]['obj'].result = result

        # 2nd Kirchhoff law residual
        P_intake_2_Pa = dens * 9.81 * dyn_height
        P_intake_2_Bar = P_intake_2_Pa / 100000
        P_intake_2 = P_intake_2_Bar
        residual = abs(P_intake_1 - P_intake_2)
        residual_vec[i] = residual

        # Pump
        x = total_Q
        fluid = down_fluid if x > 0 else up_fluid
        pump_obj = G[intake][outlet][0]['obj']
        pump_obj.fluid = fluid
        if freq == 0:
            pump_obj.state = 'OFF'
        else:
            pump_obj.state = 'ON'
            pump_obj.changeFrequency(freq)
        P_outlet, T = pump_obj.perform_calc_forward(P_intake_1, 20.0, x)
        dens = fluid.CurrentLiquidDensity_kg_m3
        WC = fluid.oil_params.volumewater_percent

        result = dict(x=x, liquid_density=dens, WC=WC, power=pump_obj.power)
        pump_obj.result = result
        result = dict(P_bar=P_outlet, T_C=20.0)
        G.nodes[outlet]['obj'].result = result

        # HKT
        nkt_obj = G[outlet][wellhead][0]['obj']
        nkt_obj.fluid = fluid
        P_wellhead, T = nkt_obj.perform_calc_forward(P_outlet, 20.0, x)
        dens = fluid.CurrentLiquidDensity_kg_m3
        WC = fluid.oil_params.volumewater_percent

        result = dict(x=x, liquid_density=dens, WC=WC)
        nkt_obj.result = result
        result = dict(P_bar=P_wellhead, T_C=20.0)
        G.nodes[wellhead]['obj'].result = result

    total_residual = np.linalg.norm(residual_vec)
    return total_residual


def estimate_mean_fluid(G, inlets, X, Y):
    fluid = None
    total_water_volume, total_liquid_volume = 0, 0
    for i, well_name in enumerate(inlets):
        total_Q = Y[2 * i]
        if total_Q <= 0:
            continue
        fluid = G.nodes[well_name]['obj'].fluid
        WC = fluid.oil_params.volumewater_percent
        dens = fluid.CurrentLiquidDensity_kg_m3
        liquid_volume = total_Q / dens
        water_volume = liquid_volume * WC / 100
        total_liquid_volume += liquid_volume
        total_water_volume += water_volume

    mean_WC_percent = 100 * total_water_volume / total_liquid_volume
    op = fluid.oil_params._replace(volumewater_percent = mean_WC_percent)
    mean_fluid = HE2_BlackOil(op)
    return mean_fluid





