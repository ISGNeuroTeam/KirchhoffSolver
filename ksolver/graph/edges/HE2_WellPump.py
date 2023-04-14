from ksolver.tools import HE2_ABC as abc
from ksolver.fluids.HE2_Fluid import gimme_dummy_BlackOil, HE2_DummyWater, HE2_BlackOil
import ksolver.uniflocpy.uTools.uconst as uc
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# from ksolver.tools.HE2_tools import check_for_nan
from ksolver.tools.HE2_Logger import check_for_nan
from logging import getLogger
import scipy.optimize as scop

logger = getLogger(__name__)

A_keff = 0.001
B_keff = 1.4

pumps_cache = None

model_substitute = {
}


def split_pumps_dataframe_to_models(full_HPX: pd.DataFrame):
    if not 'serial_number' in full_HPX.columns:
        full_HPX['serial_number'] = -1
    keys = full_HPX[['pumpModel', 'serial_number']].drop_duplicates()
    pc = dict()
    # for key in keys.to_dict(orient='records'):
    for model, sn in keys.itertuples(index=False):
        mask1 = full_HPX.pumpModel == model
        mask2 = full_HPX.serial_number == sn
        base_HPX = full_HPX[mask1 & mask2].sort_values("debit")
        base_HPX = base_HPX.drop(base_HPX[base_HPX.eff == 0].index)
        p_vec = base_HPX["pressure"].values
        q_vec = base_HPX["debit"].values
        n_vec = base_HPX["power"].values
        eff_vec = base_HPX["eff"].values
        model_curves = dict(p_vec=p_vec, q_vec=q_vec, n_vec=n_vec, eff_vec=eff_vec)
        pc[(model, sn)] = model_curves
    return pc


def create_HE2_WellPump_instance_from_dataframe(
    full_HPX: pd.DataFrame, model="", fluid=None, frequency=50, sn=-1, A=A_keff, B=B_keff
):
    global A_keff, B_keff
    A_keff = A
    B_keff = B

    global pumps_cache, model_substitute
    if pumps_cache is None:
        pumps_cache = split_pumps_dataframe_to_models(full_HPX)
    elif not (model, sn) in pumps_cache:
        pumps_cache.update(split_pumps_dataframe_to_models(full_HPX))

    try:
        if model in model_substitute:
            model = model_substitute[model]
        if frequency <= 0 or np.isnan(frequency):
            frequency = 50
        curves = pumps_cache[(model, sn)]
        pump = HE2_WellPump(**curves, model=model, fluid=fluid, frequency=frequency)
    except Exception as e:
        logger.error(f"Fail to create HE2_WellPump. Model is {model}. Error is '{e}'")
        return None
        # raise e
    return pump


class HE2_WellPump(abc.HE2_ABC_GraphEdge):
    def __init__(self, p_vec, q_vec, n_vec, eff_vec, model="", fluid=None, frequency=50):
        if fluid is None:
            fluid = gimme_dummy_BlackOil()
        self.fluid = fluid
        self.model = model
        if frequency <= 0 or np.isnan(frequency):
            frequency = 50
        self.frequency = frequency
        self._printstr = self.model
        self.base_q = q_vec  # self.base_* - НРХ насоса по воде
        self.base_p = p_vec
        self.base_n = n_vec
        self.base_eff = eff_vec
        self.stages_ratio = 1
        self.state = "on"

        qv, pv, ev, nv = self.extrapolate_to_zeros(q_vec, p_vec, eff_vec, n_vec)
        visc_approx = self.calc_fluid_approx_viscocity()
        vectors = self.rebuild_HPX_curves_to_viscosity(qv, pv, ev, nv, visc_approx)
        self.q_vec_50hz = vectors[0]
        self.p_vec_50hz = vectors[1]
        self.eff_vec_50hz = vectors[2]
        self.n_vec_50hz = vectors[3]

        self.p_vec = self.p_vec_50hz * (self.frequency / 50) ** 2 * self.stages_ratio
        self.n_vec = self.n_vec_50hz * (self.frequency / 50) ** 2

        self.min_Q_50hz = self.q_vec_50hz.min()
        self.max_Q_50hz = self.q_vec_50hz.max()
        self.min_q = self.min_Q_50hz
        self.max_q = self.max_Q_50hz
        self.power = 0
        self.efficiency = 0

        self.get_pressure_raise_1 = None
        self.get_pressure_raise_2 = None
        self.get_pressure_raise_3 = None
        self.make_extrapolators()

    def calc_fluid_approx_viscocity(self):
        if type(self.fluid) == HE2_BlackOil:
            return self.fluid.calc(30, 20, 0).CurrentOilViscosity_Pa_s
        elif type(self.fluid) == HE2_DummyWater:
            return self.fluid.viscocity
        else:
            logger.error(f'Unknown fluid {str(type(self.fluid))}')

    def calc_fluid(self, P_bar, T_C, X_kgsec):
        if type(self.fluid) == HE2_BlackOil:
            return self.fluid.calc(P_bar, T_C, X_kgsec)
        elif type(self.fluid) == HE2_DummyWater:
            return self.fluid
        else:
            logger.error(f'Unknown fluid {str(type(self.fluid))}')

    def extrapolate_to_zeros(self, q_vec, p_vec, eff_vec, n_vec):
        qv = np.array(q_vec).copy()
        pv = np.array(p_vec).copy()
        ev = np.array(eff_vec).copy()
        nv = np.array(n_vec).copy()

        coefs = np.polyfit(qv, pv, 2)
        if coefs[0] > 0:
            coefs = np.polyfit(qv, pv, 1)

        if qv[0] > 0:
            p_zero = np.polyval(coefs, 0)
            # dPdQ = (pv[1] - pv[0]) / (qv[1] - qv[0])
            # p_zero = pv[0] - dPdQ * qv[0]
            dNdQ = (nv[1] - nv[0]) / (qv[1] - qv[0])
            n_zero = nv[0] - dNdQ * qv[0]
            if n_zero <= 0:
                n_zero = nv[0]

            qv = np.concatenate(([0], qv))
            pv = np.concatenate(([p_zero], pv))
            nv = np.concatenate(([n_zero], nv))
            ev = np.concatenate(([0], ev))
        if pv[-1] > 0:
            q_last = 2 * (qv[-1] - qv[0]) # Suppose Q_max = 2*Q_nom, and Q_nom is middle of q_vec
            xs = scop.fsolve(lambda x: np.polyval(coefs, x), q_last)
            q_last = max(xs)
            dNdQ = (nv[-1] - nv[-2]) / (qv[-1] - qv[-2])
            n_last = nv[-1] + dNdQ * (q_last - qv[-1])
            if n_last <= 0:
                n_last = nv[-1]

            qv = np.concatenate((qv, [q_last]))
            pv = np.concatenate((pv, [0]))
            nv = np.concatenate((nv, [n_last]))
            ev = np.concatenate((ev, [0]))

        return qv, pv, ev, nv



    def rebuild_HPX_curves_to_viscosity(self, q_vec, p_vec, eff_vec, n_vec, viscosity_Pa_s):
        if type(self.fluid) is HE2_DummyWater:
            n_vec_ = (q_vec[1:-1] * 9.81 * 1000 / 86400) * p_vec[1:-1] / (eff_vec[1:-1] / 100)
            n_vec[1:-1] = n_vec_
            return q_vec, p_vec, eff_vec, n_vec

        pseudo_vec = (
            1.95
            * 0.04739
            * viscosity_Pa_s ** 0.5
            * (p_vec / 0.3048) ** 0.25739
            * (q_vec / 0.227) ** 0.25
        )
        Cq_vec = np.polyval(
            [0.9873, 0.009019, -0.0016233, 0.00007233, -0.0000020258, 0.000000021009][
                ::-1
            ],
            pseudo_vec,
        )
        Cp_vec = np.polyval(
            [
                1.0045,
                -0.002664,
                -0.00068292,
                0.000049706,
                -0.0000016522,
                0.000000019172,
            ][::-1],
            pseudo_vec,
        )
        Ceff_vec = np.polyval(
            [1.0522, -0.03512, -0.00090394, 0.00022218, -0.00001198, 0.00000019895][
                ::-1
            ],
            pseudo_vec,
        )
        q_vec_50hz = q_vec * Cq_vec
        p_vec_50hz = p_vec * Cp_vec
        eff_vec_50hz = eff_vec * Ceff_vec
        n_vec_50hz = n_vec.copy()
        n_vec_ = (q_vec_50hz[1:-1] * 9.81 * 1000 / 86400) * p_vec_50hz[1:-1] / (eff_vec_50hz[1:-1] / 100)
        n_vec_50hz[1:-1] = n_vec_
        return q_vec_50hz, p_vec_50hz, eff_vec_50hz, n_vec_50hz

    def __str__(self):
        return self._printstr

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        assert unifloc_direction in [0, 1, 10, 11]
        calc_direction = 1 if unifloc_direction >= 10 else -1
        flow_direction = 1 if unifloc_direction % 10 == 1 else -1

        if calc_direction == 1:
            return self.perform_calc_forward(P_bar, T_C, X_kgsec)
        else:
            return self.perform_calc_backward(P_bar, T_C, X_kgsec)

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        fl = self.calc_fluid(P_bar, T_C, X_kgsec)
        p, t = self.calculate_pressure_differrence(p, t, X_kgsec, 1, fl)
        return p, t

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        fl = self.calc_fluid(P_bar, T_C, X_kgsec)
        p, t = self.calculate_pressure_differrence(p, t, X_kgsec, -1, fl)
        return p, t

    def calculate_pressure_differrence(
        self, P_bar, T_C, X_kgsec, calc_direction, mishenko, unifloc_direction=-1
    ):
        check_for_nan(P_bar=P_bar, T_C=T_C, X_kgsec=X_kgsec)
        self.power = 0
        self.efficiency = 5
        if self.state.upper() == "OFF":
            T_rez_C = T_C
            P_rez_bar = P_bar - X_kgsec * 100500

            return P_rez_bar, T_rez_C

        # Определяем направления расчета
        liquid_debit = X_kgsec * 86400 / mishenko.CurrentLiquidDensity_kg_m3
        grav_sign, fric_sign, t_sign = self.decode_direction(X_kgsec, calc_direction, unifloc_direction)
        if liquid_debit <= self.min_q:
            get_pressure_raise = self.get_pressure_raise_1
        elif (self.min_q < liquid_debit) and (liquid_debit < self.max_q):
            get_pressure_raise = self.get_pressure_raise_2
            self.power = self.n_interpolator(liquid_debit) * 1.0
            self.efficiency = self.get_eff_2(liquid_debit) * 1.0
        else:
            get_pressure_raise = self.get_pressure_raise_3

        pressure_raise = get_pressure_raise(liquid_debit) * 9.81 * mishenko.CurrentLiquidDensity_kg_m3

        P_rez_bar = P_bar + calc_direction * uc.Pa2bar(pressure_raise)

        # T_rez_C = T_C + calc_direction * self.calculate_temperature_raise(pressure_raise, eff, mishenko)
        T_rez_C = T_C
        check_for_nan(P_rez_bar=P_rez_bar, T_rez_C=T_rez_C)

        return P_rez_bar, T_rez_C

    def decode_direction(self, flow, calc_direction, unifloc_direction):
        """
        :param unifloc_direction - направление расчета и потока относительно  координат.
            11 расчет и поток по координате
            10 расчет по координате, поток против
            00 расчет и поток против координаты
            00 расчет против координаты, поток по координате
            unifloc_direction перекрывает переданные flow, calc_direction
        """
        flow_direction = np.sign(flow) if flow != 0 else 1
        if unifloc_direction in [0, 1, 10, 11]:
            calc_direction = 1 if unifloc_direction >= 10 else -1
            flow_direction = 1 if unifloc_direction % 10 == 1 else -1

        assert calc_direction in [-1, 1]
        grav_sign = calc_direction
        fric_sign = flow_direction * calc_direction
        t_sign = calc_direction
        return grav_sign, fric_sign, t_sign

    def make_extrapolators(self):
        if self.state.upper() == 'OFF':
            return
        freq_k = self.frequency / 50
        q_vec = self.q_vec_50hz * freq_k
        min_q = self.min_Q_50hz * freq_k
        max_q = self.max_Q_50hz * freq_k
        p_vec = self.stages_ratio * self.p_vec_50hz * freq_k ** 2
        n_vec = self.n_vec_50hz * freq_k ** 3
        eff_vec = self.eff_vec_50hz  # sic!
        zero_head = p_vec[0]
        last_head = p_vec[-1]

        self.get_pressure_raise_1 = (lambda x: zero_head + A_keff * abs(min_q - x) ** B_keff)
        self.get_pressure_raise_2 = interp1d(q_vec, p_vec, kind="quadratic")
        self.get_pressure_raise_3 = lambda x: last_head - A_keff * (x - max_q) ** B_keff

        self.get_eff_2 = interp1d(q_vec, eff_vec, kind="quadratic")
        self.n_interpolator = interp1d(q_vec, n_vec, kind="quadratic")

        self.min_q = min_q
        self.max_q = max_q

    def changeFrequency(self, new_frequency):
        self.frequency = new_frequency
        if new_frequency == 0:
            self.state = 'Off'
        self.make_extrapolators()

    def change_stages_ratio(self, new_ratio):
        self.stages_ratio = new_ratio
        self.make_extrapolators()