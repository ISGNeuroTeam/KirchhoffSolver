from ksolver.tools import HE2_ABC as abc
from ksolver.tools.HE2_Logger import getLogger
from ksolver.fluids.HE2_Fluid import HE2_DummyWater
from math import pi as PI

logger = getLogger(__name__)


class HE2_Choke(abc.HE2_ABC_GraphEdge):
    def __init__(self, diam=5, d_pipe=70, fluid=None, calibr_fr=1, K=0.826, eps=0.001, max_iter=50):
        self.K = K  # Discharge coefficient (optional, default  is 0.826)
        self.diam = diam  # choke diameter
        self.fluid = fluid or HE2_DummyWater()  # fluid in choke
        self.calibr_fr = calibr_fr  # calibre coefficient
        self.eps = eps
        self.max_iter = max_iter
        self.d_pipe = d_pipe  # pipe diameter

    @property
    def diam(self):
        return self._diam

    @diam.setter
    def diam(self, value):
        if value <= 0:
            logger.error(f"Choke diameter = {value}")
        self._diam = value

    def __str__(self):
        return f"Choke diameter: {self._diam}"

    def __repr__(self):
        return self.__str__()

    def calc_q(self, p_u, p_d, t_u):
        """
        Function calculates oil flow rate through choke given downstream and upstream pressures using Perkins correlation
        :param p_u: Upstream pressure ( (atma))
        :param p_d: Downstream pressure ( (atma))
        :param t_u: Upstream temperature ( (C))
        :return: oil flow rate through choke ((sm3/day))
        """
        # # calc PVT with upstream pressure and temperature and oil parameters
        # fluid = PVT_calc(p_u, t_u, self.fluid)
        a_c = PI * self.diam ** 2 / 4  # choke throat area
        a_u = PI * self.d_pipe ** 2 / 4  # upstream area
        a_r = a_c / a_u  # area ratio
        # Calculate trial output choke pressure
        p_co = p_u - (p_u - p_d) / (1 - (self.diam / self.d_pipe) ** 1.85)
        # critical pressure ratio under fm_gas_fr = 0
        p_ri = 0
        # Compare trial pressure ratio with critical and assign actual pressure ratio
        p_ra = max(p_ri, p_co / p_u)
        w_i = self.wi_calc(p_ra, p_u, t_u, a_c, a_r)
        # Calculate isentropic mass flow rate
        w = self.K * w_i * self.calibr_fr

        # fluid = PVT_calc(p_u, t_u, self.fluid)
        # calc_choke_qliq_sm3day = w * fluid.fm_oil_fr / fluid.PVT_rho_oil_kgm3 + w * fluid.fm_wat_fr / fluid.PVT_rho_wat_kgm3

        # add constants for water, for fluid need to calculate them
        fm_oil_fr = 0
        fm_wat_fr = 1
        rho_oil_kgm3 = 780
        rho_wat_kgm3 = 1012
        # calculate oil flow rate through choke
        calc_choke_qliq_sm3day = w * fm_oil_fr / rho_oil_kgm3 + w * fm_wat_fr / rho_wat_kgm3
        return calc_choke_qliq_sm3day / 10 ** 6

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        """
        function to calculate pressure and temperature with given direction
        :param P_bar: given  pressure
        :param T_C: given  temperature
        :param X_kgsec: given oil rate
        :param unifloc_direction: first digit is direction for calculation, second digit-fluid direction (00, 01, 10, 11)
        :return: tuple pressure and temperature
        """
        assert unifloc_direction in [0, 1, 10, 11]
        calc_direction = 1 if unifloc_direction >= 10 else -1
        flow_direction = 1 if unifloc_direction % 10 == 1 else -1
        if calc_direction == 1:
            return self.perform_calc_forward(P_bar, T_C, flow_direction * abs(X_kgsec))
        else:
            return self.perform_calc_backward(P_bar, T_C, flow_direction * abs(X_kgsec))


    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        """
        function to calculate pressure and temperature
        :param P_bar: given  pressure
        :param T_C: given  temperature
        :param X_kgsec: given oil rate
        :return: tuple pressure and temperature
        """
        q = X_kgsec
        max_iter = self.max_iter
        p_sn = P_bar
        t_u = T_C
        eps = q * 0.0001
        eps_p = 0.001

        i = 1
        for counter in range(0, max_iter + 1):
            i = 2 * i
            P_en_max = p_sn * i
            q_l = self.calc_q(P_en_max, p_sn, t_u)
            if q_l > q:
                break
        if q_l <= q:
            logger.error(
                f"Calc_p_forward: no solution found, 1"
            )
            return 0
        P_en_min = i * p_sn / 2
        j = 0
        for counter in range(0, max_iter + 1):
            j = counter
            p_en = (P_en_min + P_en_max) / 2
            q_l = self.calc_q(p_en, p_sn, t_u)
            if q_l > q:
                P_en_max = p_en
            else:
                P_en_min = p_en
            q_good = abs(q - q_l) < eps
            p_good = abs(P_en_min - P_en_max) < eps_p
            if q_good and p_good:
                break
        if j > max_iter:
            logger.error(
                f"Calc_p_forward: no solution found, 2"
            )
            return 0
        return p_en, T_C

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        """
        function to calculate pressure and temperature
        :param P_bar: given  pressure
        :param T_C: given  temperature
        :param X_kgsec: given oil rate
        :return: tuple pressure and temperature
        """
        max_iter = self.max_iter
        p_sn = P_bar
        t_u = T_C
        q = X_kgsec
        eps = q * 0.0001
        # eps_p = 0.001

        q_l = self.calc_q(p_sn, 0, t_u)
        i = 1
        for counter in range(0, max_iter + 1):
            i = 2 * i
            P_en_min = p_sn / i
            q_l = self.calc_q(p_sn, P_en_min, t_u)
            if q_l > q:
                break
        if q_l <= q:
            logger.error(
                f"Calc_p_backward: no solution found, 1"
            )
            return 0
        P_en_max = 2 * p_sn / i
        j = 0
        for counter in range(0, max_iter + 2):
            j = counter
            p_en = (P_en_min + P_en_max) / 2
            q_l = self.calc_q(p_sn, p_en, t_u)
            if q_l > q:
                P_en_min = p_en
            else:
                P_en_max = p_en
            if abs(q - q_l) < eps:
                break
        if j > max_iter:
            logger.error(
                f"Calc_p_backward: no solution found, 1"
            )
            return 0
        return p_en, T_C

    def calc_diam(self, val_min, val_max, q_0, p_in, p_out, t_in):
        d_pipe = self.d_pipe
        d = 0
        eps = self.eps
        max_iter = self.max_iter
        for counter in range(0, max_iter + 1):
            d = (val_max + val_min) / 2
            choke = HE2_Choke(diam=d, d_pipe=d_pipe)
            q = choke.calc_q(p_in, p_out, t_in)
            if q - q_0 > 0:
                val_max = d
            elif q - q_0 < 0:
                val_min = d
            elif q - q_0 == 0:
                return d
            if abs(q - q_0) <= eps:
                return d
        logger.error(
            f"Calc_diam: desired accuracy not achieved"
        )
        return d

    def _wi_calc(self, p_r, p_in, t_av, a_c, a_r):
        """
        function to calculate isentropic mass flow rate
        :param p_r: pressure ratio
        :param p_in: given pressure
        :param t_av: average temperature
        :param a_c: choke throat area
        :param a_r: area ratio
        :return: isentropic mass flow rate
        """
        # check p_r > 0
        p_r = max(p_r, 0.000001)
        # calculate fluid
        # fluid = PVT_calc(p_in, t_av, self.fluid)
        # get polytropic_exponent
        # polytropic_exponent = fluid.polytropic_exponent
        # Calculate auxilary values
        # alpha = fluid.PVT_rho_gas_kgm3 * (
        #         fluid.fm_oil_fr / fluid.PVT_rho_oil_kgm3 + fluid.fm_wat_fr / fluid.PVT_rho_wat_kgm3)
        # lamba = (fluid.fm_gas_fr + (
        #         fluid.fm_gas_fr * fluid.cv_gas_JkgC + fluid.fm_oil_fr * fluid.cv_oil_JkgC + fluid.fm_wat_fr * fluid.cv_wat_JkgC) / (
        #                  fluid.cv_gas_JkgC * (fluid.heat_capacity_ratio_gas - 1)))
        # betta = fluid.fm_gas_fr / polytropic_exponent * p_r ** (-1 - 1 / polytropic_exponent)
        # Gamma = fluid.fm_gas_fr + alpha
        # Delta = fluid.fm_gas_fr * p_r ** (-1 / polytropic_exponent) + alpha

        # add constants for water, for fluid need to calculate them
        polytropic_exponent = 1
        fm_oil_fr = 0
        fm_gas_fr = 0
        fm_wat_fr = 1
        rho_gas_kgm3 = 100
        rho_oil_kgm3 = 780
        rho_wat_kgm3 = 1012
        heat_capacity_ratio_gas = 1.3
        cv_gas_JkgC = 1120
        cv_oil_JkgC = 2335
        cv_wat_JkgC = 4176
        alpha = rho_gas_kgm3 * (fm_oil_fr / rho_oil_kgm3 + fm_wat_fr / rho_wat_kgm3)
        lamba = (fm_gas_fr + (fm_gas_fr * cv_gas_JkgC + fm_oil_fr * cv_oil_JkgC + fm_wat_fr * cv_wat_JkgC) / (
                    cv_gas_JkgC * (heat_capacity_ratio_gas - 1)))
        Gamma = fm_gas_fr + alpha
        Delta = fm_gas_fr * p_r ** (-1 / polytropic_exponent) + alpha

        p_r = min(p_r, 1)
        res = 27500000 * a_c * (2 * p_in * rho_gas_kgm3 / Delta ** 2 * (
                    lamba * (1 - p_r ** (1 - 1 / polytropic_exponent)) + alpha * (1 - p_r)) / (
                                            1 - (a_r * Gamma / Delta) ** 2)) ** (1 / 2)
        return res

    def wi_calc(self, p_r, p_in, t_in, a_c, a_r):
        """
        function to calculate isentropic mass flow rate
        :param p_r: pressure ratio
        :param p_in: given pressure
        :param t_in: given temperature
        :param a_c: choke throat area
        :param a_r: area ratio
        :return: isentropic mass flow rate
        """
        # calculate fluid
        # fluid = PVT_calc(p_in, t_in, self.fluid)
        # polytropic_exponent = fluid.polytropic_exponent
        polytropic_exponent = 1
        # Calculate choke throat temperature
        t_choke_throat_C_ = (t_in + 273) * p_r ** (1 - 1 / polytropic_exponent) - 273
        # Calculate average temperature
        t_choke_av_C_ = (t_in + t_choke_throat_C_) / 2
        # get result
        res = self._wi_calc(p_r, p_in, t_choke_av_C_, a_c, a_r)
        return res
