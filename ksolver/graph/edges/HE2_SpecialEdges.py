from ksolver.tools import HE2_ABC as abc
from ksolver.fluids.HE2_Fluid import gimme_dummy_BlackOil
import scipy


class HE2_MockEdge(abc.HE2_ABC_GraphEdge):
    def __init__(self, delta_P=0, fluid=None):
        self.dP = delta_P

        self.fluid = fluid or gimme_dummy_BlackOil()


    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        # TODO имплементировать разбор направления по юнифлоку
        return P_bar + self.dP, T_C

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        return P_bar + self.dP, T_C

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        return P_bar - self.dP, T_C


class HE2_PQ_MockEdge(abc.HE2_ABC_GraphEdge):
    def __init__(self, PQ=[[0], [0]], TQ=[[0], [0]], fluid=None):
        self.PQ = PQ
        self.TQ = TQ
        self.fluid = fluid or gimme_dummy_BlackOil()
        self.p_interpolate_function = scipy.interpolate.interp1d(self.PQ[:, 0], self.PQ[:, 1], kind='slinear', bounds_error=False,fill_value='extrapolate')
        self.t_interpolate_function = scipy.interpolate.interp1d(self.TQ[:, 0], self.TQ[:, 1], kind='slinear', bounds_error=False,fill_value='extrapolate')

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        # TODO имплементировать разбор направления по юнифлоку
        return 0 + self.p_interpolate_function(X_kgsec), 0 + self.t_interpolate_function(X_kgsec)

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        return 0 + self.p_interpolate_function(X_kgsec), 0 + self.t_interpolate_function(X_kgsec)

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        return 0 - self.p_interpolate_function(X_kgsec), 0 + self.t_interpolate_function(X_kgsec)
