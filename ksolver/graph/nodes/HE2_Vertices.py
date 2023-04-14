from ksolver.tools.HE2_ABC import HE2_ABC_GraphVertex
from logging import getLogger
import scipy

logger = getLogger(__name__)


def is_junction(G, n):
    try:
        obj = G.nodes[n]["obj"]
    except Exception as e:
        logger.error(f"Invalid node {n}")
        raise e

    if type(obj) == HE2_ABC_GraphVertex:
        return True
    return False


def is_source(G, n):
    try:
        obj = G.nodes[n]["obj"]
    except Exception as e:
        logger.error(f"Invalid node {n}")
        raise e

    if type(obj) == HE2_Source_Vertex:
        return True
    return False


class HE2_Boundary_Vertex(HE2_ABC_GraphVertex):
    def __init__(self, kind, value):
        """
        Contract: Q-nodes in/out sign defines by self.is_source, sign of Q-value ignores
        :param kind:
        :param value:
        """
        self.kind = kind
        self.value = value
        self.P, self.Q = None, None
        if kind == "Q":
            self.value = abs(value)
            self.Q = self.value
        else:
            self.P = self.value

        self.is_source = False


class HE2_Source_Vertex(HE2_Boundary_Vertex):
    def __init__(self, kind, value, fluid, T, WC_Q_Curve=None):
        super().__init__(kind, value)
        self.fluid = fluid
        self.T = T
        self.WC_Q_Curve = WC_Q_Curve
        self.is_source = True
        if  not self.WC_Q_Curve is None:
            self.WC_interpolate_function = scipy.interpolate.interp1d(self.WC_Q_Curve[:, 0], self.WC_Q_Curve[:, 1], kind='slinear', bounds_error=False, fill_value='extrapolate')


