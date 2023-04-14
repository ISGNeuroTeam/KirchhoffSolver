from logging import getLogger
import pandas as pd
import numpy as np
import scipy
import os
from ksolver.tools.HE2_schema_maker import gimme_dummy_oil_params, replace_pump_model_with_closest, process_well_row
from ksolver.tools.HE2_schema_maker import get_base_oil_params_from_df
from ksolver.graph.edges.HE2_Pipe import HE2_OilPipe
from ksolver.graph.edges.HE2_Plast import HE2_Plast
from ksolver.graph.edges.HE2_WellPump import create_HE2_WellPump_instance_from_dataframe
from ksolver.fluids.HE2_Fluid import gimme_BlackOil
from ksolver.io.calculate_DF import calculate_DF
from math import pi
import shutil

logger = getLogger(__name__)


class Const:
    c_water = 4170
    c_oil = 880


class MRM:
    def __init__(self, inclination, HKT, pump_curves, row_type_col="rowType", q_step=0.01, p_step=1, start=0, end=3,
                 kind='quadratic', deletePTWCQcurves=True):
        """
        :param pump_curves: db of pumps
        :param inclination: db of inclination
        :param HKT: db of HKT
        :param row_type_col: name of column with type of row (pipe, oilwell)
        :param q_step: step for fluid rate iteration
        :param p_step: step for pad pressure
        :param start: start value of oil rate
        :param end: end value of oil rate
        :param kind: kind of interpolation
        """
        self.inclination = inclination
        self.HKT = HKT
        self.pump_curves = pump_curves
        self.row_type_col = row_type_col
        self.q_step = q_step
        self.p_step = p_step

        # p_wh(q) for all oilwells in the pad
        self.p_wh_t_by_q_for_pad_dict = dict()
        # wc for all wells in the pad
        self.wc_of_wel_for_pad_dict = dict()
        # dictionary q_pad(p_pad), wc_pad(p_pad), t(p_pad) for all pads
        self.q_pad_wc_pad_t_pad_by_p_pad_for_pad_dict = dict()
        # dictionary with functions p_pad(q), wc_pad(q) in the pad
        # TODO: Нужно поправить имя, это плохо читается
        self.p_pad_wc_pad_t_pad_by_q_pad_dict = dict()

        self.q_well_function_dict = dict()

        self.start = start
        self.end = end
        self.kind = kind

        self.deletePTWCQcurves = deletePTWCQcurves
        self.curves_folder = None

        self.initial_df = None
        self.wells_node_id_end = None

        self.default_plast_pressure_atm = 222.22222
        self.default_productivity_m3_day_atm = 0.5

        self.base_op = None


    def get_p_wh_q_for_pad(self, dataset):
        """
        function to calculate  p_wh(q) for all wells in the pad
        :param dataset: description of all oilwells
        :return: p_wh(q), t(q) for all wells in the pad in dictionary { pad: {wel : (q, p_wh, t) },
                dictionary WC for all wells  { pad: {wel : WC} }
        """
        # go throw all rows
        for _, row in dataset.iterrows():
            # calculate wellhead pressures under fluid rates
            x_kgsec_array, p_whs, t = self.well_transform(row)
            # get padNum and wellNum
            pad, well = row['padNum'], row['wellNum']
            # get wells dictionary if it exists for this pad
            wells_p_wh_by_q_dict = self.p_wh_t_by_q_for_pad_dict.get(pad, {})
            wells_wc_dict = self.wc_of_wel_for_pad_dict.get(pad, {})
            # add to wells dict wellhead pressures under fluid rates and wc for this well
            wells_p_wh_by_q_dict[well] = np.array(x_kgsec_array), np.array(p_whs), np.array(t)
            wells_wc_dict[well] = row['VolumeWater']
            # add to dict for this pad and well wellhead pressures under fluid rates and wc
            self.p_wh_t_by_q_for_pad_dict[pad] = wells_p_wh_by_q_dict
            self.wc_of_wel_for_pad_dict[pad] = wells_wc_dict

    def well_transform(self, well_row):
        """
        function to calculate wellhead pressure and temperature under different oil rate
        :param well_row: oilwell description from dataframe
        :return: arrays of wellhead  pressures and temperatures, array of oil rates
        """
        # arrays wor p and t for wellhead
        p_whs, t_whs = [], []
        # split well into 5 parts (plast-zaboi,zaboi-intake, wellpump, outlet-wellhead, wellhead-pad)
        dict_list = process_well_row(self.HKT, [], self.inclination, well_row, self.row_type_col)
        # get rid of wellhead-pad part
        # well_dict_list = dict_list[:-1]
        well_dict_list = dict_list
        # array of x_kgsec values
        x_kgsec_array = np.arange(self.start, self.end, self.q_step)
        # calculate well for different value of x_kgsec
        for x_kgsec in x_kgsec_array:
            # calculate wellhead pressure and temperature
            p_wh, t_wh = self.calculate_well(well_dict_list, x_kgsec)
            # check if wellhead pressure enough for this oil rate
            # check here to add negative wellhead pressure to array
            if p_wh is None:
                break
            # add results to arrays
            p_whs.append(p_wh)
            t_whs.append(t_wh)
            # check if wellhead pressure enough for this oil rate
            if p_wh < 0:
                break
        return x_kgsec_array[: len(p_whs)], p_whs, t_whs

    def calculate_well(self, well_dict_list, x_kgsec):
        """
        function to calculate wellhead pressure and temperature
        :param well_dict_list: dictionary with 4 parts of oilwell (plast-zaboi,zaboi-intake, wellpump, outlet-wellhead)
        :param x_kgsec:oil rate
        :return: wellhead pressure and temperature
        """
        # get base fluid properties

        VolumeWater = well_dict_list[0]["VolumeWater"] if pd.notna(well_dict_list[0]["VolumeWater"]) else 50
        fluid = gimme_BlackOil(self.base_op, volumewater_percent=VolumeWater)

        # The well is modeled as 4 consecutively combined simple hydraulic models:
        # plast-zaboi, zaboi-intake, wellpump, outlet-wellhead

        # plast-zaboi
        plast_zaboi = well_dict_list[0]
        # plast pressure and temperature
        p, t = plast_zaboi['startValue'], plast_zaboi['startT']
        # print("plast: ", p, t)
        productivity = plast_zaboi["productivity"]
        # create object HE2_Plast
        obj = HE2_Plast(productivity=productivity, fluid=fluid)
        # calculate pressure and temperature for part plast-zaboi
        p, t = obj.perform_calc_forward(p, t, x_kgsec)
        # print("zaboi: ", p, t)

        # zaboi-intake
        pipe_zaboi_intake = well_dict_list[1]
        # get pipe parameters from dictionary
        L = pipe_zaboi_intake["L"]
        uphill = pipe_zaboi_intake["uphillM"]
        diam_coef = pipe_zaboi_intake["effectiveD"]
        D = pipe_zaboi_intake["intD"]
        roughness = pipe_zaboi_intake["roughness"]
        # create object HE2_OilPipe
        obj = HE2_OilPipe([L], [uphill], [D * diam_coef], [roughness], fluid)
        # calculate pressure and temperature for part zaboi-intake
        p, t = obj.perform_calc_forward(p, t, x_kgsec)
        # print("intake: ", p, t)
        # check if pump intake pressure enough for this oil rate
        if p < 0:
            return None, None

        # wellpump
        wellpump = well_dict_list[2]
        # get pump parameters from dictionary and pump_curves
        model = wellpump["model"]
        frequency = wellpump["frequency"]
        if model not in self.pump_curves['pumpModel']:
            model = replace_pump_model_with_closest(model, self.pump_curves)
        # create object HE2_WellPump
        pump = create_HE2_WellPump_instance_from_dataframe(full_HPX=self.pump_curves, model=model, fluid=fluid,
                                                           frequency=frequency)
        K_pump = wellpump.get("K_pump", 1)
        if 0 < K_pump < float('inf'):
            pump.change_stages_ratio(K_pump)
        # calculate pressure and temperature for pump
        p, t = pump.perform_calc_forward(p, t, x_kgsec)
        # print("outlet: ", p, t)

        # outlet-wellhead
        pipe_outlet_wellhead = well_dict_list[3]
        # get pipe parameters from dictionary
        L = pipe_outlet_wellhead["L"]
        uphill = pipe_outlet_wellhead["uphillM"]
        diam_coef = pipe_outlet_wellhead["effectiveD"]
        D = pipe_outlet_wellhead["intD"]
        roughness = pipe_outlet_wellhead["roughness"]
        # create object HE2_OilPipe
        obj = HE2_OilPipe([L], [uphill], [D * diam_coef], [roughness], fluid)
        # calculate pressure and temperature for part pump outlet-wellhead
        p, t = obj.perform_calc_forward(p, t, x_kgsec)
        # print("wellhead: ", p, t)

        # wellhead-pad
        # find pad pipe
        pad_pipe = self.initial_df.loc[self.initial_df['node_id_start'] == well_dict_list[4]['node_id_end']]
        # get pipe parameters from dataframe
        L = pad_pipe["L"].values[0]
        uphill = pad_pipe["uphillM"].values[0]
        diam_coef = pad_pipe["effectiveD"].values[0]
        D = pad_pipe["intD"].values[0]
        roughness = pad_pipe["roughness"].values[0]
        # create object HE2_OilPipe
        obj = HE2_OilPipe([L], [uphill], [D * diam_coef], [roughness], fluid)
        # calculate pressure and temperature for part pump outlet-wellhead
        p, t = obj.perform_calc_forward(p, t, x_kgsec)
        return p, t

    def process_all_pads(self):
        """
        function to evaluate q_pad(p_pad), wc_pad(p_pad) for each pad
        :param kind: kind of interpolation
        :return: dictionary { pad: (p_pad, q_pad, wc_pad, t_pad)}
        """
        # go throw all pads
        for pad in self.p_wh_t_by_q_for_pad_dict:
            # evaluate q_pad(p_pad), wc_pad(p_pad) for this pad
            self.q_pad_wc_pad_t_pad_by_p_pad_for_pad_dict[pad] = self.evaluate_pad(self.p_wh_t_by_q_for_pad_dict[pad],
                                                                                   self.wc_of_wel_for_pad_dict[pad])

    def evaluate_pad(self, p_wh_t_by_q_for_well_dict, wc_for_wel_dict):
        """
        function to calculate fluid rate and wc by pad pressure
        :param p_wh_t_by_q_for_well_dict: p_wh(q), t(q) for all oilwells in the pad { well : (q, p_wh, t)}
        :param wc_for_wel_dict: dictionary { well : wc }
        :return: p_pad, q_pad, wc_pad
        """
        # find maximum wellhead pressure in this pad
        p_wh_max = max(max(p_wh) for _, (_, p_wh, _) in p_wh_t_by_q_for_well_dict.items())
        # array of pad pressure for iteration
        p_pad_array = np.arange(0, p_wh_max, self.p_step)
        # array of pad fluid rate
        q_pad_array = []
        # array of pad wc
        wc_pad_array = []
        # array of pad t
        t_pad_array = []
        # dictionaries for interpolation functions q(p), t(p) for different well
        if self.q_well_function_dict:
            self.q_well_function_dict.update(dict.fromkeys(p_wh_t_by_q_for_well_dict.keys(), None))
        else:
            self.q_well_function_dict = dict.fromkeys(p_wh_t_by_q_for_well_dict.keys(), None)
        t_well_function_dict = self.q_well_function_dict.copy()
        # iterate by pad pressure
        for p_pad in p_pad_array:
            # total pad fluid rate
            q_pad = 0
            # total pad oil rate
            q_oil_pad = 0
            # total pad water rate
            q_water_pad = 0
            # variables to find mixture temperature
            qct = 0
            qc = 0
            # go throw all wells
            for well, (q, p_wh, t) in p_wh_t_by_q_for_well_dict.items():
                # find well fluid rate solving p_wh(q_well) = p_pad
                # check if current pad pressure maximum pressure on wellhead
                if p_pad > p_wh[0]:
                    q_well = 0
                else:
                    # check if q_well(p) has been calculated
                    if self.q_well_function_dict[well] is None:
                        q_well, self.q_well_function_dict[well] = self.evaluate_q_well(q, p_wh, p_pad)
                    else:
                        q_well = self.q_well_function_dict[well](p_pad)
                # if well fluid rate negative replace by 0
                q_well = max(0, q_well)
                # find well temperature
                # check if t(q) has been calculated
                if t_well_function_dict[well] is None:
                    t_well, t_well_function_dict[well] = self.evaluate_t_well(q, t, q_well)
                else:
                    t_well = t_well_function_dict[well](q_well)
                # find well water rate
                q_water = q_well * wc_for_wel_dict[well] / 100
                # find oil water rate
                q_oil = q_well - q_water
                # add well results to pad
                q_pad += q_well
                q_oil_pad += q_oil
                q_water_pad += q_water
                # add results to find mixture temperature
                qc += Const.c_oil * q_oil + Const.c_water * q_water
                qct += Const.c_oil * q_oil * t_well + Const.c_water * q_water * t_well
                # add to array found pad fluid rate
            if q_pad == 0:
                continue
            else:
                q_pad_array.append(q_pad)
                # add to array found pad temperature
                t_pad_array.append(qct / qc)
                # add to array found pad wc
                wc_pad_array.append(q_water_pad / q_pad * 100)

        return p_pad_array[0:len(q_pad_array)], np.array(q_pad_array), np.array(wc_pad_array), np.array(t_pad_array)

    def evaluate_q_well(self, q, p_wh, p_pad):
        """
        function to evaluate well fluid rate under pad pressure
        :param q: fluid rate for well
        :param p_wh: wellhead pressure
        :param p_pad: pad pressure
        :param kind: kind of interpolation
        :return: fluid rate for well and function q_well(p)
        """
        # calculate only for positive wellhead pressure
        pos_index = p_wh > 0
        # find function q_well(p_wh)
        function = self.get_function(p_wh[pos_index], q[pos_index], kind=self.kind)
        # find q_well(p_pad)
        q_well = function(p_pad)
        return q_well, function

    def evaluate_t_well(self, q, t, q_well):
        """
        function to evaluate temperature by given well fluid rate
        :param q: fluid rate array
        :param t: fluid temperature by fluid rate
        :param q_well: given well fluid rate
        :return: temperature by given well fluid rate and function t_well(q)
        """
        function = self.get_function(q, t, kind=self.kind)
        return function(q_well), function

    def get_function(self, x, y, kind='quadratic', fill_value='extrapolate'):
        """
        function to interpolate y(x) by cubic splines
        :param x: x array
        :param y: y array
        :param kind: kind of interpolation
        :return: function y(x)
        """
        # sort x in increasing way
        sort_index = np.argsort(x)
        # find y(x)
        # function = scipy.interpolate.CubicSpline(x[sort_index], y[sort_index])
        function = scipy.interpolate.interp1d(x[sort_index], y[sort_index], kind=kind, bounds_error=False,
                                              fill_value=fill_value)
        return function

    def evaluate_pad_functions(self, kind='slinear'):
        """
        function to evaluate p_pad(q), wc_pad(q)
        :return: dictionary with functions p_pad(q_pad), wc_pad(q_pad) in the pad { pad : ( p_pad(q_pad), wc_pad(q_pad), t_pad(q_pad) ) }
        """
        # go throw all pads
        for pad, (p_pad, q_pad, wc_pad, t_pad) in self.q_pad_wc_pad_t_pad_by_p_pad_for_pad_dict.items():
            # evaluate p_pad(q_pad) and wc_pad(q_pad) for this pad
            PQ_func = self.get_function(q_pad, p_pad, kind=kind)
            WCQ_func = self.get_function(q_pad, wc_pad, kind=kind)
            TQ_func = self.get_function(q_pad, t_pad, kind=kind)
            self.p_pad_wc_pad_t_pad_by_q_pad_dict[pad] = PQ_func, WCQ_func, TQ_func

    def save_tables(self, curves_folder):
        for pad, (p_pad, q_pad, wc_pad, t_pad) in self.q_pad_wc_pad_t_pad_by_p_pad_for_pad_dict.items():
            pd.DataFrame({'q': q_pad, 'p': p_pad}).to_csv(os.path.join(curves_folder, f'pad_{pad}_PQ.csv'),
                                                          header=False, index=False)
            pd.DataFrame({'q': q_pad, 't': t_pad}).to_csv(os.path.join(curves_folder, f'pad_{pad}_TQ.csv'),
                                                          header=False, index=False)
            pd.DataFrame({'q': q_pad, 'wc': wc_pad}).to_csv(os.path.join(curves_folder, f'pad_{pad}_WCQ.csv'),
                                                            header=False, index=False)

    def apply_patch_to_bad_productivity(self, initial_df):
        mean_productivity = initial_df.loc[initial_df[self.row_type_col] == 'oilwell', 'productivity'].mean()
        def_value = mean_productivity if mean_productivity > 0 else self.default_productivity_m3_day_atm

        mask1 = initial_df[self.row_type_col] == 'oilwell'
        mask2 = initial_df['productivity'].isna()
        initial_df.loc[mask1 & mask2, 'productivity'] = def_value

        for row in initial_df[mask1 & mask2].itertuples():
            logger.error(f'Missed productivity for well {row.node_name_start}, filled by {def_value} m3/day/atm')
        return initial_df

    def apply_patch_to_bad_plast_pressure(self, initial_df):
        mean_P_plast = initial_df.loc[initial_df[self.row_type_col] == 'oilwell', 'startValue'].mean()
        def_value = mean_P_plast if mean_P_plast > 0 else self.default_plast_pressure_atm

        mask1 = initial_df[self.row_type_col] == 'oilwell'
        mask2 = initial_df['startKind'] == 'P'
        mask3 = initial_df['startValue'].isna()
        mask = mask1 & mask2 & mask3
        initial_df.loc[mask, 'startValue'] = def_value

        for row in initial_df[mask].itertuples():
            logger.error(f'Missed plast pressure for well {row.node_name_start}, filled by {def_value} atm')
        return initial_df


    def build_df_with_curves(self, folder, initial_df, curves_folder='PWCTQ_curves'):
        self.curves_folder = os.path.join(folder, curves_folder)
        try:
            os.mkdir(self.curves_folder)
        except FileExistsError:
            logger.warning("Directory for curves already exist")

        initial_df = self.apply_patch_to_bad_productivity(initial_df)
        initial_df = self.apply_patch_to_bad_plast_pressure(initial_df)

        self.base_op = get_base_oil_params_from_df(initial_df)

        self.initial_df = initial_df
        # get dataset for oilwells
        well_dataset = self.initial_df[self.initial_df[self.row_type_col] == 'oilwell']
        # evaluate p_wh(q) for all wells in all pads
        self.get_p_wh_q_for_pad(well_dataset)
        # evaluate q_pad(p_pad) and wc_pad(p_pad) for all pads
        self.process_all_pads()
        # save tables in csv files

        self.save_tables(self.curves_folder)

        # self.plot_PTWCQ_curves()

        # get aggregate dataframe
        aggregate_df = self.make_aggregate_df()
        return aggregate_df

    def make_aggregate_df(self):
        self.wells_node_id_end = set(self.initial_df.loc[self.initial_df[self.row_type_col] == 'oilwell', 'node_id_end'])

        mask1 = self.initial_df[self.row_type_col] != 'oilwell'
        mask2 = ~self.initial_df['node_id_start'].isin(self.wells_node_id_end)
        aggregate_df = self.initial_df.loc[mask1 & mask2]

        for pad in aggregate_df.padNum.unique():
            if pad != '-1':
                aggregate_df.loc[aggregate_df["padNum"] == pad, "startKind"] = 'PQCurve'
                aggregate_df.loc[aggregate_df["padNum"] == pad, "startIsSource"] = True
                aggregate_df.loc[aggregate_df["padNum"] == pad, "startValue"] = os.path.join(self.curves_folder,
                                                                                             f'pad_{pad}_PQ.csv')
                aggregate_df.loc[aggregate_df["padNum"] == pad, "startT"] = os.path.join(self.curves_folder,
                                                                                         f'pad_{pad}_TQ.csv')
                aggregate_df.loc[aggregate_df["padNum"] == pad, "VolumeWater"] = os.path.join(self.curves_folder,
                                                                                              f'pad_{pad}_WCQ.csv')
        return aggregate_df

    def convert_results(self, initial_df, aggregate_df_res):
        # df with pipes
        pipes_df = aggregate_df_res
        # df with pipes from well to pad
        well_pipes = initial_df.loc[
            (initial_df[self.row_type_col] == 'pipe') & (initial_df['node_id_start'].isin(self.wells_node_id_end))]
        # df with oilwells
        well_result_df = initial_df[initial_df[self.row_type_col] == 'oilwell']
        # results which need to fill
        result_columns = ['endT', 'startP', 'endP', 'start_Q_m3_day', 'end_Q_m3_day',
                          'res_liquid_density_kg_m3', 'res_pump_power_watt', 'X_kg_sec', 'velocity_m_sec', 'res_watercut_percent']
        well_result_df.loc[:, result_columns] = None
        well_pipes.loc[:, result_columns] = None
        # results that already in df
        # plast pressure
        well_result_df['startP'] = well_result_df['startValue']
        # temperature
        well_result_df['endT'] = well_result_df['startT']
        well_pipes['endT'] = well_pipes['startT']
        # VolumeWater
        well_result_df['res_watercut_percent'] = well_result_df['VolumeWater']
        # find oil density

        if len(well_result_df['Oildensity_kg_m3'].unique()) == 1:
            oildensity_kg_m3 = well_result_df['Oildensity_kg_m3'].unique()[0]
        else:
            # use default value
            oildensity_kg_m3 = 826
        # find water density
        if len(well_result_df['Waterdensity_kg_m3'].unique()) == 1:
            waterdensity_kg_m3 = well_result_df['Waterdensity_kg_m3'].unique()[0]
        else:
            # use default value
            waterdensity_kg_m3 = 1015
        # fill res_liquid_density_kg_m3
        well_result_df['res_liquid_density_kg_m3'] = well_result_df['VolumeWater'] / 100 * waterdensity_kg_m3 + (
                1 - well_result_df['VolumeWater'] / 100) * oildensity_kg_m3
        # fill VolumeWater and res_liquid_density_kg_m3 for pipes from well to pad
        for row in well_pipes.itertuples():
            well_pipes.loc[row.Index, 'res_watercut_percent'] = well_result_df.loc[
                well_result_df['node_id_end'] == well_pipes.loc[row.Index, 'node_id_start'], 'res_watercut_percent'].values[0]
            well_pipes.loc[row.Index, 'res_liquid_density_kg_m3'] = well_result_df.loc[
                well_result_df['node_id_end'] == well_pipes.loc[row.Index, 'node_id_start'], 'res_liquid_density_kg_m3'].values[0]
        # fill endP for pipe from well to pad as startP for pad pipe
        for pad in aggregate_df_res.padNum.unique():
            if pad != '-1':
                well_pipes.loc[well_pipes['padNum'] == pad, "endP"] = \
                    aggregate_df_res.loc[aggregate_df_res['padNum'] == pad, "startP"].values[0]

        # fill X_kg_sec by interpolation throw wellhead pressure
        for row in well_pipes.itertuples():
            well = row.wellNum
            p_pad = row.endP
            q_well = self.q_well_function_dict[well](p_pad)
            X_kg_sec = max(q_well, 0.0) + 0.0
            well_pipes.loc[row.Index, 'X_kg_sec'] = X_kg_sec
            # find startP as pipe.perform_calc_backward
            # get pipe parameters from row
            L = row.L
            uphill = row.uphillM
            diam_coef = row.effectiveD
            D = row.intD
            roughness = row.roughness
            # create object HE2_OilPipe
            obj = HE2_OilPipe([L], [uphill], [D * diam_coef], [roughness], gimme_BlackOil(gimme_dummy_oil_params(), volumewater_percent=row.res_watercut_percent))
            # calculate pressure and temperature for part zaboi-intake
            p, _ = obj.perform_calc_backward(row.endP, row.endT, X_kg_sec)
            well_pipes.loc[row.Index, "startP"] = p
            # fill and velocity_m_sec using found X_kg_sec
            # well_result_df.loc[index, "start_Q_m3_day"] = X_kg_sec / row.res_liquid_density_kg_m3 * 3600 * 24
            Q = X_kg_sec / row.res_liquid_density_kg_m3
            area = pi * row.intD ** 2 / 4
            well_pipes.loc[row.Index, "velocity_m_sec"] = Q / area

        # fill X_kg_sec for well
        for row in well_result_df.itertuples():
            well_result_df.loc[row.Index, 'endP'] = well_pipes.loc[
                well_pipes['node_id_start'] == well_result_df.loc[row.Index, 'node_id_end'], 'startP'].values[0]
            X_kg_sec = well_pipes.loc[well_pipes['node_id_start'] == well_result_df.loc[row.Index, 'node_id_end'], 'X_kg_sec'].values[0]
            well_result_df.loc[row.Index, 'X_kg_sec'] = X_kg_sec
            well_result_df.loc[row.Index, "start_Q_m3_day"] = X_kg_sec / row.res_liquid_density_kg_m3 * 3600 * 24

        # Here we calc well pump power consumption for results

        # TODO расчет мощности насоса надо переделать. Сейчас здесь заглушено давление и температура на входе
        # Нужно либо считать скважину сверху донизу
        # Либо добавить в кривые скважины еще и кривую мощности
        for row in well_result_df.itertuples():
            wc = row.VolumeWater if pd.notna(row.VolumeWater) else 50
            fluid = gimme_BlackOil(self.base_op, volumewater_percent=wc)

            x_kgsec = 0 + row.X_kg_sec

            model = row.model
            frequency = row.frequency
            if model not in self.pump_curves['pumpModel']:
                model = replace_pump_model_with_closest(model, self.pump_curves)
            # create object HE2_WellPump
            kwargs = dict(full_HPX=self.pump_curves, model=model, fluid=fluid, frequency=frequency)
            pump = create_HE2_WellPump_instance_from_dataframe(**kwargs)
            K_pump = row.K_pump
            if 0 < K_pump < float('inf'):
                pump.change_stages_ratio(K_pump)
            # calculate pressure and temperature for pump
            P_intake = 30
            T_intake = row.startT
            pump.perform_calc_forward(P_intake, T_intake, x_kgsec)
            well_result_df.loc[row.Index, "res_pump_power_watt"] = pump.power


        # delete PTWCQ curves
        if self.deletePTWCQcurves:
            try:
                shutil.rmtree(self.curves_folder)
            except OSError as e:
                logger.error("Error: %s : %s" % (self.curves_folder, e.strerror))
        return pd.concat([pipes_df, well_result_df, well_pipes])


if __name__ == '__main__':
    data_folder = os.getcwd()
    path = os.path.join(data_folder, "df_merged.xlsx")
    dataset = pd.read_excel(path)

    # load inclination, HKT and PumpChart
    inclination = pd.read_parquet(os.path.join(data_folder, "inclination"), engine="pyarrow")
    HKT = pd.read_parquet(os.path.join(data_folder, "HKT"), engine="pyarrow")
    pump_curves = pd.read_csv(os.path.join(data_folder, "PumpChart.csv"))

    my_mrm = MRM(inclination, HKT, pump_curves, row_type_col='rowType', p_step=1, q_step=0.04)
    aggregate_df = my_mrm.build_df_with_curves(data_folder, dataset)
    mdf = calculate_DF(aggregate_df, data_folder=os.getcwd(), solver_params=dict(threshold=2), row_type_col="rowType",
                       check_input_df=False, return_graph=False)
    result_df = my_mrm.convert_results(dataset, mdf)
