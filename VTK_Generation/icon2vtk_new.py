import time, os
import xarray as xr

import numpy as np
import metpy
from metpy.units import units

import pyvista as pv
from pyvista import CellType

from datetime import datetime, timedelta
from pyproj import Transformer

class icon_mesh():

    def __init__(self, fgrid, ficon, fext, nlayers, fdata=None):
        ds_grid = xr.open_dataset(fgrid)
        ds_icon = xr.open_dataset(ficon)
        ds_ext = xr.open_dataset(fext)
        if fdata is not None:
            ds_fdata = xr.open_dataset(fdata)

        self.clon_vertices = np.rad2deg(ds_grid.clon_vertices.values)
        self.clat_vertices = np.rad2deg(ds_grid.clat_vertices.values)
        self.ncells, nv = self.clon_vertices.shape[0], self.clon_vertices.shape[1]

        if fdata is not None:
            self.height_half = ds_fdata.sizes['height_3']
        else:
            self.height_half = ds_icon.sizes['height_2'] # _3

        if fdata is not None:
            self.height_full = ds_fdata.sizes['height_2']
        else:
            self.height_full = ds_icon.sizes['height']
        self.nlayers = nlayers
        self.nfaces = self.nlayers + 1

        # Grid variables
        self.np_points = np.ndarray([3 * (self.ncells * self.nfaces), nv])
        self.cell_type = np.ndarray(self.ncells * self.nlayers).astype(object)
        self.cells = np.ndarray([self.ncells * self.nlayers, 25]).astype(int)

        # height info z_ifc :: half-level, z_mc :: full-level
        self.z_ifc = np.ndarray([self.ncells * self.nfaces]).astype(float)
        self.z_mc = np.ndarray([self.ncells * self.nlayers]).astype(float)

        # at cell center
        self.w = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.u = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.v = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.temp = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.qv = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.qc = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.pres = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.theta_v = np.ndarray([self.ncells * self.nlayers]).astype(float)

        # Auxiliary variable
        self.idx = 0

        self.ds_icon = ds_icon
        self.ds_ext = ds_ext
        self.z_ifc = ds_icon.z_ifc[(self.height_half-self.nfaces):self.height_half].values

        if fdata is not None:
            self.ds_fdata = ds_fdata

    def get_info(self):
        return {
            "z_ifc": self.z_ifc,
            "height_half": self.height_half,
            "height_full": self.height_full,
            "ncells": self.ncells,
            "nfaces": self.nfaces,
            "nlayers": self.nlayers,
            "cells": self.cells,
            "cell_type": self.cell_type,
        }

    def _parse_time(self,time_str):
        date_part = time_str.split('.')[0]
        fractional_part = time_str.split('.')[1]
        fractional_days = float('0.' + fractional_part)

        # Parse the date part
        dt = datetime.strptime(date_part, '%Y%m%d')

        # Add fractional day as a timedelta
        dt += timedelta(days=fractional_days)

        # Round the datetime to the closest minute
        rounded_dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 else dt.replace(
            second=0, microsecond=0)

        return rounded_dt

    def print_time(self):
        # Extract the time variable as an array of strings (e.g., ['20230906.123456', ...])
        if self.ds_fdata is not None:
            time_strs = self.ds_fdata['time'].values.astype(str)
        else:
            time_strs = self.ds_icon['time'].values.astype(str)
        # Convert all time strings to datetime objects
        times = np.array([self._parse_time(time_str) for time_str in time_strs])
        #print(times)

    def _get_time(self, target_date_str):
        # Extract the time variable as an array of strings (e.g., ['20230906.123456', ...])
        if self.ds_fdata is not None:
            time_strs = self.ds_fdata['time'].values.astype(str)
        else:
            time_strs = self.ds_icon['time'].values.astype(str)
        # Convert all time strings to datetime objects
        times = np.array([self._parse_time(time_str) for time_str in time_strs])

        # Convert the target date to a datetime object
        target_date = datetime.strptime(target_date_str, '%Y%m%d %H:%M:%S')

        # Find the index of the closest date by calculating the absolute difference
        time_diffs = np.abs(times - target_date)
        closest_index = np.argmin(time_diffs)
        return closest_index

    def _get_time_ic(self, target_date_str):
        # Extract the time variable as an array of strings (e.g., ['20230906.123456', ...])
        time_strs = self.ds_icon['time'].values.astype(str)
        # Convert all time strings to datetime objects
        times = np.array([self._parse_time(time_str) for time_str in time_strs])

        # Convert the target date to a datetime object
        target_date = datetime.strptime(target_date_str, '%Y%m%d %H:%M:%S')

        # Find the index of the closest date by calculating the absolute difference
        time_diffs = np.abs(times - target_date)
        closest_index = np.argmin(time_diffs)
        return closest_index

    def _add_vertice(self, _lon, _lat, _z):
        self.np_points[self.idx, 0] = _lon
        self.np_points[self.idx, 1] = _lat
        self.np_points[self.idx, 2] = _z
        self.idx += 1

    def face_to_center(self, _var, date_str=None):
        if date_str is None:
            print("face_to_center: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            if self.ds_fdata is not None:
                _var = self.ds_fdata[_var][idx, :, :].values
            else:
                _var = self.ds_icon[_var][idx, :, :].values
            _nvar = np.zeros_like(_var, dtype=np.float64).flatten()[:-self.ncells]
            if self.ds_fdata is not None:
                L = len(self.ds_fdata.height_2)
            else:
                L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(1, self.nfaces):
                    _nvar[(i * self.nlayers) + (z - 1)] = (_var[L - z - 1, i] + _var[L - z, i]) / 2
        _nvar = _nvar.reshape(150, 88640)
        return _nvar

    def add_w(self, date_str=None):
        if date_str is None:
            print("add_w: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Interpolate w to cell center
            self.w = self.face_to_center('w',date_str)
            self.mesh.cell_data['w'] = self.w
            if self.ds_fdata is not None:
                self.topo.cell_data['w'] = self.ds_fdata['w'][idx, -1, :].values
            else:
                self.topo.cell_data['w'] = self.ds_icon['w'][idx, -1, :].values

    def add_wind_vector(self, date_str=None):
        if date_str is None:
            print("add_wind_vector: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            if self.ds_fdata is not None:
                _u = self.ds_fdata['u'][idx, :, :].values
                _v = self.ds_fdata['v'][idx, :, :].values
            else:
                _u = self.ds_icon['u'][idx, :, :].values
                _v = self.ds_icon['v'][idx, :, :].values
            # Interpolate w to cell center
            _w = self.face_to_center('w',date_str)
            if self.ds_fdata is not None:
                L = len(self.ds_fdata.height_2)
            else:
                L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.u[(i * self.nlayers) + z] = _u[(L - 1) - z, i]
                    self.v[(i * self.nlayers) + z] = _v[(L - 1) - z, i]
                    self.w[(i * self.nlayers) + z] = _w[(L - 1) - z, i]
            self.mesh.cell_data['u'] = self.u
            self.mesh.cell_data['v'] = self.v
            self.mesh.cell_data['w'] = self.w
            self.mesh.cell_data['wind'] = np.column_stack((self.u, self.v, self.w))

            #self.topo.cell_data['u'] = self.ds_icon['u'][idx, -1, :].values
            #self.topo.cell_data['v'] = self.ds_icon['v'][idx, -1, :].values
            #self.topo.cell_data['w'] = self.ds_icon['w'][idx, -1, :].values

    def add_theta(self, date_str=None):
        if date_str is None:
            print("add_theta: Please provide a date")
            pass
        if isinstance(date_str,str):
            idx = self._get_time_ic(date_str)
            _temp = self.ds_icon['temp'][idx, :, :].values
            _pres = self.ds_icon['pres'][idx, :, :].values
            L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.temp[(i * self.nlayers) + z] = _temp[(L - 1) - z, i]
                    self.pres[(i * self.nlayers) + z] = _pres[(L - 1) - z, i]

            self.mesh.cell_data['theta'] = metpy.calc.potential_temperature(self.pres * units.Pa,
                                                                            self.temp * units.kelvin).magnitude

    def add_theta_v(self, date_str=None):
        if date_str is None:
            print("add_theta_v: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            _theta_v = self.ds_icon['theta_v'][idx, :, :].values
            L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.theta_v[(i * self.nlayers) + z] = _theta_v[(L - 1) - z, i]

            self.mesh.cell_data['theta_v'] = self.theta_v
            self.topo.cell_data['theta_v'] = self.ds_icon['theta_v'][idx, -1, :].values

    
    def add_qv(self, date_str=None):
        if date_str is None:
            print("add_qv: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            _qv = self.ds_icon['qv'][idx, :, :].values
            L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.qv[(i * self.nlayers) + z] = _qv[(L - 1) - z, i]
            self.mesh.cell_data['qv'] = self.qv
            self.topo.cell_data['qv'] = self.ds_icon['qv'][idx, -1, :].values

    def add_qc(self, date_str=None):
        if date_str is None:
            print("add_qc: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            _qc = self.ds_icon['qc'][idx, :, :].values
            L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.qc[(i * self.nlayers) + z] = _qc[(L - 1) - z, i]
            self.mesh.cell_data['qc'] = self.qc
            self.topo.cell_data['qc'] = self.ds_icon['qc'][idx, -1, :].values

    def add_temp(self, date_str=None):
        if date_str is None:
            print("add_temp: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            _temp = self.ds_icon['temp'][idx, :, :].values
            L = len(self.ds_icon.height)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.temp[(i * self.nlayers) + z] = _temp[(L - 1) - z, i]
            self.mesh.cell_data['temp'] = self.temp
            self.topo.cell_data['temp'] = self.ds_icon['temp'][idx, -1, :].values

    def add_ice(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            self.topo.cell_data['ice'] = self.ds_ext['ICE'][:].values

    def add_shfl(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            self.topo.cell_data['shfl_s'] = self.ds_icon['shfl_s'][idx, :].values

    def add_tsk(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            self.topo.cell_data['t_sk'] = self.ds_icon['t_sk'][idx, :].values

    def add_t2m(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            self.topo.cell_data['t_2m'] = self.ds_icon['t_2m'][idx, 0, :].values

    def get_mesh(self):
        return self.mesh

    def get_vertices(self):
        return self.np_points

    def get_cells(self):
        return self.cells

    def get_cell_types(self):
        return self.cell_type

    def get_z_ifc(self):
        return self.z_ifc

    def get_z_mc(self):
        return self.z_mc

    def create_topo(self):

        np_points = np.ndarray([3 * self.ncells, 3])
        cells = np.ndarray([self.ncells, 4]).astype(int)
        cell_type = []

        # WGS84 zu UTM Zone 32N (Österreich liegt meist in Zone 32N)
        transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

        idx = 0
        for i in range(0, self.ncells, 1):
            for j in range(3):
                clon = self.clon_vertices[i, j]
                clat = self.clat_vertices[i, j]
                # Umrechnen
                x, y = transformer.transform(clon, clat)
                np_points[idx, 0] = np.array(x)
                np_points[idx, 1] = np.array(y)
                np_points[idx, 2] = np.array(0)
                idx = idx + 1
            cells[i, :] = [3, idx - 3, idx - 2, idx - 1]
            cell_type = cell_type + [CellType.TRIANGLE]

        topo = pv.UnstructuredGrid(cells, cell_type, np_points)
        topo.cell_data['z_ifc'] = self.z_ifc[self.nfaces - 1, :]

        # Remove duplicated
        ctopo = topo.clean()

        # Convert cell data to point data
        ptopo = ctopo.cell_data_to_point_data()

        # Assign height to coordinates
        for i in range(ptopo.points.shape[0]):
            ptopo.points[i, 2] = ptopo.point_data['z_ifc'][i]

        self.topo = ptopo

        return ptopo

    def create_grid(self):

        _z_mc = self.ds_icon.z_mc.values

        # WGS84 zu UTM Zone 32N (Österreich liegt meist in Zone 32N)
        transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

        for i in range(0, self.ncells, 1):
            for z in range(self.nfaces):
                for j in range(3):
                    x, y = transformer.transform(self.clon_vertices[i, j], self.clat_vertices[i, j])
                    _x = x
                    _y = y
                    _z = z
                    self._add_vertice(_x, _y, _z)
                if z > 0:
                    # Store first cell index
                    ids = self.idx - 6

                    # Create cell
                    self.cells[(i * self.nlayers) + (z - 1), :] = [24, 5,
                                                                   3, ids, ids + 1, ids + 2,
                                                                   3, ids + 3, ids + 4, ids + 5,
                                                                   4, ids, ids + 1, ids + 4, ids + 3,
                                                                   4, ids + 1, ids + 2, ids + 5, ids + 4,
                                                                   4, ids + 2, ids, ids + 3, ids + 5]

                    # Define cell type
                    self.cell_type[(i * self.nlayers) + (z - 1)] = CellType.POLYHEDRON

                    # Height information at cell center
                    self.z_mc[(i * self.nlayers) + (z - 1)] = _z_mc[self.height_full - z, i]

        # Create mnesh
        cmesh = pv.UnstructuredGrid(self.get_cells(), self.get_cell_types(), self.get_vertices())

        # Remove duplicated vertices
        cmesh = cmesh.clean()

        # Add height information as cell data
        cmesh.cell_data['z_mc'] = self.get_z_mc()
        pmesh = cmesh.cell_data_to_point_data()

        # Assign height to coordinates
        pmesh.points[:, 2] = pmesh.point_data['z_mc'][:]

        # for i in range(pmesh.points.shape[0]):
        #     if i < (self.topo.points.shape[0]):
        #         pmesh.points[i, 2] = self.topo.point_data['z_ifc'][i]
        #     else:
        #         pmesh.points[i, 2] = pmesh.point_data['z_mc'][i]

        self.mesh = pmesh
        return self.mesh

def main():
    from pathlib import Path
    
    fext = r"external_parameter_icon_hef_DOM01_tiles.nc"
    fgrid = r"hef_51m_DOM01.nc"
    icon_root = Path(r"/work/bb1461/hefex3/v370_2030/exp_R3B15_51m/output")

    # Generally ficon and fdata are the same file, they only need to be changed when 
    # working with the 30min files since they dont have all geometry attributed for the vtk
    ficon = sorted(icon_root.glob("LES_51m_ml*.nc"))[0]
    fdata = sorted(icon_root.glob("LES_30min_51m_ml*.nc"))[0]

    nlayers = 50
    
    imesh = icon_mesh(fgrid, ficon, fext, nlayers=nlayers, fdata=fdata)
    _ = imesh.create_grid()
    _ = imesh.create_topo()

    # In principle one could start a for loop here, open new timestamps (or new files) and add them via imesh.add to the existing file (if the time index is handeled).
    ds = xr.open_dataset(fdata)
    imesh.ds_fdata = ds

    t = str(ds.time[0].values)
    date_part = t.split('.')[0]
    fractional_part = t.split('.')[1]
    fractional_days = float('0.' + fractional_part)
    dt = datetime.strptime(date_part, '%Y%m%d')
    dt += timedelta(days=fractional_days)
    t = dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 else dt.replace(
        second=0, microsecond=0)
    t_str = str(t).replace("-", "")

    imesh.add_theta_v(t_str)

    imesh.mesh.save("test_vtk.vtk")


if __name__ == "__main__":
    main()
