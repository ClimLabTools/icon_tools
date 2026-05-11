import geopandas as gpd
import matplotlib.colors as mcolors
import xarray as xr
import pyvista as pv
from scipy.ndimage import gaussian_filter
from pyproj import Transformer
import os
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import LinearNDInterpolator, griddata
import matplotlib.pyplot as plt
import numpy.ma as ma


# Further Steps
    # Adapt height to the new icon2vtk

    # Add functionality for spline inputs


class VerticalPlotter:
    '''
    Class to generate vertical plots from icon vtk files
    Recquires:
    1. vtk file,
    2. vector file of the transect line,
    3. name of the variable to be plotted,
    4. maximum height of the transect as a list of [maximum array height, maximum plot height]
    '''
    def __init__(self, icon_vtk, epsg, gdf_line, plot_variable, max_height, grid_width=1, interp_method='linear', itopo=None):
        self.icon_vtk = icon_vtk.cell_data_to_point_data()
        self.epsg = epsg
        self.grid_width = grid_width
        self.interp_method = interp_method
        self.plot_variable = plot_variable
        self.max_height = max_height
        self.gdf_line = gdf_line
        self.vector_transform_option = 1
        self.pv_line = None
        self.slice = None
        self.loni = None
        self.zi = None
        self.lon = None
        self.z = None
        self.values = None
        self.grid_values = None
        self.grid_values_x = None
        self.grid_values_y = None
        self.grid_values_z = None
        self.grid_values_lon = None
        self.grid_values_z_new = None
        self.grid_values_perp = None
        self.offsets = None
        self.rv_mode = False
        self.itopo = itopo
        self.height_mask = None

        self.scalar = True if len(self.icon_vtk.get_array(self.plot_variable).shape) == 1 else False

    def plotter_info(self):
        print(self.icon_vtk)

    def adadpt_z(self, col_new='z_ifc'):
        '''
        Used to transform the z-values of the normalized vtk z-coords to the true meters stored in the z_ifc column
        :param col_new: The column name containing the correct z-values in meters
        '''

        col_old = 'old_z_vals'
        new_z = np.array(self.icon_vtk.get_array(col_new), dtype=self.icon_vtk.points.dtype)
        old_z = self.icon_vtk.points[:, 2]
        points = self.icon_vtk.points.copy()
        points[:, 2] = new_z
        self.icon_vtk.points = points

        self.icon_vtk[col_old] = old_z

        self.icon_vtk = self.icon_vtk.threshold((0, self.max_height[0]), scalars="z_ifc")

        self.icon_vtk = pv.UnstructuredGrid(self.icon_vtk)

    def gpd_line_2_pv_line(self, n_points):
        '''
        Takes the gdf line and transforms it into a pv spline to be compatible with the pv.slice_along_line function
        :param n_points: Choose how many points the spline should have.
        '''
        if self.gdf_line.crs != self.epsg:
            self.gdf_line = self.gdf_line.to_crs(self.epsg)

        #print(f"Line length: {self.gdf_line.length}")

        coords = np.array(self.gdf_line.geometry.values[0].coords)
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

        self.pv_line = pv.Spline(coords, n_points)

    @staticmethod
    def reorder_slice_points_by_spline(slice, spline):
        spline_pts = spline.points.copy()
        slice_pts = slice.points.copy()

        cumdist = np.zeros(len(spline_pts), dtype=float)
        for i in range(1, len(spline_pts)):
            cumdist[i] = cumdist[i - 1] + np.linalg.norm(spline_pts[i] - spline_pts[i - 1])

        tree = cKDTree(spline_pts[:, :2])

        dists, nearest_idx = tree.query(slice_pts[:, :2], k=1)

        order_values = cumdist[nearest_idx]

        tie_break = dists
        sort_idx = np.lexsort((tie_break, order_values))

        ############## Helper #############

        def reorder_polydata_points_inplace(mesh: pv.PolyData, new_order: np.ndarray) -> pv.PolyData:
            mesh = mesh.copy()
            n = mesh.n_points
            new_order = np.asarray(new_order, dtype=np.int64)
            if new_order.shape[0] != n:
                raise ValueError("new_order must have length equal to mesh.n_points")

            inv = np.empty(n, dtype=np.int64)
            inv[new_order] = np.arange(n, dtype=np.int64)

            old_points = mesh.points.copy()
            mesh.points = old_points[new_order]

            for name in list(mesh.point_data.keys()):
                arr = mesh.point_data[name]
                mesh.point_data[name] = arr[new_order].copy()

            def remap_indexed_array(arr):
                if arr is None or arr.size == 0:
                    return arr
                arr = arr.copy()
                i = 0
                L = arr.size
                while i < L:
                    k = int(arr[i])
                    if k <= 0:
                        i += 1
                        continue
                    for j in range(i + 1, i + 1 + k):
                        old_id = int(arr[j])
                        arr[j] = int(inv[old_id])
                    i += k + 1
                return arr

            if mesh.verts is not None and mesh.verts.size > 0:
                mesh.verts = remap_indexed_array(mesh.verts)
            if mesh.lines is not None and mesh.lines.size > 0:
                mesh.lines = remap_indexed_array(mesh.lines)
            if mesh.faces is not None and mesh.faces.size > 0:
                mesh.faces = remap_indexed_array(mesh.faces)

            return mesh

        ############ Helper End ###########

        ordered_mesh = reorder_polydata_points_inplace(slice, new_order=sort_idx)

        ordered_mesh.point_data['ordered_idx'] = np.arange(ordered_mesh.n_points, dtype=float)

        return ordered_mesh

    def generate_icon_slice(self):
        '''
        Generates a 2D vertical slice along the defined pv spline through the vtk.
        Note that the function always seems to extend the slice beyond the bounds of the spline until it reaches the boundaries of the vtk.
        Thus, the file result needs to be filtered to contain only the relevant data.
        '''

        x_bounds = (self.pv_line.points[:, 0].min(), self.pv_line.points[:, 0].max())
        y_bounds = (self.pv_line.points[:, 1].min(), self.pv_line.points[:, 1].max())
        z_bounds = (self.icon_vtk.points[:, 2].min(), self.icon_vtk.points[:, 2].max())

        all_points = self.icon_vtk.points
        mask = (
                (all_points[:, 0] >= x_bounds[0]) & (all_points[:, 0] <= x_bounds[1]) &
                (all_points[:, 1] >= y_bounds[0]) & (all_points[:, 1] <= y_bounds[1]) &
                (all_points[:, 2] >= z_bounds[0]) & (all_points[:, 2] <= z_bounds[1])
        )

        clipped_vtk = self.icon_vtk.extract_points(mask, include_cells=True)

        self.slice = clipped_vtk.slice_along_line(self.pv_line, progress_bar=False)

        self.slice = self.reorder_slice_points_by_spline(self.slice, self.pv_line)

        if self.itopo is not None:
            x_bounds = (self.pv_line.points[:, 0].min(), self.pv_line.points[:, 0].max())
            y_bounds = (self.pv_line.points[:, 1].min(), self.pv_line.points[:, 1].max())
            z_bounds = (self.itopo.points[:, 2].min(), self.itopo.points[:, 2].max())

            all_points = self.itopo.points
            mask = (
                    (all_points[:, 0] >= x_bounds[0]) & (all_points[:, 0] <= x_bounds[1]) &
                    (all_points[:, 1] >= y_bounds[0]) & (all_points[:, 1] <= y_bounds[1]) &
                    (all_points[:, 2] >= z_bounds[0]) & (all_points[:, 2] <= z_bounds[1])
            )

            clipped_itopo = self.itopo.extract_points(mask, include_cells=True)

            self.z_slice = clipped_itopo.slice_along_line(self.pv_line, progress_bar=False)
            self.z_slice = self.reorder_slice_points_by_spline(self.z_slice, self.pv_line)

    @staticmethod
    def project2LOS(slice, station_coords, invert=True):
        points = slice.points

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        u = slice.get_array("u")
        v = slice.get_array("v")
        w = slice.get_array("w")
        pos = np.column_stack([x, y, z])
        uvec = np.column_stack([u, v, w])
        sensor = np.array([station_coords[0], station_coords[1], station_coords[2]])

        R = pos - sensor

        dist = np.linalg.norm(R, axis=1)

        tol = 1e-12
        near0 = dist < tol

        Rhat = np.empty_like(R)
        Rhat[~near0] = R[~near0] / dist[~near0, None]
        Rhat[near0] = np.nan

        v_rad = np.einsum('ij,ij->i', uvec, Rhat)

        #v_rad_toward_sensor = -v_rad

        v_rad[near0] = np.nan
        #v_rad_toward_sensor[near0] = np.nan

        if invert:
            return v_rad#_toward_sensor
        else:
            return v_rad

    def interpolate_icon_slice(self, rv_mode=False, station_coords=None, invert=True):
        '''
        Interpolates the slice according to the chosen interpolation method and stores the resulting arrays and arrays coordinates in the class.
        '''


        if self.scalar:
            concat_array = np.concatenate((self.slice.points, self.slice.get_array(self.plot_variable)[:, None]), axis=1)
            points = self.slice.points
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            if self.epsg == "EPSG:4326":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
                x, y = transformer.transform(x, y)

            x = np.array(x)
            y = np.array(y)

            # Compute consecutive distances
            dsts = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
            line_length = np.sum(dsts)
            lon = np.concatenate([[0], np.cumsum(dsts)])

            values = self.slice.get_array(self.plot_variable)

            loni = np.arange(lon.min(), lon.max(), self.grid_width)
            zi = np.arange(z.min(), z.max(), self.grid_width)
            loni, zi = np.meshgrid(loni, zi)

            grid_values = griddata((lon, z), values, (loni, zi), method=self.interp_method)

            self.grid_values = grid_values
            self.loni = loni
            self.zi = zi
            self.lon = lon
            self.z = z
            self.values = values

        else:
            concat_array = np.concatenate((self.slice.points, self.slice.get_array(self.plot_variable)), axis=1)

            pts = concat_array[:, 0:2]
            z_vals = concat_array[:, 2]

            if self.epsg == "EPSG:4326":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
                x, y = transformer.transform(x, y)

            if self.plot_variable == "wind":
                uvecs_3d = np.column_stack(
                    [self.slice.get_array("u"), self.slice.get_array("v"), self.slice.get_array("w")])
            else:
                print("Only wind as vector implemented...")
                return None

            N = len(pts)
            z_hat = np.array([0.0, 0.0, 1.0])

            same_next = np.logical_and(np.isclose(pts[1:, 0], pts[:-1, 0]), np.isclose(pts[1:, 1], pts[:-1, 1]))
            keep_mask = np.ones(N, dtype=bool)
            keep_mask[1:] = ~same_next

            unique_xy = pts[keep_mask]
            M = len(unique_xy)

            dx = np.diff(unique_xy[:, 0])
            dy = np.diff(unique_xy[:, 1])
            seg = np.hypot(dx, dy)
            s_unique = np.concatenate([[0.0], np.cumsum(seg)])  # length M

            if M >= 7:
                wl = 7 if (7 % 2 == 1) else 7 + 1
                x_s = savgol_filter(unique_xy[:, 0], wl, polyorder=2, mode='interp')
                y_s = savgol_filter(unique_xy[:, 1], wl, polyorder=2, mode='interp')
            else:
                x_s = unique_xy[:, 0].copy()
                y_s = unique_xy[:, 1].copy()

            dx_s = np.gradient(x_s)
            dy_s = np.gradient(y_s)
            speed_s = np.hypot(dx_s, dy_s)
            speed_s[speed_s < 1e-12] = np.nan
            tx = dx_s / speed_s
            ty = dy_s / speed_s
            valid = ~np.isnan(tx)
            if not valid.all():
                idx_valid = np.where(valid)[0]
                tx = np.interp(np.arange(M), idx_valid, tx[idx_valid])
                ty = np.interp(np.arange(M), idx_valid, ty[idx_valid])
            t_unique = np.vstack([tx, ty, np.zeros_like(tx)]).T

            n_unique = np.cross(np.array([0., 0., 1.]), t_unique)
            n_norm = np.linalg.norm(n_unique, axis=1)
            n_norm[n_norm < 1e-12] = 1.0
            n_unique = (n_unique.T / n_norm).T

            tree = cKDTree(unique_xy)
            dists, idx_nearest = tree.query(pts, k=1)

            s_samples = s_unique[idx_nearest]
            t_samples = t_unique[idx_nearest]
            n_samples = n_unique[idx_nearest]

            dot_un = np.einsum('ij,ij->i', uvecs_3d, n_samples)
            u_proj = uvecs_3d - dot_un[:, None] * n_samples

            u_along = np.einsum('ij,ij->i', u_proj, t_samples)
            u_vert = u_proj[:, 2]

            points = np.column_stack([s_samples, z_vals])
            interp_along = LinearNDInterpolator(points, u_along)
            interp_vert = LinearNDInterpolator(points, u_vert)

            interp_x = LinearNDInterpolator(points, uvecs_3d[:, 0])
            interp_y = LinearNDInterpolator(points, uvecs_3d[:, 1])
            interp_w = LinearNDInterpolator(points, uvecs_3d[:, 2])

            if rv_mode:
                self.rv_mode = True
                v_rad = self.project2LOS(slice = self.slice, station_coords=station_coords, invert=invert)
                interp_rad = LinearNDInterpolator(points, v_rad)

            s_grid = np.arange(s_unique.min(), s_unique.max() + self.grid_width, self.grid_width)
            z_grid = np.arange(z_vals.min(), z_vals.max(), self.grid_width)
            loni, zi = np.meshgrid(s_grid, z_grid)

            Ns = len(s_grid)
            Nz = len(z_grid)
            if rv_mode:
                grid_v_rad = np.empty((Nz, Ns))
            grid_u_along = np.empty((Nz, Ns))
            grid_u_vert = np.empty((Nz, Ns))

            grid_x = np.empty((Nz, Ns))
            grid_y = np.empty((Nz, Ns))
            grid_w = np.empty((Nz, Ns))

            for iz, z0 in enumerate(z_grid):
                pts_row = np.column_stack([s_grid, np.full(Ns, z0)])
                grid_u_along[iz, :] = interp_along(pts_row)
                grid_u_vert[iz, :] = interp_vert(pts_row)

                grid_x[iz, :] = interp_x(pts_row)
                grid_y[iz, :] = interp_y(pts_row)
                grid_w[iz, :] = interp_w(pts_row)

                if rv_mode:
                    grid_v_rad[iz, :] = interp_rad(pts_row)

            signed_offsets = np.einsum('ij,ij->i', uvecs_3d, n_samples)

            interp_offsets = LinearNDInterpolator(points, signed_offsets)
            grid_offsets = np.empty((Nz, Ns))

            for iz, z0 in enumerate(z_grid):
                pts_row = np.column_stack([s_grid, np.full(Ns, z0)])
                grid_offsets[iz, :] = interp_offsets(pts_row)

            self.grid_values_lon = grid_u_along  # Nz x Ns
            self.grid_values_z_new = grid_w  # Nz x Ns (original 3rd component on grid)
            self.grid_values_x = grid_x  # Nz x Ns
            self.grid_values_y = grid_y  # Nz x Ns
            self.grid_values_z = grid_w  # Nz x Ns    (mirror of grid_values_z_new to keep same names)
            self.loni = loni  # Nz x Ns (meshgrid of s, z)
            self.zi = zi  # Nz x Ns
            self.lon = s_samples  # length N, per-sample 'distance along' (maps to self.values)
            self.z = z_vals  # length N, per-sample z
            self.values = uvecs_3d  # N x 3 (original vectors u,v,w)
            self.offsets = grid_offsets  # Nz x Ns (interpolated signed offsets)
            if rv_mode:
                self.rv_grid = grid_v_rad

        if self.itopo is not None:
            h = self.z_slice.get_array("z_ifc")
            points = self.z_slice.points
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            if self.epsg == "EPSG:4326":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
                x, y = transformer.transform(x, y)

            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

            dsts = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
            line_length = np.sum(dsts)
            lon = np.concatenate([[0], np.cumsum(dsts)])
            loni = np.arange(lon.min(), lon.max(), self.grid_width)

            #zi = np.arange(self.slice.points[:,2].min(), self.slice.points[:,2].max(), self.grid_width)
            zmin = self.slice.points[:, 2].min()
            zmax = self.slice.points[:, 2].max()
            extent = zmax - zmin
            nz = max(2, int(np.round(extent / self.grid_width)) + 1)
            zi = np.linspace(zmin, zmax, nz)
            
            h_new = np.interp(loni, lon, h)

            if self.grid_values is not None:
                nz_ref, nx_ref = self.grid_values.shape
            else:
                nz_ref, nx_ref = self.grid_values_x.shape
            
            nx = min(nx_ref, h_new.shape[0])
            nz = min(nz_ref, zi.shape[0])
            
            h_new = h_new[:nx]
            loni  = loni[:nx]
            zi    = zi[:nz]
            
            if self.grid_values is not None:
                self.grid_values = self.grid_values[:nz, :nx]
            else:
                self.grid_values_x = self.grid_values_x[:nz, :nx]
                self.grid_values_y = self.grid_values_y[:nz, :nx]
                self.grid_values_z = self.grid_values_z[:nz, :nx]
                self.grid_values_lon = self.grid_values_lon[:nz, :nx]
                self.grid_values_z_new = self.grid_values_z_new[:nz, :nx]
                self.offsets = self.offsets[:nz, :nx]
                if self.rv_mode:
                    self.rv_grid = self.rv_grid[:nz, :nx]

            zi = zi[:, None]
            h_new = h_new[None, :]

            self.height_mask = zi > h_new
            self.loni = loni        # (nx,)
            self.zi = zi[:, 0] if zi.ndim == 2 else zi   # ensure (nz,)


    def clip_result_array(self):
        '''
        Used to filter for only the valid data points. Takes away the artifacts of the slice generation. Sets terrain
        and too high points to nan.
        '''

        if self.height_mask is not None:
            if self.scalar:
                self.grid_values = np.where(self.height_mask, self.grid_values, np.nan)
            else:
                self.grid_values_x = np.where(self.height_mask, self.grid_values_x, np.nan)
                self.grid_values_y = np.where(self.height_mask, self.grid_values_y, np.nan)
                self.grid_values_z = np.where(self.height_mask, self.grid_values_z, np.nan)
                self.grid_values_lon = np.where(self.height_mask, self.grid_values_lon, np.nan)
                self.grid_values_z_new = np.where(self.height_mask, self.grid_values_z_new, np.nan)
                self.offsets = np.where(self.height_mask, self.offsets, np.nan)
                if self.rv_mode:
                    self.rv_grid = np.where(self.height_mask, self.rv_grid, np.nan)
        else:
            lon_min, lon_max = self.lon.min(), self.lon.max()
            z_min, z_max = self.z.min(), self.z.max()
            lon_norm = (self.lon - lon_min) / (lon_max - lon_min)
            z_norm = (self.z - z_min) / (z_max - z_min)
            loni_norm = (self.loni - lon_min) / (lon_max - lon_min)
            zi_norm = (self.zi - z_min) / (z_max - z_min)

            pts = np.column_stack((lon_norm, z_norm))
            grid_pts = np.column_stack((loni_norm.ravel(), zi_norm.ravel()))

            tree = cKDTree(pts)
            distances, _ = tree.query(grid_pts, k=1)

            sample_dists, _ = tree.query(pts, k=2)
            nn_spacing = np.median(sample_dists[:, 1])

            threshold = 0.015 #0.015 #0.02

            if self.scalar:
                self.grid_values[distances.reshape(self.grid_values.shape) > threshold] = np.nan
            else:
                self.grid_values_x[distances.reshape(self.grid_values_x.shape) > threshold] = np.nan
                self.grid_values_y[distances.reshape(self.grid_values_y.shape) > threshold] = np.nan
                self.grid_values_z[distances.reshape(self.grid_values_z.shape) > threshold] = np.nan
                self.grid_values_lon[distances.reshape(self.grid_values_lon.shape) > threshold] = np.nan
                self.grid_values_z_new[distances.reshape(self.grid_values_z_new.shape) > threshold] = np.nan
                self.offsets[distances.reshape(self.offsets.shape) > threshold] = np.nan
                if self.rv_mode:
                    self.rv_grid[distances.reshape(self.rv_grid.shape) > threshold] = np.nan



    def return_interpolation_result(self):
        '''
        Function to return the interpolated results if desired.
        :return:
        '''

        result = {
            'extent': [self.lon.min(), self.lon.max(), self.z.min(), self.z.max()]
        }

        if self.scalar:
            result['type'] = 'scalar'
            result['grid_values'] = self.grid_values
        else:
            result['type'] = 'vector'
            result['grid_values'] = {
                'x': self.grid_values_x,
                'y': self.grid_values_y,
                'z': self.grid_values_z,
                'lon': self.grid_values_lon,
                'z_new': self.grid_values_z_new,
                'offset': self.offsets,
            }
            result['interpolation_grid'] = {
                'lon': self.loni,
                'z': self.zi
            }

        if self.rv_mode:
            result['grid_values']['rv_grid'] = self.rv_grid

        return result

    def vertical_profile_plot(self, plot_type='standard', nan_color="black",
                              cmap_name='viridis', label=None, discrete=True,
                              contour=True, bins=10, c_lines=10, c_color='black',
                              scale=100, density=2, save_as='test', offset=True):
        '''
        Function to automatically generate vertical plots. Options are standard for scalar data and quiver, streamplot and standard for vector data.
        Standard for vector generates a three-element plot displaying the x, y and z components just as scalar data.
        :param plot_type: Choose between standard and quiver, streamplot.
        :param nan_color: Color for nan values. Important for color of the terrain.
        :param cmap_name: Name of the desired colormap.
        :param label: Label to be displayed on the variable name.
        :param discrete: Discretize the colormap. Default is True.
        :param contour: Add contour lines to the plot. Default is True. Currently only works for standard plots
        :param bins: Number of bins for the discrete colormap.
        :param c_lines: Levels for the contour lines. Either integer or list of desired levels. Default is 10
        :param c_color: Color for the contour lines. Default is black.
        :param scale: Scale for the quiver plot. Default is 100.
        :param density: Density for the streamplot. Default is 2.
        :param save_as: Save under this variable name.
        '''

        extent = [self.lon.min(), self.lon.max(), self.z.min(), self.z.max()]

        if self.rv_mode:
            cmap = plt.get_cmap("coolwarm", bins).copy()
        else:
            cmap = plt.get_cmap(cmap_name, bins).copy()

        if label == None:
            label = self.plot_variable

        if self.scalar or self.rv_mode:
            if self.rv_mode:
                vmin, vmax = np.nanmin(self.rv_grid), np.nanmax(self.rv_grid)
            else:
                vmin, vmax = np.nanmin(self.grid_values), np.nanmax(self.grid_values)
            if discrete:
                colors = cmap(np.linspace(0, 1, bins))
                new_cmap = mcolors.ListedColormap(colors)
                new_cmap.set_bad(color=nan_color)
            else:
                new_cmap = cmap.copy()
                new_cmap.set_bad(color=nan_color)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))

            if self.rv_mode:
                img1 = ax.imshow(self.rv_grid, origin='lower', extent=extent, cmap=new_cmap, vmin=vmin, vmax=vmax)
            else:
                img1 = ax.imshow(self.grid_values, origin='lower', extent=extent, cmap=new_cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(img1, ax=ax, shrink=0.5, label=label)

            if contour:
                if self.rv_mode:
                    contour = ax.contour(self.rv_grid, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                else:
                    contour = ax.contour(self.grid_values, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                ax.clabel(contour, inline=True, fontsize=10, fmt="%.0f")

            ax.set_xlabel("Line Distance (m)")
            ax.set_ylabel("Altitude (m)")
            ax.set_aspect('equal')
            ax.set_title(f"Interpolated Vertical Plot Displaying {label}")
            ax.set_xlim([min(self.lon), max(self.lon)])
            ax.set_ylim([min(self.z), self.max_height[1]])

            plt.tight_layout()

            plt.savefig(f"{save_as}_Scalar_Standard.png", dpi=300, bbox_inches="tight")
            plt.show()

        else:

            if plot_type == 'quiver':
                X = self.loni
                Y = self.zi
                U = self.grid_values_lon
                V = self.grid_values_z_new
                magnitude = np.sqrt(U ** 2 + V ** 2)

                step = 50
                X_sub = X[::step, ::step]
                Y_sub = Y[::step, ::step]
                U_sub = U[::step, ::step]
                V_sub = V[::step, ::step]
                magnitude_sub = magnitude[::step, ::step]

                #print(X_sub.shape, Y_sub.shape, U_sub.shape, V_sub.shape, self.offsets[::step, ::step].shape)

                if offset:
                    vmin, vmax = np.nanmin(self.offsets[::step, ::step]), np.nanmax(self.offsets[::step, ::step])
                else:
                    vmin, vmax = np.nanmin(magnitude_sub), np.nanmax(magnitude_sub)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                if discrete:
                    colors = cmap(np.linspace(0, 1, bins))
                    new_cmap = mcolors.ListedColormap(colors)
                    new_cmap.set_bad(color=nan_color)
                else:
                    new_cmap = cmap.copy()
                    new_cmap.set_bad(color=nan_color)


                # Background
                bg = np.ones_like(self.grid_values_x)
                bg[np.isnan(self.grid_values_x)] = 0

                plt.figure(figsize=(10, 6))
                plt.imshow(bg, origin='lower', extent=extent, cmap="gray")
                if offset:
                    Q = plt.quiver(X_sub, Y_sub, U_sub, V_sub, self.offsets[::step, ::step], cmap=new_cmap, scale=scale,
                                   pivot='middle', norm=norm)
                else:
                    Q = plt.quiver(X_sub, Y_sub, U_sub, V_sub, magnitude_sub, cmap=new_cmap, scale=scale, pivot='middle', norm=norm)
                cbar = plt.colorbar(Q, ax=plt.gca(), shrink=0.5, label=label)
                cbar.set_label(label)
                plt.xlabel("Line Distance (m)")
                plt.ylabel("Altitude (m)")
                plt.title(f"Interpolated Vertical Quiver Plot Displaying {label}")
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.ylim([min(self.z), self.max_height[1]])
                plt.savefig(f"{save_as}_Quiver.png", dpi=300, bbox_inches="tight")
                plt.show()

            elif plot_type == 'streamplot':
                X = self.loni
                Y = self.zi
                U = self.grid_values_lon
                V = self.grid_values_z_new
                magnitude = np.sqrt(U ** 2 + V ** 2)

                step = 50
                X_sub = X[::step, ::step]
                Y_sub = Y[::step, ::step]
                U_sub = U[::step, ::step]
                V_sub = V[::step, ::step]
                magnitude_sub = magnitude[::step, ::step]

                if offset:
                    vmin, vmax = np.nanmin(self.offsets[::step, ::step]), np.nanmax(self.offsets[::step, ::step])
                else:
                    vmin, vmax = np.nanmin(magnitude_sub), np.nanmax(magnitude_sub)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                if discrete:
                    colors = cmap(np.linspace(0, 1, bins))
                    new_cmap = mcolors.ListedColormap(colors)
                    new_cmap.set_bad(color=nan_color)
                else:
                    new_cmap = cmap.copy()
                    new_cmap.set_bad(color=nan_color)

                # Background
                bg = np.ones_like(self.grid_values_x)
                bg[np.isnan(self.grid_values_x)] = 0

                plt.figure(figsize=(10, 6))
                plt.imshow(bg, origin='lower', extent=extent, cmap="gray")
                if offset:
                    plt.streamplot(X_sub, Y_sub, U_sub, V_sub, density=density, linewidth=1, color=self.offsets[::step, ::step],
                                   cmap=new_cmap, norm=norm)
                else:
                    plt.streamplot(X_sub, Y_sub, U_sub, V_sub, density=density, linewidth=1, color=magnitude_sub, cmap=new_cmap, norm=norm)
                sm = cm.ScalarMappable(cmap=new_cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=plt.gca(), shrink=0.5, label=label)

                plt.xlabel("Line Distance (m)")
                plt.ylabel("Altitude (m)")
                plt.title(f"Interpolated Vertical Streamplot Displaying {label}")
                plt.gca().set_aspect('equal')
                plt.ylim([min(self.z), self.max_height[1]])
                plt.savefig(f"{save_as}_Streamplot.png", dpi=300, bbox_inches="tight")
                plt.show()

            else:

                if discrete:
                    colors = cmap(np.linspace(0, 1, bins))
                    new_cmap = mcolors.ListedColormap(colors)
                    new_cmap.set_bad(color=nan_color)
                else:
                    new_cmap = cmap.copy()
                    new_cmap.set_bad(color=nan_color)

                fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(10, 18))

                img1 = ax[0].imshow(self.grid_values_x, origin='lower', extent=extent, cmap=new_cmap, vmin=np.nanmin(self.grid_values_x), vmax=np.nanmax(self.grid_values_x))
                fig.colorbar(img1, ax=ax[0], shrink=0.5, label=f'{label}_x_component')

                if contour:
                    contour = ax[0].contour(self.grid_values_x, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                    ax[0].clabel(contour, inline=True, fontsize=10, fmt="%.0fm/s")


                img2 = ax[1].imshow(self.grid_values_y, origin='lower', extent=extent, cmap=new_cmap, vmin=np.nanmin(self.grid_values_y), vmax=np.nanmax(self.grid_values_y))
                fig.colorbar(img2, ax=ax[1], shrink=0.5, label=f'{label}_y_component')

                if contour:
                    contour = ax[1].contour(self.grid_values_y, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                    ax[1].clabel(contour, inline=True, fontsize=10, fmt="%.0fm/s")


                img3 = ax[2].imshow(self.grid_values_z, origin='lower', extent=extent, cmap=new_cmap, vmin=np.nanmin(self.grid_values_z), vmax=np.nanmax(self.grid_values_z))
                fig.colorbar(img3, ax=ax[2], shrink=0.5, label=f'{label}_z_component')

                if contour:
                    contour = ax[2].contour(self.grid_values_z, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                    ax[2].clabel(contour, inline=True, fontsize=10, fmt="%.0fm/s")

                components = ["x", "y", "z"]
                for axis, component in zip(ax.flatten(), components):
                    axis.set_xlabel("Line Distance (m)")
                    axis.set_ylabel("Altitude (m)")
                    axis.set_aspect('equal')
                    axis.set_title(f"Interpolated Vertical Plot Displaying {label} {component}-Component")
                    axis.set_xlim([min(self.lon), max(self.lon)])
                    axis.set_ylim([min(self.z), self.max_height[1]])

                plt.tight_layout()

                plt.savefig(f"{save_as}_Vector_Standard.png", dpi=300, bbox_inches="tight")
                plt.show()

    def pv_3d_visualization(self):
        '''
        Creates a pyvista 3D visualization of the line and slice. Currently, it has some issues with the z-position due to rescaling to and from meters.
        '''
        vtk_copy = self.icon_vtk.copy()
        slice_copy = self.slice.copy()
        line_copy = self.pv_line.copy()
        line_points = line_copy.points.copy()
        line_points[:,2] = 0
        line_copy.points = line_points

        vtk_z_min = np.min(vtk_copy.points[:,2])
        vtk_z_max = np.max(vtk_copy.points[:,2])
        slice_points = slice_copy.points.copy()
        slice_min_z = np.min(slice_points[:,2])
        slice_max_z = np.max(slice_points[:,2])
        scale = (vtk_z_max - vtk_z_min) / (slice_max_z - slice_min_z)
        shift = vtk_z_min - slice_min_z * scale
        slice_points[:, 2] = scale * slice_points[:, 2] + shift
        slice_copy.points = slice_points

        plotter = pv.Plotter()
        plotter.add_mesh(vtk_copy.outline(), color="black")
        plotter.add_mesh(slice_copy, scalars=self.plot_variable, cmap="viridis", show_scalar_bar=True, show_edges=True) # Slice is still not at the perfect z position but works for now
        plotter.add_mesh(line_copy, color="red", line_width=5)
        #plotter.add_mesh(vtk_copy, scalars=self.plot_variable, cmap="viridis", show_scalar_bar=True, show_edges=True)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.show()

    def full_run(self, old_vtk=False, height_col_name='z_ifc', number_line_points=1000, autoplot=True, return_result=False, plot_3D=False,
                 plot_type='standard', nan_color="black",
                 cmap_name='viridis', label=None, discrete=True,
                 contour=True, bins=10, c_lines=10, c_color='black',
                 scale=100, density=2, save_as='test', rv_mode=False, invert=True, station_coords=None):

        '''
        Performes a full run of the vertical profile plot workflow and generate the result arrays. Return and automatically plots are optional.
        :param height_col_name: Name of the vtk array name containing height information in meters.
        :param number_line_points: Number of horizontal segments along the line.
        :param autoplot: If true, plots are automatically generated.
        :param return_result: If true, the interpolated result arrays are returned. Receiving variable need to be set according to the variable type (scalar or vector).
        :param plot_3D: If true, a pyvista 3D visualization is generated. Default is False.
        :param plot_type: Choose between standard and quiver, streamplot.
        :param nan_color: Color for nan values. Important for color of the terrain.
        :param cmap_name: Name of the desired colormap.
        :param label: Label to be displayed on the variable name.
        :param discrete: Discretize the colormap. Default is True.
        :param contour: Add contour lines to the plot. Default is True. Currently only works for standard plots
        :param bins: Number of bins for the discrete colormap.
        :param c_lines: Levels for the contour lines. Either integer or list of desired levels. Default is 10
        :param c_color: Color for the contour lines. Default is black.
        :param scale: Scale for the quiver plot. Default is 100.
        :param density: Density for the streamplot. Default is 2.
        :param save_as: Save under this variable name.
        :return:
        '''

        if old_vtk:
            self.adadpt_z(col_new=height_col_name)
        self.gpd_line_2_pv_line(number_line_points)
        self.generate_icon_slice()
        self.interpolate_icon_slice(rv_mode=rv_mode, invert=invert, station_coords=station_coords)
        self.clip_result_array()
        if plot_3D:
            self.pv_3d_visualization()
        if autoplot:
            self.vertical_profile_plot(plot_type=plot_type, nan_color=nan_color,
                                       cmap_name=cmap_name, label=label, discrete=discrete,
                                       contour=contour, bins=bins, c_lines=c_lines, c_color=c_color,
                                       scale=scale, density=density, save_as=save_as)
        if return_result:
            return self.return_interpolation_result()




def main():

    icon_vtk = pv.read(r"test_vtk.vtk")
    gdf_line = gpd.read_file(r"testline_25.geojson")

    VP_1 = VerticalPlotter(icon_vtk, "EPSG:32632", gdf_line, 'theta_v', max_height=[4000, 3800], grid_width=1, interp_method='linear')
    _ = VP_1.full_run(plot_type='standard', number_line_points=1000, save_as="Theta_Spline", plot_3D=False)



if __name__ == "__main__":
    main()
