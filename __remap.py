"""
Regrid ICON unstructured icosahedral grid data to a regular lat-lon grid.

Method: Area-weighted conservative binning (first-order conservative remapping).
Each unstructured cell is assigned to the lat-lon box containing its centre.
The target grid value is the area-weighted mean of all contributing cells.

Why this method preserves non-negativity:
  target_value = sum(w_i * x_i) / sum(w_i),  where w_i = cell_area_i > 0
  This is a convex combination (all weights positive, sum to 1), so:
    min(x_i) <= target_value <= max(x_i)
  If all x_i >= 0, then target_value >= 0.  No clipping needed.

Contrast with bilinear/barycentric interpolation, which uses signed weights
and CAN produce negative values from non-negative inputs at sharp gradients.
"""

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# 1.  Compute remapping weights (reusable across fields / timesteps)
# ---------------------------------------------------------------------------

def compute_remap_weights(grid_path, dlon=1.0, dlat=1.0,
                          lon_bounds=None, lat_bounds=None):
    """
    Precompute the mapping from unstructured cells -> regular lat-lon bins.

    Parameters
    ----------
    grid_path : str
        Path to ICON grid file (clat, clon in radians; optionally cell_area_p
        or cell_area).
    dlon, dlat : float
        Target grid spacing in degrees.
    lon_bounds, lat_bounds : (float, float) or None
        (min, max) in degrees.  Default: inferred from grid with 1-cell padding.

    Returns
    -------
    dict with keys:
        lat, lon        : 1-D centre coordinate arrays (degrees)
        lat_edges,
        lon_edges       : 1-D edge arrays
        bin_idx         : int array (ncells,) — flat index into (nlat, nlon)
        area            : float array (ncells,) — cell areas
        valid           : bool array (ncells,) — cells inside domain
        area_sum        : float array (nlat*nlon,) — total area per target cell
        nlat, nlon      : grid dimensions
    """
    gd = xr.open_dataset(grid_path)
    clat_deg = np.asarray(gd.clat) * (180.0 / np.pi)
    clon_deg = np.asarray(gd.clon) * (180.0 / np.pi)

    # --- cell area ---
    if 'cell_area_p' in gd:
        area = np.asarray(gd.cell_area_p).ravel()
    elif 'cell_area' in gd:
        area = np.asarray(gd.cell_area).ravel()
    else:
        # fallback: uniform weight (degrades to simple averaging)
        area = np.ones_like(clat_deg)

    gd.close()

    # --- target grid edges & centres ---
    if lon_bounds is None:
        lon_bounds = (np.floor(clon_deg.min()) - dlon,
                      np.ceil(clon_deg.max()) + dlon)
    if lat_bounds is None:
        lat_bounds = (max(-90.0, np.floor(clat_deg.min()) - dlat),
                      min(90.0,  np.ceil(clat_deg.max()) + dlat))

    lon_edges = np.arange(lon_bounds[0], lon_bounds[1] + dlon * 0.5, dlon)
    lat_edges = np.arange(lat_bounds[0], lat_bounds[1] + dlat * 0.5, dlat)
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    nlon = len(lon)
    nlat = len(lat)

    # --- bin each unstructured cell into a target box ---
    j = np.floor((clon_deg - lon_bounds[0]) / dlon).astype(int)
    i = np.floor((clat_deg - lat_bounds[0]) / dlat).astype(int)

    valid = (j >= 0) & (j < nlon) & (i >= 0) & (i < nlat)
    flat_idx = np.where(valid, i * nlon + j, 0)

    # --- pre-sum areas per target cell ---
    area_sum = np.zeros(nlat * nlon, dtype=np.float64)
    np.add.at(area_sum, flat_idx[valid], area[valid])

    return dict(
        lat=lat, lon=lon,
        lat_edges=lat_edges, lon_edges=lon_edges,
        bin_idx=flat_idx, area=area, valid=valid,
        area_sum=area_sum, nlat=nlat, nlon=nlon,
    )


# ---------------------------------------------------------------------------
# 2.  Apply weights to one or more fields
# ---------------------------------------------------------------------------

def apply_remap(data, weights, fill_nearest=True, max_fill_deg=None):
    """
    Apply precomputed remap weights to data.

    Parameters
    ----------
    data : array-like, shape (..., ncells)
        Source data.  Last axis must be the cell dimension.
    weights : dict
        Output of ``compute_remap_weights``.
    fill_nearest : bool
        Fill empty target cells using the nearest source cell value.
        Still preserves non-negativity (copies, no interpolation).
    max_fill_deg : float or None
        Max Euclidean distance (degrees) for nearest fill.
        Default: 3 * max(dlon, dlat) estimated from the grid spacing.

    Returns
    -------
    np.ndarray, shape (..., nlat, nlon)
    """
    data = np.asarray(data, dtype=np.float64)
    flat_idx = weights['bin_idx']
    area     = weights['area']
    valid    = weights['valid']
    area_sum = weights['area_sum']
    nlat     = weights['nlat']
    nlon     = weights['nlon']

    leading_shape = data.shape[:-1]
    ncells = data.shape[-1]
    nfields = int(np.prod(leading_shape)) if leading_shape else 1
    data2d = data.reshape(nfields, ncells)

    result = np.full((nfields, nlat * nlon), np.nan, dtype=np.float64)
    has_data = area_sum > 0

    for f in range(nfields):
        wsum = np.zeros(nlat * nlon, dtype=np.float64)
        # NaN-safe: treat NaN source values as zero contribution
        vals = data2d[f, valid].copy()
        w    = area[valid].copy()
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            vals[nan_mask] = 0.0
            w[nan_mask]    = 0.0
            # recompute area_sum without NaN cells for this field
            a_sum = np.zeros(nlat * nlon, dtype=np.float64)
            np.add.at(a_sum, flat_idx[valid], w)
            np.add.at(wsum, flat_idx[valid], vals * w)
            ok = a_sum > 0
            result[f, ok] = wsum[ok] / a_sum[ok]
        else:
            np.add.at(wsum, flat_idx[valid], vals * w)
            result[f, has_data] = wsum[has_data] / area_sum[has_data]

    # --- nearest-neighbour gap fill ---
    if fill_nearest:
        from scipy.spatial import cKDTree

        dlon_est = weights['lon'][1] - weights['lon'][0] if len(weights['lon']) > 1 else 1.0
        dlat_est = weights['lat'][1] - weights['lat'][0] if len(weights['lat']) > 1 else 1.0
        if max_fill_deg is None:
            max_fill_deg = 3.0 * max(dlon_est, dlat_est)

        empty_mask = (area_sum == 0).reshape(nlat, nlon)
        if empty_mask.any():
            # read source coordinates back from the grid weights
            # (we need original clat/clon — store lightweight version)
            # Use target-cell centres of filled cells to build tree
            lat_grid, lon_grid = np.meshgrid(weights['lat'], weights['lon'],
                                             indexing='ij')
            filled = (~empty_mask)
            tree = cKDTree(np.column_stack([lat_grid[filled],
                                            lon_grid[filled]]))
            empty_coords = np.column_stack([lat_grid[empty_mask],
                                            lon_grid[empty_mask]])
            dist, idx = tree.query(empty_coords)

            fill_ok = dist <= max_fill_deg
            for f in range(nfields):
                field = result[f].reshape(nlat, nlon)
                filled_vals = field[filled]
                field[empty_mask] = np.where(
                    fill_ok, filled_vals[idx], np.nan
                )

    return result.reshape(*leading_shape, nlat, nlon)


# ---------------------------------------------------------------------------
# 3.  Convenience wrapper: one-call regrid
# ---------------------------------------------------------------------------

def regrid_icon_to_latlon(grid_path, data, dlon=1.0, dlat=1.0,
                          lon_bounds=None, lat_bounds=None,
                          fill_nearest=True, max_fill_deg=None,
                          var_name='data'):
    """
    Regrid ICON unstructured data to a regular lat-lon grid.

    Uses area-weighted conservative binning — guarantees non-negativity
    for non-negative input (safe for concentrations, mixing ratios, fluxes).

    Parameters
    ----------
    grid_path : str
        Path to ICON grid description file.
    data : array-like, shape (..., ncells)
        Source field(s).  Last axis = unstructured cell dimension.
        Supports arbitrary leading dimensions (time, level, ensemble, …).
    dlon, dlat : float
        Target grid spacing (degrees).
    lon_bounds, lat_bounds : (float, float) or None
        Domain bounds (degrees).  Default: inferred from grid.
    fill_nearest : bool
        Fill empty target cells via nearest-neighbour (non-negative safe).
    max_fill_deg : float or None
        Max distance (degrees) for nearest fill.
    var_name : str
        Variable name in the output Dataset.

    Returns
    -------
    xr.Dataset  with dims (…, lat, lon)

    Example
    -------
    >>> # Single 2-D field
    >>> ds = regrid_icon_to_latlon('icon_grid.nc', conc, dlon=0.25, dlat=0.25)
    >>> ds['data'].plot()

    >>> # Precompute weights for many fields / timesteps
    >>> w = compute_remap_weights('icon_grid.nc', dlon=0.1, dlat=0.1)
    >>> t2m_ll  = apply_remap(t2m,  w)      # shape (nlat, nlon)
    >>> conc_ll = apply_remap(conc, w)       # same grid, reuses weights

    Notes
    -----
    Why area-weighted binning and NOT bilinear interpolation?
      Bilinear uses signed barycentric weights and can produce negative values
      from non-negative inputs at sharp concentration gradients.  Area-weighted
      binning is a convex combination (all weights > 0), so:
        min(sources) <= result <= max(sources)
    """
    weights = compute_remap_weights(
        grid_path, dlon=dlon, dlat=dlat,
        lon_bounds=lon_bounds, lat_bounds=lat_bounds,
    )
    regridded = apply_remap(data, weights,
                            fill_nearest=fill_nearest,
                            max_fill_deg=max_fill_deg)

    # --- build xarray output ---
    data_arr = np.asarray(data)
    leading_shape = data_arr.shape[:-1]

    if len(leading_shape) == 0:
        dims = ['lat', 'lon']
    else:
        dims = [f'dim_{i}' for i in range(len(leading_shape))] + ['lat', 'lon']

    ds = xr.Dataset(
        {var_name: (dims, regridded)},
        coords={'lat': weights['lat'], 'lon': weights['lon']},
    )
    ds.lat.attrs = {'units': 'degrees_north', 'long_name': 'latitude'}
    ds.lon.attrs = {'units': 'degrees_east',  'long_name': 'longitude'}

    return ds


# ---------------------------------------------------------------------------
# 4.  Quick synthetic test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Synthetic test (no grid file needed) ===")

    # Fake an unstructured grid: 50 000 random cells on a sphere patch
    np.random.seed(42)
    ncells = 50_000
    clat_deg = np.random.uniform(45, 55, ncells)
    clon_deg = np.random.uniform(5, 15, ncells)
    area = np.ones(ncells)

    # Non-negative "concentration" field with a sharp gradient
    conc = np.exp(-0.5 * ((clat_deg - 50)**2 + (clon_deg - 10)**2) / 4.0)
    conc = np.maximum(conc, 0.0)

    # --- manually build weights (bypass file I/O for test) ---
    dlon, dlat = 0.5, 0.5
    lon_bounds = (4.0, 16.0)
    lat_bounds = (44.0, 56.0)

    lon_edges = np.arange(lon_bounds[0], lon_bounds[1] + dlon * 0.5, dlon)
    lat_edges = np.arange(lat_bounds[0], lat_bounds[1] + dlat * 0.5, dlat)
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    nlon, nlat = len(lon), len(lat)

    j = np.floor((clon_deg - lon_bounds[0]) / dlon).astype(int)
    i = np.floor((clat_deg - lat_bounds[0]) / dlat).astype(int)
    valid = (j >= 0) & (j < nlon) & (i >= 0) & (i < nlat)
    flat_idx = np.where(valid, i * nlon + j, 0)
    area_sum = np.zeros(nlat * nlon)
    np.add.at(area_sum, flat_idx[valid], area[valid])

    w = dict(lat=lat, lon=lon, lat_edges=lat_edges, lon_edges=lon_edges,
             bin_idx=flat_idx, area=area, valid=valid,
             area_sum=area_sum, nlat=nlat, nlon=nlon)

    result = apply_remap(conc, w, fill_nearest=False)

    print(f"Input  min = {conc.min():.6f},  max = {conc.max():.6f}")
    print(f"Output min = {np.nanmin(result):.6f},  max = {np.nanmax(result):.6f}")
    print(f"Any negative? {(result[~np.isnan(result)] < 0).any()}")
    print(f"Output shape: {result.shape}  (nlat={nlat}, nlon={nlon})")
    print("PASSED" if not (result[~np.isnan(result)] < 0).any() else "FAILED")