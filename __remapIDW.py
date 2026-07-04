"""
Remap ICON unstructured icosahedral grid data to a regular lat-lon grid.

Method: Inverse Distance Weighting (IDW)
  - Output = convex combination of k nearest input neighbors
  - Guarantees output ∈ [min, max] of neighbors → non-negative inputs
    always produce non-negative outputs (safe for concentration, mixing
    ratio, etc.)

Implementation uses scipy.spatial.cKDTree for neighbor lookup on the
unit sphere and scipy.sparse for memory-efficient weight application.
Variables are remapped one at a time with explicit cleanup to avoid OOM.
"""

import gc
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def RemapIDW(
    data,
    grid,
    spacing,
    k=6,
    power=2,
    cell_dim="ncells",
):
    """
    Remap ICON unstructured grid data to a regular lat-lon grid.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input on the ICON unstructured grid.  Any variable containing
        ``cell_dim`` as a dimension is remapped; others are passed through.
    grid : str or xr.Dataset
        Path to ICON grid file **or** preloaded grid dataset.
        Must contain ``clat`` and ``clon`` (cell-center coordinates in
        **radians**) with dimension ``cell_dim``.
    spacing : float
        Target grid spacing in **degrees**.
    k : int, optional
        Number of nearest neighbors for IDW.  Default 6 matches the
        typical hexagonal ICON cell topology.
    power : float, optional
        Distance exponent.  Higher → more local interpolation.
        Default 2 (standard IDW).
    cell_dim : str, optional
        Name of the unstructured cell dimension.  Default ``"ncells"``.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Remapped data with ``cell_dim`` replaced by ``(lat, lon)``.
        All other dimensions and coordinates are preserved unchanged.

    Notes
    -----
    * The KDTree is built in 3-D Cartesian on the unit sphere so that
      Euclidean distance correctly ranks great-circle neighbors.
    * Angular (great-circle) distance is used for the IDW weights
      themselves, not the Euclidean chord.
    * Rows that land exactly on a source cell use pure nearest-neighbor
      (weight = 1) to avoid division by zero.
    * For global grids (longitude span > 350°) the output covers
      lat ∈ [-90, 90], lon ∈ [0, 360).  For limited-area grids the
      bounding box of the source cells is used (snapped to ``spacing``).
    """
    input_is_da = isinstance(data, xr.DataArray)

    if input_is_da:
        ds = data.to_dataset(name="__single_var__")
    else:
        ds = data

    # ---- load grid geometry ------------------------------------------------
    if isinstance(grid, str):
        gd = xr.open_dataset(grid)
    else:
        gd = grid

    clat_rad = gd.clat.values          # (ncells,) radians
    clon_rad = gd.clon.values          # (ncells,) radians
    n_source = clat_rad.shape[0]

    # ---- KDTree on unit-sphere Cartesian -----------------------------------
    src_xyz = _lonlat_to_xyz(clon_rad, clat_rad)
    tree = cKDTree(src_xyz)
    del src_xyz
    gc.collect()

    # ---- define target regular grid ----------------------------------------
    lat_deg = np.rad2deg(clat_rad)
    lon_deg = np.rad2deg(clon_rad)

    is_global = (lon_deg.max() - lon_deg.min()) > 350.0

    if is_global:
        lat_out = np.arange(-90.0, 90.0 + spacing * 0.5, spacing)
        lon_out = np.arange(0.0, 360.0, spacing)
    else:
        lat_lo = np.floor(lat_deg.min() / spacing) * spacing
        lat_hi = np.ceil(lat_deg.max() / spacing) * spacing
        lon_lo = np.floor(lon_deg.min() / spacing) * spacing
        lon_hi = np.ceil(lon_deg.max() / spacing) * spacing
        lat_out = np.arange(lat_lo, lat_hi + spacing * 0.5, spacing)
        lon_out = np.arange(lon_lo, lon_hi + spacing * 0.5, spacing)

    n_lat, n_lon = len(lat_out), len(lon_out)
    n_target = n_lat * n_lon

    # ---- query neighbors & build sparse weight matrix ----------------------
    lon_mesh, lat_mesh = np.meshgrid(np.deg2rad(lon_out), np.deg2rad(lat_out))
    tgt_xyz = _lonlat_to_xyz(lon_mesh.ravel(), lat_mesh.ravel())

    chord_dist, idx = tree.query(tgt_xyz, k=k)
    del tgt_xyz, tree, lon_mesh, lat_mesh
    gc.collect()

    W = _build_idw_sparse(chord_dist, idx, n_target, n_source, power)
    del chord_dist, idx
    gc.collect()

    # ---- remap each variable one at a time ---------------------------------
    result_vars = {}

    for var_name in list(ds.data_vars):
        da = ds[var_name]

        if cell_dim not in da.dims:
            # variable does not live on the unstructured grid → pass through
            result_vars[var_name] = da
            continue

        result_vars[var_name] = _remap_variable(
            da, W, n_lat, n_lon, lat_out, lon_out, cell_dim,
        )

        del da
        gc.collect()

    # ---- assemble output ---------------------------------------------------
    if input_is_da:
        return result_vars["__single_var__"]

    # preserve dataset-level coords that don't depend on cell_dim
    ds_coords = {
        name: coord
        for name, coord in ds.coords.items()
        if cell_dim not in coord.dims and name not in result_vars
    }
    return xr.Dataset(result_vars, coords=ds_coords, attrs=ds.attrs)


# ===================================================================
# Internal helpers
# ===================================================================

def _lonlat_to_xyz(lon_rad, lat_rad):
    """Convert lon/lat (radians) → 3-D Cartesian on the unit sphere."""
    cos_lat = np.cos(lat_rad)
    return np.column_stack([
        cos_lat * np.cos(lon_rad),
        cos_lat * np.sin(lon_rad),
        np.sin(lat_rad),
    ])


def _build_idw_sparse(chord_dist, indices, n_target, n_source, power):
    """
    Construct a sparse CSR weight matrix (n_target × n_source) from
    KDTree chord distances and neighbor indices.

    Weights are normalised so each row sums to 1 (convex combination).
    """
    # chord → angular distance on the unit sphere
    ang = 2.0 * np.arcsin(np.clip(chord_dist * 0.5, 0.0, 1.0))

    exact = ang < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(exact, 0.0, 1.0 / np.power(ang, power))

    # exact hits: nearest-neighbor fallback (avoids 1/0)
    has_exact = exact.any(axis=1)
    w[has_exact] = 0.0
    w[exact] = 1.0

    # row-normalise → convex combination → range-preserving
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    w /= row_sum

    k = indices.shape[1]
    rows = np.repeat(np.arange(n_target), k)
    cols = indices.ravel()
    vals = w.ravel()

    return csr_matrix((vals, (rows, cols)), shape=(n_target, n_source))


def _remap_variable(da, W, n_lat, n_lon, lat_out, lon_out, cell_dim):
    """
    Apply the sparse weight matrix *W* to one DataArray, replacing
    ``cell_dim`` with ``(lat, lon)`` while keeping every other
    dimension in its original position.
    """
    cell_axis = da.dims.index(cell_dim)
    vals = da.values

    # move cell axis to front for sparse matmul: (n_source, ...)
    vals = np.moveaxis(vals, cell_axis, 0)
    trailing = vals.shape[1:]
    vals_2d = vals.reshape(vals.shape[0], -1)           # (n_source, M)

    out_2d = W.dot(vals_2d)                              # (n_target, M)
    del vals, vals_2d

    out = out_2d.reshape((n_lat, n_lon) + trailing)
    del out_2d

    # restore axis order: (lat, lon, ...) → (..., lat, lon, ...)
    out = np.moveaxis(out, [0, 1], [cell_axis, cell_axis + 1])

    # ---- build xarray metadata -------------------------------------------
    new_dims = (
        list(da.dims[:cell_axis])
        + ["lat", "lon"]
        + list(da.dims[cell_axis + 1:])
    )

    new_coords = {
        name: coord
        for name, coord in da.coords.items()
        if cell_dim not in coord.dims
    }
    new_coords["lat"] = ("lat", lat_out, {"units": "degrees_north"})
    new_coords["lon"] = ("lon", lon_out, {"units": "degrees_east"})

    return xr.DataArray(out, dims=new_dims, coords=new_coords, attrs=da.attrs)