"""
Render ICON unstructured icosahedral grid data natively as map polygons.

Method: Native cell-polygon rendering (no interpolation).
Each ICON cell is drawn as its own polygon using the grid's vertex
coordinates (``clon_vertices``, ``clat_vertices``).  Data values map to
discrete colours and are attached as per-polygon face colours in a single
matplotlib ``PolyCollection``.

Why this method (and NOT regridding) for plotting:
  Regridding to a regular lat-lon mesh (see the companion regrid module)
  resamples the field and smooths sharp gradients.  For a *faithful* map of
  what the model actually computed, we draw the cells themselves:
    - no resampling, no smoothing, no invented values;
    - a cell's colour is a direct function of that one cell's value, so
      min(data) <= every drawn colour-level <= max(data);
    - the geometry is exactly the model mesh.
  Use this for publication maps of the raw field; use regridding when you
  need a rectangular array for arithmetic, differencing, or compositing.

Dateline handling:
  Triangles whose vertices straddle +/-180 deg would otherwise smear a
  horizontal band across the whole map.  Such cells are split into a western
  piece (positive lons clamped to -180) and a duplicated eastern piece
  (negative lons clamped to +180), and the value array is indexed through a
  ``src_index`` map so both pieces keep the correct colour.

Design:
  The module exposes small, composable primitives rather than one monolithic
  figure function, so it drops in alongside existing matplotlib/cartopy code:
    build_cell_polygons  -> reusable geometry (compute once per grid)
    icon_polycollection  -> a coloured PolyCollection (one per field/timestep)
    plot_icon_field      -> convenience: add the collection to an axis
  The caller still owns the figure, the projection, the axis limits and the
  colorbar, exactly as in hand-written matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection


# ---------------------------------------------------------------------------
# 1.  Build cell-polygon geometry (reusable across fields / timesteps)
# ---------------------------------------------------------------------------

def build_cell_polygons(grid, lon_split=100.0):
    """
    Build the per-cell polygon vertex array from an ICON grid.

    Vertices are read from ``clon_vertices`` / ``clat_vertices`` (radians) and
    converted to degrees.  Cells that cross the +/-180 deg dateline are split
    into two polygons (see module docstring).

    Parameters
    ----------
    grid : xarray.Dataset or object with ``clon_vertices`` / ``clat_vertices``
        ICON grid, each of shape (ncells, nverts) in radians.
    lon_split : float
        Threshold (degrees) for detecting a dateline-crossing cell: a cell is
        split if it has at least one vertex < -lon_split and one > +lon_split.

    Returns
    -------
    dict with keys:
        verts      : float array (npoly, nverts, 2) — (lon, lat) in degrees,
                     ready to hand to ``PolyCollection``.
        src_index  : int array (npoly,) — maps each polygon back to its source
                     cell, so ``data[src_index]`` gives per-polygon values.
        ncells     : int — number of source cells (npoly >= ncells).
    """
    clon = np.asarray(grid.clon_vertices) * (180.0 / np.pi)
    clat = np.asarray(grid.clat_vertices) * (180.0 / np.pi)
    ncells = clon.shape[0]

    # --- detect dateline-crossing cells ---
    crossing = np.any(clon < -lon_split, axis=1) & np.any(clon > lon_split, axis=1)
    idx = np.where(crossing)[0]

    lon = clon.copy()
    lat = clat.copy()
    src = np.arange(ncells)

    if idx.size:
        # western piece: clamp this cell's positive lons to -180
        block = lon[idx]
        block[clon[idx] > 0] = -180.0
        lon[idx] = block

        # eastern duplicate: clamp the original's negative lons to +180
        dup_lon = clon[idx].copy()
        dup_lon[clon[idx] < 0] = 180.0
        dup_lat = clat[idx].copy()

        lon = np.concatenate([lon, dup_lon], axis=0)
        lat = np.concatenate([lat, dup_lat], axis=0)
        src = np.concatenate([src, idx])

    verts = np.stack([lon, lat], axis=2)  # (npoly, nverts, 2)

    return dict(verts=verts, src_index=src, ncells=ncells)


# ---------------------------------------------------------------------------
# 2.  Map data to discrete colours + build the PolyCollection
# ---------------------------------------------------------------------------
