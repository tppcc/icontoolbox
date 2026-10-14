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

def _discrete_colors(values, data_min, data_max, nlev, cmap_name):
    """
    Map values to a discrete (nlev-1)-colour palette over [data_min, data_max].

    Values are clamped into the range, so out-of-range cells take the end
    colours (equivalent to matplotlib's ``extend='both'``) instead of being
    dropped.  Returns (facecolors, norm, cmap, levels); the ``norm``/``cmap``
    pair is what the caller passes to ``fig.colorbar(ScalarMappable(...))``.
    """
    cmap = plt.get_cmap(cmap_name, nlev - 1)
    levels = np.linspace(data_min, data_max, nlev)
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    vals = np.asarray(values, dtype=np.float64)
    vclip = np.clip(vals, data_min, data_max)
    colors = cmap(norm(vclip))
    return colors, norm, cmap, levels


def icon_polycollection(data, polygons, data_min, data_max,
                        nlev=11, cmap='seismic',
                        edgecolor='face', linewidth=0.0,
                        transform=None):
    """
    Build a coloured ``PolyCollection`` for one ICON field.

    Parameters
    ----------
    data : array-like, shape (ncells,)
        Field values, one per source cell (last/only axis = cell dimension).
    polygons : dict
        Output of :func:`build_cell_polygons`.
    data_min, data_max : float
        Colour-scale limits.  Values outside take the end colours.
    nlev : int
        Number of level edges (=> nlev-1 discrete colours), as in the original.
    cmap : str
        Named matplotlib colormap.
    edgecolor : color or 'face'
        Polygon edge colour.  'face' matches the fill and hides mesh gaps;
        use a fixed colour (e.g. 'k') with ``linewidth>0`` to draw the mesh.
    linewidth : float
        Edge width (0 = no visible edges, fastest).
    transform : cartopy CRS or None
        Coordinate system of the vertices (use ``ccrs.PlateCarree()`` when
        adding to a projected GeoAxes).  ``None`` for a plain matplotlib axis.

    Returns
    -------
    dict with keys:
        collection : matplotlib.collections.PolyCollection
        norm, cmap : for ``fig.colorbar(ScalarMappable(norm, cmap))``
        levels     : the level edges used
    """
    data = np.asarray(data).ravel()
    if data.shape[0] != polygons['ncells']:
        raise ValueError(
            f"data has {data.shape[0]} cells but grid has {polygons['ncells']}"
        )

    values = data[polygons['src_index']]
    colors, norm, cmap_obj, levels = _discrete_colors(
        values, data_min, data_max, nlev, cmap
    )

    kw = dict(closed=True, facecolor=colors, edgecolor=edgecolor,
              linewidth=linewidth)
    if edgecolor == 'face':
        kw['edgecolor'] = colors          # per-polygon edge = per-polygon fill
    if transform is not None:
        kw['transform'] = transform

    coll = PolyCollection(polygons['verts'], **kw)
    return dict(collection=coll, norm=norm, cmap=cmap_obj, levels=levels)


# ---------------------------------------------------------------------------
# 3.  Convenience wrapper: add a field to an existing axis
# ---------------------------------------------------------------------------

def plot_icon_field(ax, data, grid, data_min, data_max,
                    nlev=11, cmap='seismic',
                    edgecolor='face', linewidth=0.0,
                    transform=None, polygons=None,
                    set_limits=False, autolim=True):
    """
    Draw one ICON field onto an existing axis as native cell polygons.

    The caller keeps full control of the figure, projection, title, ticks and
    colorbar -- this only adds the coloured collection, so it slots in next to
    hand-written matplotlib exactly like ``ax.contourf(...)`` would.

    Parameters
    ----------
    ax : matplotlib Axes or cartopy GeoAxes
        Target axis.  For a projected map, create it with a projection and
        pass ``transform=ccrs.PlateCarree()``.
    data : array-like, shape (ncells,)
        Field values (e.g. ``vars_all[1][itime, ilev, :]``).
    grid : xarray.Dataset
        ICON grid providing ``clon_vertices`` / ``clat_vertices``.
    data_min, data_max : float
        Colour-scale limits.
    nlev, cmap, edgecolor, linewidth, transform
        See :func:`icon_polycollection`.
    polygons : dict or None
        Precomputed :func:`build_cell_polygons` output.  Pass it to reuse the
        geometry across many fields / timesteps (the expensive part is built
        once); ``None`` builds it here.
    set_limits : bool
        If True, set the axis x/y limits to the geometry's bounding box.
        Leave False (default) when a projection or a chosen extent governs the
        view, to avoid fighting cartopy.
    autolim : bool
        Passed to ``ax.add_collection`` (let the collection update data limits).

    Returns
    -------
    dict
        Same as :func:`icon_polycollection`, plus ``polygons`` (reusable).

    Example
    -------
    >>> import cartopy.crs as ccrs
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(8, 5))
    >>> ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))
    >>> ax.coastlines('50m', linewidth=1)
    >>> # reuse geometry for several timesteps:
    >>> poly = build_cell_polygons(g)
    >>> out = plot_icon_field(ax, vars_all[1][itime, ilev, :], g, 230, 280,
    ...                       transform=ccrs.PlateCarree(), polygons=poly)
    >>> cb = fig.colorbar(plt.cm.ScalarMappable(norm=out['norm'], cmap=out['cmap']),
    ...                   ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    >>> cb.set_label('Temperature in K')
    """
    if polygons is None:
        polygons = build_cell_polygons(grid)

    out = icon_polycollection(
        data, polygons, data_min, data_max,
        nlev=nlev, cmap=cmap, edgecolor=edgecolor,
        linewidth=linewidth, transform=transform,
    )

    ax.add_collection(out['collection'], autolim=autolim)

    if set_limits:
        verts = polygons['verts']
        ax.set_xlim(verts[:, :, 0].min(), verts[:, :, 0].max())
        ax.set_ylim(verts[:, :, 1].min(), verts[:, :, 1].max())

    out['polygons'] = polygons
    return out


# ---------------------------------------------------------------------------
# 4.  Quick synthetic test (no grid file / no cartopy needed)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Synthetic test (no grid file needed) ===")

    class _FakeGrid:
        """Minimal stand-in exposing clon_vertices / clat_vertices (radians)."""
        def __init__(self, clon_deg, clat_deg):
            self.clon_vertices = np.deg2rad(clon_deg)
            self.clat_vertices = np.deg2rad(clat_deg)

    # Three triangles; the last straddles the +/-180 dateline.
    clon_deg = np.array([
        [  10.0,  12.0,  11.0],   # ordinary
        [ -40.0, -38.0, -39.0],   # ordinary
        [ 170.0, -170.0, 175.0],  # crosses the dateline
    ])
    clat_deg = np.array([
        [ 50.0, 50.0, 52.0],
        [-10.0, -8.0, -9.0],
        [  5.0,  6.0,  7.0],
    ])
    g = _FakeGrid(clon_deg, clat_deg)

    poly = build_cell_polygons(g)
    verts = poly['verts']

    # Non-negative field, one value per source cell.
    data = np.array([0.0, 0.5, 1.0])

    out = icon_polycollection(data, poly, data_min=0.0, data_max=1.0, nlev=6)
    facecolors = out['collection'].get_facecolor()

    print(f"source cells        = {poly['ncells']}")
    print(f"drawn polygons      = {verts.shape[0]}  (crossing cell duplicated)")
    print(f"facecolors          = {facecolors.shape[0]}")
    print(f"all lons in [-180,180]? "
          f"{bool((verts[:, :, 0] >= -180).all() and (verts[:, :, 0] <= 180).all())}")
    print(f"src_index           = {poly['src_index'].tolist()}")

    ok = (
        verts.shape[0] == poly['ncells'] + 1 and          # one cell split
        facecolors.shape[0] == verts.shape[0] and          # one colour per poly
        (verts[:, :, 0] >= -180).all() and
        (verts[:, :, 0] <= 180).all() and
        poly['src_index'][-1] == 2                          # duplicate maps back
    )
    print("PASSED" if ok else "FAILED")
