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

Vertical cross-sections (height-longitude slices):
  The same philosophy is applied to vertical slices through the mesh.  Cells
  within a narrow latitude band are selected and ordered by longitude; each
  (cell, level) becomes a quad whose horizontal extent is the cell's own
  longitude footprint (from ``clon_vertices``) and whose vertical extent comes
  from the model-level interface heights -- again no interpolation, each quad's
  colour is a direct function of that one (cell, level) value.  The slice
  primitives mirror the map ones:
    build_slice_polygons      -> reusable (lon, height) quad geometry
    icon_slice_polycollection -> a coloured PolyCollection for one 2-D slice
    plot_icon_slice           -> convenience: add the slice to an axis
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
# 4.  Vertical cross-section: height-longitude slice on the native grid
# ---------------------------------------------------------------------------

def _full_to_ifc(z):
    """
    Turn full-level heights into (nlev+1) level-interface heights.

    Interior interfaces are midpoints between neighbouring full levels; the two
    outermost interfaces are reflected outward so every level keeps a finite
    thickness.  Operates along axis 0, so it accepts a 1-D array ``(nlev,)``
    (shared column) or a 2-D array ``(nlev, nsel)`` (terrain-following, one
    column per selected cell).
    """
    z = np.asarray(z, dtype=np.float64)
    mids = 0.5 * (z[:-1] + z[1:])
    top = z[:1] - (mids[:1] - z[:1])
    bot = z[-1:] + (z[-1:] - mids[-1:])
    return np.concatenate([top, mids, bot], axis=0)


def build_slice_polygons(grid, lat0, dlat=0.5, z=None, nlev=None,
                         z_is_interface=False, lon_range=None):
    """
    Build (longitude, height) quad geometry for a height-longitude slice.

    Cells whose centre latitude lies within ``+/-dlat`` of ``lat0`` are
    selected and ordered by longitude.  Each selected cell is paired with every
    model level to form a quad: its horizontal extent is the cell's own
    longitude footprint (min/max of ``clon_vertices``, in degrees) and its
    vertical extent is the gap between two level interfaces.  No interpolation
    is performed -- this is the vertical-slice analogue of
    :func:`build_cell_polygons`.

    Parameters
    ----------
    grid : xarray.Dataset or object with ``clon_vertices`` / ``clat_vertices``
        ICON grid, each of shape (ncells, nverts) in radians.
    lat0 : float
        Target latitude of the slice, in degrees.
    dlat : float
        Half-width (degrees) of the latitude band used to pick cells.  Widen it
        on coarse meshes if the band comes back empty; keep it small so the
        slice stays close to a single latitude.
    z : array-like or None
        Vertical coordinate.  ``None`` uses model-level indices (0..nlev) on the
        y-axis and then ``nlev`` is required.  A 1-D array gives a shared
        column; a 2-D array ``(nz, ncells)`` gives terrain-following heights
        (indexed down to the selected cells here).  By default ``z`` holds
        full-level heights; set ``z_is_interface=True`` if it already holds the
        nlev+1 interface heights.
    nlev : int or None
        Number of model levels.  Only needed (and only used) when ``z is None``.
    z_is_interface : bool
        See ``z``.  ``False`` (default): ``z`` are full levels, interfaces are
        derived.  ``True``: ``z`` are the nlev+1 interfaces, used as given.
    lon_range : (lon_min, lon_max) or None
        Optional longitude window (degrees) to clip the slice.  Longitudes are
        not unwrapped, so choose a window that does not cross +/-180.

    Returns
    -------
    dict with keys:
        verts      : float array (nlev*nsel, 4, 2) — (lon, height) quad corners,
                     ready to hand to ``PolyCollection``.  Ordered level-major
                     then cell, matching a ``(nlev, nsel)`` field raveled in
                     C-order.
        cell_index : int array (nsel,) — the source ICON cell ids, sorted by
                     longitude, so ``field[:, cell_index]`` builds the slice.
        shape      : (nlev, nsel) — expected shape of the 2-D slice field.
        ncells     : int — number of cells in the full grid.
        nlev, nsel : int — levels and selected columns.
        lon        : float array (nsel,) — selected cell-centre longitudes (deg).
    """
    clonv = np.asarray(grid.clon_vertices) * (180.0 / np.pi)
    clatv = np.asarray(grid.clat_vertices) * (180.0 / np.pi)
    ncells = clonv.shape[0]

    lon_c = clonv.mean(axis=1)
    lat_c = clatv.mean(axis=1)

    # --- select the latitude band and order the cells west-to-east ---
    sel = np.where(np.abs(lat_c - lat0) <= dlat)[0]
    if lon_range is not None:
        lo, hi = lon_range
        sel = sel[(lon_c[sel] >= lo) & (lon_c[sel] <= hi)]
    if sel.size == 0:
        raise ValueError(
            f"no cells within +/-{dlat} deg of lat={lat0}"
            + (f" and lon in {lon_range}" if lon_range is not None else "")
        )
    sel = sel[np.argsort(lon_c[sel])]
    nsel = sel.size

    # --- horizontal extent: each cell's own longitude footprint ---
    lon_lo = clonv[sel].min(axis=1)       # (nsel,)
    lon_hi = clonv[sel].max(axis=1)       # (nsel,)

    # --- vertical interfaces (shared 1-D, or per-column 2-D) ---
    if z is None:
        if nlev is None:
            raise ValueError("pass nlev when z is None (level-index y-axis)")
        z_ifc = np.arange(nlev + 1, dtype=np.float64)        # (nlev+1,)
    else:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2:                                       # (nz, ncells)
            if z.shape[1] != ncells:
                raise ValueError(
                    f"2-D z has {z.shape[1]} columns but grid has {ncells} cells"
                )
            z = z[:, sel]                                     # (nz, nsel)
        z_ifc = z if z_is_interface else _full_to_ifc(z)
        nlev = z_ifc.shape[0] - 1

    # --- assemble quads, vectorised, in (level, cell) C-order ---
    if z_ifc.ndim == 1:
        ztop = np.broadcast_to(z_ifc[:-1, None], (nlev, nsel))
        zbot = np.broadcast_to(z_ifc[1:, None],  (nlev, nsel))
    else:
        ztop = z_ifc[:-1]
        zbot = z_ifc[1:]
    xlo = np.broadcast_to(lon_lo[None, :], (nlev, nsel))
    xhi = np.broadcast_to(lon_hi[None, :], (nlev, nsel))

    corners = np.stack([
        np.stack([xlo, zbot], axis=-1),   # lower-left
        np.stack([xhi, zbot], axis=-1),   # lower-right
        np.stack([xhi, ztop], axis=-1),   # upper-right
        np.stack([xlo, ztop], axis=-1),   # upper-left
    ], axis=2)                            # (nlev, nsel, 4, 2)
    verts = corners.reshape(nlev * nsel, 4, 2)

    return dict(verts=verts, cell_index=sel, shape=(nlev, nsel),
                ncells=ncells, nlev=nlev, nsel=nsel, lon=lon_c[sel])


def icon_slice_polycollection(data2d, slice_poly, data_min, data_max,
                              nlev=11, cmap='seismic',
                              edgecolor='face', linewidth=0.0,
                              transform=None):
    """
    Build a coloured ``PolyCollection`` for one height-longitude slice.

    The vertical-slice analogue of :func:`icon_polycollection`.  ``data2d`` is
    the field on the slice, shape ``(nz, nsel)`` to match ``slice_poly`` exactly,
    or ``(nz, ncells)`` on the full grid -- in that case it is indexed down to
    the selected columns here for you.

    Parameters
    ----------
    data2d : array-like, shape (nz, nsel) or (nz, ncells)
        Field values at (level, cell) on the slice.  ``nz`` must equal the slice
        geometry's ``nlev``.
    slice_poly : dict
        Output of :func:`build_slice_polygons`.
    data_min, data_max : float
        Colour-scale limits.  Values outside take the end colours.
    nlev : int
        Number of colour-level edges (=> nlev-1 discrete colours), as elsewhere
        in this module.  (Distinct from the geometry's vertical level count.)
    cmap, edgecolor, linewidth, transform
        See :func:`icon_polycollection`.  ``transform`` stays ``None`` for the
        usual plain (longitude, height) axis.

    Returns
    -------
    dict with keys ``collection``, ``norm``, ``cmap``, ``levels`` -- as in
    :func:`icon_polycollection`.
    """
    data2d = np.asarray(data2d, dtype=np.float64)
    nlev_v, nsel = slice_poly['shape']

    if data2d.ndim != 2 or data2d.shape[0] != nlev_v:
        raise ValueError(
            f"data2d has {data2d.shape} but slice expects {nlev_v} levels"
        )
    if data2d.shape[1] == slice_poly['ncells']:
        data2d = data2d[:, slice_poly['cell_index']]
    elif data2d.shape[1] != nsel:
        raise ValueError(
            f"data2d has {data2d.shape[1]} columns but slice expects "
            f"{nsel} (selected) or {slice_poly['ncells']} (full grid)"
        )

    values = data2d.reshape(-1)           # level-major, matches verts order
    colors, norm, cmap_obj, levels = _discrete_colors(
        values, data_min, data_max, nlev, cmap
    )

    kw = dict(closed=True, facecolor=colors, edgecolor=edgecolor,
              linewidth=linewidth)
    if edgecolor == 'face':
        kw['edgecolor'] = colors
    if transform is not None:
        kw['transform'] = transform

    coll = PolyCollection(slice_poly['verts'], **kw)
    return dict(collection=coll, norm=norm, cmap=cmap_obj, levels=levels)


def plot_icon_slice(ax, data2d, grid, lat0, data_min, data_max,
                    dlat=0.5, z=None, nlev_v=None, z_is_interface=False,
                    lon_range=None, nlev=11, cmap='seismic',
                    edgecolor='face', linewidth=0.0, transform=None,
                    slice_poly=None, set_limits=True):
    """
    Draw one height-longitude slice onto an existing axis as native quads.

    The vertical-slice analogue of :func:`plot_icon_field`.  The caller keeps
    full control of the figure, title, ticks and colorbar; this only adds the
    coloured collection, so it slots in next to hand-written matplotlib exactly
    like ``ax.contourf(...)`` would.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axis (a plain axis: x = longitude in degrees, y = height).
    data2d : array-like, shape (nz, nsel) or (nz, ncells)
        Field on the slice (e.g. ``ds[VAR].isel(time=t).values``).
    grid : xarray.Dataset
        ICON grid providing ``clon_vertices`` / ``clat_vertices``.
    lat0, dlat, z, z_is_interface, lon_range
        Slice definition -- see :func:`build_slice_polygons`.  ``nlev_v`` maps to
        that function's ``nlev`` (number of model levels), needed only when
        ``z is None``.
    data_min, data_max, nlev, cmap, edgecolor, linewidth, transform
        Colour mapping -- see :func:`icon_slice_polycollection`.
    slice_poly : dict or None
        Precomputed :func:`build_slice_polygons` output.  Pass it to reuse the
        geometry across many fields / timesteps; ``None`` builds it here.
    set_limits : bool
        If True (default for a plain slice axis), set x/y limits to the quad
        bounding box.

    Returns
    -------
    dict
        Same as :func:`icon_slice_polycollection`, plus ``slice_poly`` (reusable).

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> # one timestep, full column: field shape (nz, ncells)
    >>> field = ds["ash_giant"].isel(time=0).values
    >>> sp = build_slice_polygons(grid, lat0=-20.545, dlat=0.5,
    ...                           z=grid.z_mc if hasattr(grid, "z_mc") else None,
    ...                           nlev=ds.sizes["height"])
    >>> fig, ax = plt.subplots(figsize=(9, 4))
    >>> out = plot_icon_slice(ax, field, grid, lat0=-20.545, dlat=0.5,
    ...                       data_min=vmin, data_max=vmax, slice_poly=sp,
    ...                       cmap="YlOrRd", nlev=11)
    >>> ax.set_xlabel("Longitude"); ax.set_ylabel("Height")
    >>> cb = fig.colorbar(plt.cm.ScalarMappable(norm=out['norm'], cmap=out['cmap']),
    ...                   ax=ax, fraction=0.025, pad=0.02)
    """
    if slice_poly is None:
        slice_poly = build_slice_polygons(
            grid, lat0, dlat=dlat, z=z, nlev=nlev_v,
            z_is_interface=z_is_interface, lon_range=lon_range,
        )

    out = icon_slice_polycollection(
        data2d, slice_poly, data_min, data_max,
        nlev=nlev, cmap=cmap, edgecolor=edgecolor,
        linewidth=linewidth, transform=transform,
    )

    ax.add_collection(out['collection'], autolim=False)

    if set_limits:
        verts = slice_poly['verts']
        ax.set_xlim(verts[:, :, 0].min(), verts[:, :, 0].max())
        ax.set_ylim(verts[:, :, 1].min(), verts[:, :, 1].max())

    out['slice_poly'] = slice_poly
    return out


# ---------------------------------------------------------------------------
# 5.  Quick synthetic test (no grid file / no cartopy needed)
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

    print("\n=== Synthetic slice test (height-longitude) ===")

    # Five cells spread in longitude, all near lat = -20 (one off-band).
    sclon = np.array([
        [ 10.0, 11.0, 10.5],
        [ 12.0, 13.0, 12.5],
        [ 14.0, 15.0, 14.5],
        [ 16.0, 17.0, 16.5],
        [ 18.0, 19.0, 18.5],
    ])
    sclat = np.array([
        [-20.1, -20.0, -19.9],
        [-20.1, -20.0, -19.9],
        [-20.1, -20.0, -19.9],
        [-25.0, -25.0, -24.9],   # outside the +/-0.5 deg band
        [-20.1, -20.0, -19.9],
    ])
    sg = _FakeGrid(sclon, sclat)

    nz = 4
    zmc = np.array([8000.0, 5000.0, 3000.0, 1000.0])    # full-level heights
    sp = build_slice_polygons(sg, lat0=-20.0, dlat=0.5, z=zmc)

    nsel = sp['nsel']
    # field on the FULL grid: (nz, ncells); gradient in level and longitude
    field = (np.arange(nz)[:, None] + 0.1 * np.arange(sclon.shape[0])[None, :])
    sout = icon_slice_polycollection(field, sp, data_min=0.0, data_max=5.0,
                                     nlev=6, cmap='viridis')
    sverts = sp['verts']
    sfaces = sout['collection'].get_facecolor()

    print(f"selected columns    = {nsel}  (expected 4; one cell off-band)")
    print(f"levels              = {sp['nlev']}")
    print(f"drawn quads         = {sverts.shape[0]}  (= nlev*nsel)")
    print(f"facecolors          = {sfaces.shape[0]}")
    print(f"lon ordered W->E?   = {bool((np.diff(sp['lon']) > 0).all())}")
    print(f"height span         = [{sverts[:, :, 1].min():.0f}, "
          f"{sverts[:, :, 1].max():.0f}]")

    ok_slice = (
        nsel == 4 and
        sp['nlev'] == nz and
        sverts.shape[0] == nz * nsel and
        sfaces.shape[0] == sverts.shape[0] and
        (np.diff(sp['lon']) > 0).all() and                  # west-to-east
        sverts.shape[1] == 4                                 # quads
    )
    print("PASSED" if ok_slice else "FAILED")
