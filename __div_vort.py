#!/usr/bin/env python3
"""Compute horizontal divergence and relative vorticity.

The public :func:`ComputeDV` routine supports both file-to-file processing and
in-memory xarray objects.  File input is opened lazily and chunked so that one
horizontal plane is processed at a time.  In-memory input is never reopened.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import TypeAlias

import numpy as np
import xarray as xr

try:  # Dask is only needed for memory-bounded file-to-file processing.
    import dask
except ImportError:  # pragma: no cover - depends on the runtime environment
    dask = None

R_EARTH = 6_371_000.0  # metres

PathLike: TypeAlias = str | os.PathLike[str]
InputData: TypeAlias = xr.Dataset | Iterable[xr.DataArray] | PathLike

_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")

__all__ = ["ComputeDV"]


def _coordinate_in_radians(coordinate: xr.DataArray) -> np.ndarray:
    """Return a one-dimensional angular coordinate in radians."""
    if coordinate.ndim != 1:
        raise ValueError(
            f"Coordinate {coordinate.name!r} must be one-dimensional; "
            "curvilinear latitude/longitude grids are not supported."
        )

    values = np.asarray(coordinate.values, dtype=np.float64)
    if values.size < 2:
        raise ValueError(
            f"Coordinate {coordinate.name!r} needs at least two points."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Coordinate {coordinate.name!r} contains non-finite values.")

    units = str(coordinate.attrs.get("units", "")).lower()
    if "radian" in units or units in {"rad", "rads"}:
        radians = values
    else:
        # CF latitude/longitude coordinates normally use degrees_north/east.
        radians = np.deg2rad(values)

    if np.any(np.diff(radians) == 0):
        raise ValueError(f"Coordinate {coordinate.name!r} contains duplicate points.")
    return radians


def _find_coordinate(
    array: xr.DataArray,
    aliases: tuple[str, ...],
    requested: str | None,
) -> xr.DataArray:
    """Find a latitude or longitude coordinate by explicit name or alias."""
    lookup = {str(name).lower(): str(name) for name in array.coords}

    if requested is not None:
        actual = lookup.get(requested.lower())
        if actual is None and requested in array.dims:
            actual = requested
        if actual is None:
            raise ValueError(
                f"Coordinate {requested!r} is not attached to variable {array.name!r}."
            )
        coordinate = array[actual]
    else:
        actual = next((lookup[name] for name in aliases if name in lookup), None)
        if actual is None:
            dim_lookup = {dim.lower(): dim for dim in array.dims}
            actual = next(
                (dim_lookup[name] for name in aliases if name in dim_lookup), None
            )
        if actual is None:
            kind = "/".join(aliases)
            raise ValueError(
                f"Could not find a {kind} coordinate on variable {array.name!r}."
            )
        coordinate = array[actual]

    if coordinate.ndim != 1 or coordinate.dims[0] not in array.dims:
        raise ValueError(
            f"Coordinate {coordinate.name!r} must be one-dimensional and index "
            f"variable {array.name!r}."
        )
    return coordinate


def _find_variable(dataset: xr.Dataset, requested: str, component: str) -> xr.DataArray:
    """Find a wind component by name, case-insensitively, then standard_name."""
    names = {name.lower(): name for name in dataset.data_vars}
    actual = names.get(requested.lower())
    if actual is not None:
        return dataset[actual]

    standard_name = "eastward_wind" if component == "u" else "northward_wind"
    matches = [
        array
        for array in dataset.data_vars.values()
        if str(array.attrs.get("standard_name", "")).lower() == standard_name
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Could not find the {component!r} wind variable {requested!r} in the dataset."
    )


def _wind_arrays(
    data: xr.Dataset | Iterable[xr.DataArray],
    u_name: str,
    v_name: str,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Extract the two wind components without opening any files."""
    if isinstance(data, xr.Dataset):
        return (
            _find_variable(data, u_name, "u"),
            _find_variable(data, v_name, "v"),
            dict(data.attrs),
        )
    if isinstance(data, xr.DataArray):
        raise TypeError(
            "Pass both wind components as an iterable, for example [u, v]."
        )

    try:
        arrays = tuple(data)
    except TypeError as exc:
        raise TypeError(
            "Input must be a path, an xarray.Dataset, or an iterable containing "
            "the u and v xarray.DataArray objects."
        ) from exc

    if len(arrays) != 2 or not all(isinstance(item, xr.DataArray) for item in arrays):
        raise TypeError(
            "The iterable input must contain exactly two xarray.DataArray objects."
        )

    # Named arrays may be supplied in either order.  Otherwise positional order
    # is interpreted as (u, v), which also supports unnamed DataArrays.
    by_name = {
        str(array.name).lower(): array for array in arrays if array.name is not None
    }
    named_u = by_name.get(u_name.lower())
    named_v = by_name.get(v_name.lower())
    if named_u is not None:
        u = named_u
        v = (
            named_v
            if named_v is not None
            else next(item for item in arrays if item is not u)
        )
    elif named_v is not None:
        v = named_v
        u = next(item for item in arrays if item is not v)
    else:
        u, v = arrays
    if u is v:
        raise ValueError(
            f"The iterable identifies the same DataArray as both {u_name!r} and {v_name!r}."
        )
    return u, v, {}


def _compute_div_vort(
    u: np.ndarray,
    v: np.ndarray,
    lat_rad: np.ndarray,
    lon_rad_or_dlat: np.ndarray | float,
    dlon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy kernel for horizontal divergence and relative vorticity.

    The last two axes of ``u`` and ``v`` must be latitude and longitude.  The
    preferred fourth argument is the complete longitude coordinate in radians,
    allowing non-uniform and descending coordinates.  Passing scalar ``dlat``
    and ``dlon`` retains compatibility with the module's former kernel API.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    lat_rad = np.asarray(lat_rad, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}.")
    if u.ndim < 2:
        raise ValueError("u and v must have latitude and longitude axes.")
    if u.shape[-2:] != (lat_rad.size, u.shape[-1]):
        raise ValueError("The latitude coordinate does not match the wind arrays.")

    if dlon is None:
        lon_spacing: np.ndarray | float = np.asarray(
            lon_rad_or_dlat, dtype=np.float64
        )
        if lon_spacing.ndim != 1 or lon_spacing.size != u.shape[-1]:
            raise ValueError("The longitude coordinate does not match the wind arrays.")
        lat_spacing: np.ndarray | float = lat_rad
    else:
        lat_spacing = float(lon_rad_or_dlat)
        lon_spacing = float(dlon)

    if u.shape[-2] < 2 or u.shape[-1] < 2:
        raise ValueError("Latitude and longitude dimensions need at least two points.")

    lat_edge_order = 2 if u.shape[-2] >= 3 else 1
    lon_edge_order = 2 if u.shape[-1] >= 3 else 1
    cos_shape = (1,) * (u.ndim - 2) + (lat_rad.size, 1)
    cos_lat = np.cos(lat_rad).reshape(cos_shape)
    if np.any(np.isclose(cos_lat, 0.0, atol=1e-12)):
        raise ValueError(
            "The spherical divergence/vorticity formula is singular at the poles."
        )

    du_dlon = np.gradient(
        u, lon_spacing, axis=-1, edge_order=lon_edge_order
    )
    dv_dlon = np.gradient(
        v, lon_spacing, axis=-1, edge_order=lon_edge_order
    )
    dvcoslat_dlat = np.gradient(
        v * cos_lat, lat_spacing, axis=-2, edge_order=lat_edge_order
    )
    ducoslat_dlat = np.gradient(
        u * cos_lat, lat_spacing, axis=-2, edge_order=lat_edge_order
    )

    inv_rcoslat = 1.0 / (R_EARTH * cos_lat)
    div = (inv_rcoslat * (du_dlon + dvcoslat_dlat)).astype(np.float32)
    vort = (inv_rcoslat * (dv_dlon - ducoslat_dlat)).astype(np.float32)
    return div, vort


def _compute_xarray(
    data: xr.Dataset | Iterable[xr.DataArray],
    *,
    u_name: str,
    v_name: str,
    lat_name: str | None,
    lon_name: str | None,
) -> xr.Dataset:
    """Compute div/vort while preserving arbitrary non-horizontal dimensions."""
    u, v, global_attrs = _wind_arrays(data, u_name, v_name)
    latitude = _find_coordinate(u, _LAT_NAMES, lat_name)
    longitude = _find_coordinate(u, _LON_NAMES, lon_name)
    lat_dim = latitude.dims[0]
    lon_dim = longitude.dims[0]
    if lat_dim == lon_dim:
        raise ValueError("Latitude and longitude must index different dimensions.")
    if lat_dim not in v.dims or lon_dim not in v.dims:
        raise ValueError("u and v must use the same latitude and longitude dimensions.")

    # Exact alignment prevents a silent inner join or coordinate reordering.
    try:
        u, v = xr.align(u, v, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("u and v coordinates must match exactly.") from exc
    u, v = xr.broadcast(u, v)

    lat_rad = _coordinate_in_radians(latitude)
    lon_rad = _coordinate_in_radians(longitude)
    div, vort = xr.apply_ufunc(
        _compute_div_vort,
        u,
        v,
        xr.DataArray(lat_rad, dims=(lat_dim,)),
        xr.DataArray(lon_rad, dims=(lon_dim,)),
        input_core_dims=[
            [lat_dim, lon_dim],
            [lat_dim, lon_dim],
            [lat_dim],
            [lon_dim],
        ],
        output_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )

    # apply_ufunc moves core dimensions to the end; restore the wind layout.
    div = div.transpose(*u.dims).rename("div")
    vort = vort.transpose(*u.dims).rename("vort")
    div.attrs = {
        "standard_name": "divergence_of_wind",
        "long_name": "Horizontal divergence",
        "units": "s-1",
    }
    vort.attrs = {
        "standard_name": "atmosphere_relative_vorticity",
        "long_name": "Relative vorticity",
        "units": "s-1",
    }

    result = xr.Dataset({"div": div, "vort": vort})
    result.attrs = global_attrs
    history = "Horizontal divergence and relative vorticity computed by icontoolbox"
    if result.attrs.get("history"):
        history = f"{result.attrs['history']}\n{history}"
    result.attrs["history"] = history
    return result


def _output_encoding(result: xr.Dataset, lat_dim: str, lon_dim: str) -> dict:
    """Build compression/chunk settings for plane-wise netCDF output."""
    chunks = tuple(
        result.sizes[dim] if dim in {lat_dim, lon_dim} else 1
        for dim in result["div"].dims
    )
    settings = {
        "dtype": "float32",
        "zlib": True,
        "complevel": 1,
        "shuffle": True,
        "_FillValue": np.float32(np.nan),
        "chunksizes": chunks,
    }
    return {"div": dict(settings), "vort": dict(settings)}


def ComputeDV(
    data: InputData,
    output: PathLike | None = None,
    *,
    u_name: str = "u",
    v_name: str = "v",
    lat_name: str | None = None,
    lon_name: str | None = None,
) -> xr.Dataset | Path:
    """Compute horizontal divergence and relative vorticity.

    Parameters
    ----------
    data
        An input netCDF path, an :class:`xarray.Dataset`, or any iterable
        containing exactly two :class:`xarray.DataArray` objects in ``(u, v)``
        order.  Named arrays may be supplied in either order.
    output
        Output netCDF path.  It is required for path input and optional for
        in-memory input.  With no output, the computed xarray Dataset is
        returned directly.
    u_name, v_name
        Wind variable names.  Matching is case-insensitive; CF
        ``standard_name`` attributes are also recognised.
    lat_name, lon_name
        Optional explicit coordinate names.  By default both ``lat``/``lon``
        and ``latitude``/``longitude`` are recognised.

    Returns
    -------
    xarray.Dataset or pathlib.Path
        An in-memory/lazy Dataset when ``output`` is omitted, otherwise the
        path of the written netCDF file.
    """
    is_path = isinstance(data, (str, os.PathLike))
    if is_path and output is None:
        raise ValueError("An output path is required when the input is a file path.")

    source: xr.Dataset | Iterable[xr.DataArray]
    opened: xr.Dataset | None = None
    if is_path:
        input_path = Path(data)
        output_path = Path(output)  # type: ignore[arg-type]
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must be different.")

        # Native chunks keep the file lazy while coordinate/dimension names are
        # discovered.  Rechunk below to make each horizontal core a whole plane.
        opened = xr.open_dataset(input_path, chunks={} if dask is not None else None)
        u = _find_variable(opened, u_name, "u")
        latitude = _find_coordinate(u, _LAT_NAMES, lat_name)
        longitude = _find_coordinate(u, _LON_NAMES, lon_name)
        lat_dim, lon_dim = latitude.dims[0], longitude.dims[0]
        relevant_dims = set(u.dims) | set(_find_variable(opened, v_name, "v").dims)
        chunk_map = {
            dim: -1 if dim in {lat_dim, lon_dim} else 1 for dim in relevant_dims
        }
        source = opened.chunk(chunk_map) if dask is not None else opened
    else:
        source = data  # type: ignore[assignment]

    try:
        result = _compute_xarray(
            source,
            u_name=u_name,
            v_name=v_name,
            lat_name=lat_name,
            lon_name=lon_name,
        )
        if output is None:
            return result

        output_path = Path(output)
        latitude = _find_coordinate(result["div"], _LAT_NAMES, lat_name)
        longitude = _find_coordinate(result["div"], _LON_NAMES, lon_name)
        encoding = _output_encoding(result, latitude.dims[0], longitude.dims[0])
        scheduler = (
            dask.config.set(scheduler="synchronous")
            if dask is not None
            else nullcontext()
        )
        with scheduler:
            result.to_netcdf(output_path, encoding=encoding)
        return output_path
    finally:
        # For file output the dask write is complete before this closes.  A
        # path input can never return a lazy result because output is required.
        if opened is not None:
            opened.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute horizontal divergence/vorticity from u and v."
    )
    parser.add_argument("input", type=Path, help="input netCDF containing u and v")
    parser.add_argument("output", type=Path, help="output netCDF for div and vort")
    arguments = parser.parse_args()
    ComputeDV(arguments.input, arguments.output)
