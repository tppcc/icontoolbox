"""
Convert ERA5 Volumetric Soil Moisture (SWVL1-4) to Soil Moisture Index (SMI)
using the ERA5/HTESSEL soil type classification (1-7).
 
SMI = (theta - theta_PWP) / (theta_FC - theta_PWP)
 
    SMI = 0  -->  permanent wilting point (no plant water uptake)
    SMI = 1  -->  field capacity (optimal plant growth)
    SMI > 1  -->  above field capacity (gravitational drainage)
 
Soil hydraulic thresholds (PWP, FC) are from the ECMWF IFS documentation
(Balsamo et al., 2009; IFS CY41R2, Table 8.9; ECMWF CY32R3 release notes).
Type 7 (Tropical Organic) uses the same VG-derived values as Type 6 (Organic)
since HTESSEL assigns identical van Genuchten parameters to both.
 
References
----------
- Balsamo et al. (2009), J. Hydrometeor., doi:10.1175/2008JHM1068.1
- ECMWF IFS Documentation CY41R2, Part IV, Chapter 8, Table 8.9
- https://www.ecmwf.int/en/forecasts/documentation-and-support/evolution-ifs/
  cycles/change-soil-hydrology-scheme-ifs-cycle
 
Requirements
------------
    pip install numpy xarray netcdf4
 
Usage
-----
    from era5_swvl_to_smi import swvl_to_smi
 
    # ds_swvl contains one or more of swvl1-4; slt is a DataArray of soil type
    ds_smi = swvl_to_smi(ds_swvl, slt)
"""
 
import numpy as np
import xarray as xr
 
# ---------------------------------------------------------------------------
# HTESSEL soil hydraulic parameters
# ---------------------------------------------------------------------------
# Keys: ERA5 soil type code (integer value of parameter SLT / GRIB code 43)
# Values: (PWP [m3/m3], FC [m3/m3])
#
# Source: ECMWF IFS documentation & CY32R3 change notes.
# Type 7 (Tropical Organic) shares VG parameters with Type 6 (Organic).
# ---------------------------------------------------------------------------
# Lookup table for Soil type --> PWP & FC
SOIL_PARAMS = {
    1: {"name": "Coarse",           "pwp": 0.059, "fc": 0.242},
    2: {"name": "Medium",           "pwp": 0.151, "fc": 0.346},
    3: {"name": "Medium-fine",      "pwp": 0.133, "fc": 0.382},
    4: {"name": "Fine",             "pwp": 0.279, "fc": 0.448},
    5: {"name": "Very fine",        "pwp": 0.335, "fc": 0.541},
    6: {"name": "Organic",          "pwp": 0.267, "fc": 0.662},
    7: {"name": "Tropical Organic", "pwp": 0.151, "fc": 0.346},
}


# HTESSEL soil layer depths (bottom boundary) [m]
_LAYER_BOUNDS = {
    1: (0.0, 0.07),
    2: (0.07, 0.28),
    3: (0.28, 1.0),
    4: (1.0, 2.89),
}
 
 

def _build_lookup_arrays():
    """Build numpy arrays for vectorised PWP/FC lookup by soil type index.
 
    Returns arrays of length 8 (index 0 unused) so that
        pwp_arr[slt]  and  fc_arr[slt]
    give the correct values for slt in {1,...,7}.
    Index 0 is set to NaN (no valid soil type 0).
    """
    pwp = np.full(8, np.nan, dtype=np.float64)
    fc = np.full(8, np.nan, dtype=np.float64)
    for k, v in SOIL_PARAMS.items():
        pwp[k] = v["pwp"]
        fc[k] = v["fc"]
    return pwp, fc
 
 
def _compute_smi_array(theta, slt_int, pwp_arr, fc_arr):
    """Core numpy computation: (theta - PWP) / (FC - PWP), masked for invalid types."""
    theta = np.array(theta, dtype=np.float64)
    pwp = pwp_arr[slt_int]
    fc = fc_arr[slt_int]
    denom = fc - pwp
    denom = np.where(denom == 0, np.nan, denom)
    smi = (theta - pwp) / denom
    smi = np.where(slt_int == 0, np.nan, smi)
    smi = np.where(np.isnan(smi), -1e-6, smi)  # map NaN → sentinel
    return smi
 
 
def SWV2SMI(ds, slt):
    """Convert ERA5 volumetric soil moisture to Soil Moisture Index.
 
    Detects all ``swvl{1-4}`` variables in *ds*, converts each to SMI,
    and returns a new CF-compliant Dataset with variables renamed to
    ``smi{1-4}``.
 
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing one or more of ``swvl1``, ``swvl2``, ``swvl3``,
        ``swvl4`` (volumetric soil water content, m3 m-3).
    slt : xarray.DataArray or xarray.Dataset
        ERA5 soil type field (integer values 1-7, GRIB code 43).
        If a Dataset is passed, the soil type variable is auto-detected
        (looks for ``slt``, ``sotype``, or falls back to the first
        data variable).  Must be broadcastable to the spatial dimensions
        of *ds*.  Singleton ``time``, ``valid_time``, and ``step``
        dimensions are dropped automatically.
 
    Returns
    -------
    xarray.Dataset
        CF-compliant dataset with ``smi1`` .. ``smi4`` (for each input
        layer found), global attributes, and per-variable metadata
        following CF-1.8 conventions.
 
    Raises
    ------
    ValueError
        If no ``swvl{1-4}`` variables are found in *ds*.
    """
    pwp_arr, fc_arr = _build_lookup_arrays()

    # --- prepare soil type DataArray ----------------------------------------
    if isinstance(slt, xr.Dataset):
        _slt_candidates = ["slt", "sotype", "stl", "soil_type"]
        _slt_name = next((n for n in _slt_candidates if n in slt), None)
        if _slt_name is None:
            _slt_name = list(slt.data_vars)[0]
        slt_da = slt[_slt_name]
    else:
        slt_da = slt

    slt_da = slt_da.load() if hasattr(slt_da, "load") else slt_da.copy()

    # SLT is a static 2D field — strip everything except lat/lon.
    _SPATIAL = {"latitude", "longitude", "lat", "lon", "x", "y"}
    for dim in list(slt_da.dims):
        if dim.lower() not in _SPATIAL:
            slt_da = slt_da.isel({dim: 0}, drop=True)

    slt_vals = np.array(slt_da.values, dtype=np.float64, copy=True)
    slt_int = np.clip(np.rint(slt_vals).astype(int), 0, 7)

    # --- detect layers ------------------------------------------------------
    swvl_map = {}
    for lyr in (1, 2, 3, 4):
        vname = f"swvl{lyr}"
        if vname in ds:
            swvl_map[lyr] = vname
        elif vname.upper() in ds:
            swvl_map[lyr] = vname.upper()
    if not swvl_map:
        raise ValueError(
            "No SWVL variables found in dataset. "
            "Expected one or more of: swvl1, swvl2, swvl3, swvl4."
        )

    # --- compute SMI per layer ----------------------------------------------
    # Dimensions that are download/encoding artifacts — never physical axes.
    # Everything else (time, depth, lat, lon, …) is preserved.
    _ARTIFACT_DIMS = {"expver", "number"}

    smi_vars = {}
    for lyr, vname in sorted(swvl_map.items()):
        da = ds[vname].load() if hasattr(ds[vname], "load") else ds[vname]

        # Squeeze only known artifact dims (singleton), keep depth etc.
        da_squeezed = da
        for dim in list(da.dims):
            if dim.lower() in _ARTIFACT_DIMS and da.sizes[dim] == 1:
                da_squeezed = da_squeezed.isel({dim: 0}, drop=True)

        smi_vals = _compute_smi_array(da_squeezed.values, slt_int, pwp_arr, fc_arr)
        top, bot = _LAYER_BOUNDS[lyr]

        smi_da = da_squeezed.copy(data=smi_vals)
        smi_da.name = f"smil{lyr}"
        smi_da.attrs = {
            "standard_name": "soil_moisture_index",
            "long_name": f"Soil Moisture Index, layer {lyr} ({top:.2f}-{bot:.2f} m)",
            "units": "1",
            "valid_range": [0.0, np.nan],
            "comment": (
                "SMI = (theta - theta_PWP) / (theta_FC - theta_PWP). "
                "0 = permanent wilting point; 1 = field capacity; "
                ">1 = above field capacity."
            ),
            "cell_methods": "area: mean",
            "source_variable": vname,
            "layer_top_depth": f"{top} m",
            "layer_bottom_depth": f"{bot} m",
        }
        for key, val in ds[vname].attrs.items():
            if key.startswith("GRIB_"):
                smi_da.attrs[key] = val
        smi_vars[smi_da.name] = smi_da

    # --- assemble output dataset --------------------------------------------
    ds_smi = xr.Dataset(smi_vars)
    ds_smi.attrs = {
        "Conventions": "CF-1.8",
        "title": "Soil Moisture Index derived from ERA5 SWVL and HTESSEL soil type",
        "institution": "European Centre for Medium-Range Weather Forecasts",
        "source": "ERA5 volumetric soil water (SWVL1-4) and soil type (SLT)",
        "history": "SMI computed using HTESSEL PWP/FC lookup by soil type",
        "references": (
            "doi:10.1175/2008JHM1068.1; "
            "ECMWF IFS Documentation CY41R2, Part IV, Chapter 8, Table 8.9"
        ),
        "comment": (
            "PWP and FC values correspond to van Genuchten matric potentials "
            "of -15 bar and -0.1 bar respectively, for HTESSEL soil types 1-7."
        ),
    }
    for key, val in ds.attrs.items():
        ds_smi.attrs.setdefault(key, val)
    for coord in ds_smi.coords:
        if coord in ds.coords and ds[coord].attrs:
            ds_smi[coord].attrs.update(ds[coord].attrs)

    return ds_smi