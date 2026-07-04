import sys
import xarray as xr
import os
import numpy as np
import cdsapi
from datetime import datetime, timedelta
from .__smi import SWV2SMI
from .__config import load_default_config
import subprocess

"""
Subroutine for fetching and preprocessing all necessary IFS Reanalysis for ICON simulation

    This routine fetches the available dataset from DKRZ's collection of ERA Reanalysis,
    subsequently downloads the remaining variables from CDS Copernicus. The data is then
    regridded and projected to the closest resolution (0.25).
    Subsequent IFS->ICON transform is performed using ICON-TOOLS.

    Required dependencies:
        Climate Data Operator  (CDO) : Max Planck Institut fur Meteorologie
        ICON Tool                    : Deutscher Wetter Dienst
"""


def IFS4ICON(
    init_date,
    remap_nml=None,
    output_grid=None,
    output_path=None,
    config_yaml=None,
    iconremap_bin="/home/b/b382290/vol_work/icon_tools/dwd_icon_tools/icontools/iconremap",
):
    """
    Main entry point of IFS4ICON function. Prepare data for ICON simulation.

    Parameters
    ----------
    init_date : str
        Initialization date string in YYYY-MM-DD format.
    remap_nml : list of tuples or None
        Each tuple is (NAMELIST_ICONREMAP, NAMELIST_INPUT_FIELD) for one
        domain.  If None, namelists are auto-generated from the YAML
        ``iconremap`` section and ``output_grid`` is required.
    output_grid : str or None
        Path to ICON output grid file.  Required when ``remap_nml`` is None.
    output_path : str or None
        Output directory. Defaults to current working directory.
    config_yaml : str or None
        Path to custom YAML config. If None, uses the package default.
    iconremap_bin : str
        Path to the iconremap binary.

    End point: ifs2icon_<YYYYMMDD>_R<n>B<k>.nc
    """
    if output_path is None:
        output_path = os.getcwd()

    config = load_default_config(config_yaml)

    ml_path = config["paths"]["ml"]
    sf_path = config["paths"]["sf"]
    var_ml = config["var_ml"]
    var_sf = config["var_sf"]
    retrieve_vars = config["retrieve_vars"]
    spectral_vars = config["spectral_vars"]
    var_smi = config["var_smi"]

    init_vars = {
        "init_date": init_date,
        "output_path": output_path,
    }

    __cds_request(init_vars, retrieve_vars)

    __era_fetch(init_vars, var_ml, var_sf, ml_path, sf_path, spectral_vars)

    __swv_smi(init_vars, var_sf, retrieve_vars, var_smi)

    # ------------------------------------------------------------------
    # Resolve remap namelists
    # ------------------------------------------------------------------
    if remap_nml is None:
        if output_grid is None:
            raise ValueError(
                "output_grid is required when remap_nml is not provided. "
                "Supply either remap_nml (list of namelist-file tuples) or "
                "output_grid (path to the ICON grid file)."
            )
        remap_nml = __generate_remap_nml(
            config["iconremap"], output_path, init_date, output_grid
        )

    __ifs2icon(remap_nml, iconremap_bin)


# –––––––––––––––––––––––––––––––––––
# Helper functions
# –––––––––––––––––––––––––––––––––––


def __cds_request(init_vars, retrieve_vars):
    """Download invariant / surface fields from CDS and remap to Gaussian N320."""
    procs = []
    for vname, param in retrieve_vars.items():
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [vname],
            "date": init_vars["init_date"],
            "time": ["00:00", "12:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [90, -180, -90, 180],
        }

        f_in = os.path.join(init_vars["output_path"], f"temp_{param}.nc")
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(f_in)

        date_str = init_vars["init_date"]
        f_out = os.path.join(
            init_vars["output_path"], f"E5sf00_1H_{date_str}_{param}.nc"
        )
        proc = subprocess.Popen(
            ["bash", "-c", f"module load cdo && cdo remapcon,n320 {f_in} {f_out}"],
        )
        procs.append((proc, f_in))

    for proc, f_in in procs:
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"CDO remapcon failed for {f_in}")
        os.remove(f_in)


def __era_fetch(init_vars, var_ml, var_sf, ml_path, sf_path, spectral_vars):
    """Fetch ERA5 data from DKRZ archive and convert to NetCDF on Gaussian N320."""
    procs = []
    date_str = init_vars["init_date"]

    # Model level variables: spectral -> reduced Gaussian N320 regular
    for key, param in var_ml.items():
        fname = os.path.join(ml_path, f"{param}/E5ml00_1H_{date_str}_{param}.grb")
        outname = os.path.join(
            init_vars["output_path"], f"E5ml00_1H_{date_str}_{param}.nc"
        )
        if key in spectral_vars:
            cmd = (
                f"module load cdo && "
                f"cdo -f nc -t ecmwf -remapcon,n320 -setgridtype,regular -sp2gpl {fname} {outname}"
            )
        else:
            cmd = (
                f"module load cdo && "
                f"cdo -f nc -t ecmwf -remapcon,n320 -setgridtype,regular {fname} {outname}"
            )
        proc = subprocess.Popen(["bash", "-c", cmd])
        procs.append((proc, outname))

    # Surface level variables
    for key, param in var_sf.items():
        fname = os.path.join(sf_path, f"{param}/E5sf00_1H_{date_str}_{param}.grb")
        outname = os.path.join(
            init_vars["output_path"], f"E5sf00_1H_{date_str}_{param}.nc"
        )
        cmd = (
            f"module load cdo && "
            f"cdo -f nc -t ecmwf -setgridtype,regular {fname} {outname}"
        )
        proc = subprocess.Popen(["bash", "-c", cmd])
        procs.append((proc, outname))

    for proc, outname in procs:
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"CDO conversion failed for {outname}")


def __swv_smi(init_vars, var_sf, retrieve_vars, var_smi):
    """Convert volumetric soil water (SWVL) to Soil Moisture Index (SMI)."""
    date_str = init_vars["init_date"]
    fname_slt = os.path.join(
        init_vars["output_path"],
        f"E5sf00_1H_{date_str}_{retrieve_vars['soil_type']}.nc",
    )
    slt = xr.open_dataset(fname_slt)

    for idx in ["swvl1", "swvl2", "swvl3", "swvl4"]:
        param = var_sf[idx]
        f_in = os.path.join(
            init_vars["output_path"], f"E5sf00_1H_{date_str}_{param}.nc"
        )
        f_out = os.path.join(
            init_vars["output_path"], f"E5sf00_1H_{date_str}_{var_smi[param]}.nc"
        )
        swvl = xr.open_dataset(f_in)
        smi = SWV2SMI(swvl, slt)
        smi.to_netcdf(f_out)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Namelist auto-generation from YAML config
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def __generate_remap_nml(iconremap_cfg, output_path, init_date, output_grid):
    """Build ICON remap namelist files from the ``iconremap`` YAML config.

    Merges all referenced ERA5 NetCDF files into a single input file
    (``ifs_ana_<date>.nc``), then writes a NAMELIST_ICONREMAP and a
    NAMELIST_INPUT_FIELD suitable for ``iconremap``.

    Duplicate source files (e.g. GEOP_SFC and GEOP_ML both reading
    from the ML z variable) are de-duplicated for the merge step but
    kept as separate ``&input_field_nml`` blocks.

    Parameters
    ----------
    iconremap_cfg : dict
        The ``iconremap`` sub-dict from ``load_iconremap_config``.
    output_path : str
        Working / output directory.
    init_date : str
        Date string (YYYY-MM-DD).
    output_grid : str
        Path to the ICON grid file.

    Returns
    -------
    list of (str, str)
        Single-element list: ``[(NAMELIST_ICONREMAP, NAMELIST_INPUT_FIELD)]``.
    """
    remap_defaults = iconremap_cfg["remap_nml"]
    input_fields = iconremap_cfg["input_fields"]
    date_str = init_date

    # --- Collect unique source files for the merge ------------------------
    seen = set()
    merge_files = []
    for field in input_fields:
        key = (field["file_prefix"], field["file_param"])
        if key in seen:
            continue
        seen.add(key)
        fpath = os.path.join(
            output_path,
            f"{field['file_prefix']}_1H_{date_str}_{field['file_param']}.nc",
        )
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Expected ERA5 input file not found: {fpath}"
            )
        merge_files.append(fpath)

    # --- Merge into single input file -------------------------------------
    merged_file = os.path.join(output_path, f"ifs_ana_{date_str}.nc")
    subprocess.run(
        [
            "bash", "-c",
            f"module load cdo && cdo -O seltimestep,1 -merge {' '.join(merge_files)} {merged_file}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    out_file = os.path.join(output_path, f"ifs2icon_{date_str}.nc")

    # --- Write NAMELIST_ICONREMAP -----------------------------------------
    nml_remap_path = os.path.join(output_path, "NAMELIST_ICONREMAP")
    _write_remap_nml(
        nml_remap_path,
        remap_defaults,
        in_filename=merged_file,
        out_filename=out_file,
        out_grid_filename=output_grid,
    )

    # --- Write NAMELIST_INPUT_FIELD ---------------------------------------
    nml_input_path = os.path.join(output_path, "NAMELIST_INPUT_FIELD")
    _write_input_field_nml(nml_input_path, input_fields)

    return [(nml_remap_path, nml_input_path)]


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Fortran namelist writers
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def _write_remap_nml(
    filepath, defaults, *, in_filename, out_filename, out_grid_filename
):
    """Write a ``&remap_nml`` Fortran namelist file.

    Runtime file paths are written first, followed by every key-value pair
    from *defaults* (the YAML ``remap_nml`` block).  Adding a new parameter
    to the YAML is therefore sufficient — no code change needed.

    Parameters
    ----------
    filepath : str
        Destination path.
    defaults : dict
        Static namelist parameters from the YAML ``remap_nml`` block.
    in_filename, out_filename, out_grid_filename : str
        Paths injected into the namelist at runtime.
    """
    with open(filepath, "w") as f:
        f.write("&remap_nml\n")
        f.write(f'  in_grid_filename  = "{in_filename}"\n')
        f.write(f'  in_filename       = "{in_filename}"\n')
        f.write(f'  out_grid_filename = "{out_grid_filename}"\n')
        f.write(f'  out_filename      = "{out_filename}"\n')

        for key, val in defaults.items():
            f.write(f"  {key:18s} = {_fortran_literal(val)}\n")

        f.write("/\n")


def _write_input_field_nml(filepath, input_fields):
    """Write repeated ``&input_field_nml`` blocks to a Fortran namelist file.

    Internal bookkeeping keys (``file_prefix``, ``file_param``) are
    excluded from the output.

    Parameters
    ----------
    filepath : str
        Destination path.
    input_fields : list of dict
        Each dict becomes one ``&input_field_nml`` block.
    """
    _INTERNAL_KEYS = {"file_prefix", "file_param"}

    with open(filepath, "w") as f:
        for field in input_fields:
            f.write("&input_field_nml\n")
            for key, val in field.items():
                if key in _INTERNAL_KEYS:
                    continue
                f.write(f"  {key:16s} = {_fortran_literal(val)}\n")
            f.write("/\n")


def _fortran_literal(val):
    """Convert a Python value to a Fortran namelist literal."""
    if isinstance(val, bool):
        return ".TRUE." if val else ".FALSE."
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, float):
        # Preserve sign and avoid scientific notation for small values
        return f"{val}"
    return str(val)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ICON remap execution
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def __ifs2icon(remap_nml, iconremap_bin):
    """Execute ICON remap for each domain sequentially.

    Parameters
    ----------
    remap_nml : list of tuples
        Each element is (NAMELIST_ICONREMAP, NAMELIST_INPUT_FIELD).
        One tuple per domain (e.g. DOM01, DOM02) or a single tuple
        when namelists are auto-generated.
    iconremap_bin : str
        Absolute path to the iconremap binary.
    """
    for nml_remap, nml_input in remap_nml:
        # Clean up residual weight file from previous remap if present
        for f in _find_remap_residuals(nml_remap):
            os.remove(f)

        result = subprocess.run(
            [
                iconremap_bin,
                "--remap_nml", nml_remap,
                "--input_field_nml", nml_input,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Clean up residual weight file produced by remap
        for f in _find_remap_residuals(nml_remap):
            os.remove(f)


def _find_remap_residuals(nml_remap):
    """Find iconremap residual files (weight files ending in '0') in the same directory."""
    nml_dir = os.path.dirname(os.path.abspath(nml_remap))
    residuals = []
    for f in os.listdir(nml_dir):
        if f.endswith("0") and "remap" in f.lower():
            residuals.append(os.path.join(nml_dir, f))
    return residuals


def data_check(output_path, init_date):
    """Data integrity check on the final ICON input file."""
    ds = xr.open_dataset(
        os.path.join(output_path, f"ifs_ana_{init_date}.nc")
    )

    for var in ds.data_vars:
        data = ds[var].values
        if np.all(np.isnan(data)):
            print(f"{var}: WARNING - contains only NaN values")
        else:
            print(f"{var}: min={np.nanmin(data):.4g}, max={np.nanmax(data):.4g}")