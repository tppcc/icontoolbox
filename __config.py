from importlib.resources import files
import yaml


def load_config():
    config_path = files("icontoolbox.configs").joinpath("ifs4icon_param.yaml")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_default_config(config_yaml=None):
    """Load variable configuration from YAML.

    Parameters
    ----------
    config_yaml : str or None
        Path to a user-supplied YAML override. If None, the package-bundled
        default (ifs4icon_param.yaml) is used.

    Returns
    -------
    dict with keys: paths, var_ml, var_sf, retrieve_vars, spectral_vars,
                    var_smi, iconremap
    """
    if config_yaml is not None:
        with open(config_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return load_config()