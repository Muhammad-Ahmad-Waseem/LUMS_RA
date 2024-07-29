import os
import geopandas as gpd

__all__ = ['available', 'get_path']

_module_path = os.path.dirname(__file__)
_available_dir = [p for p in next(os.walk(_module_path))[1]
                  if not p.startswith('__')]
_available_zip = {} #{'nybb': 'nybb_16a.zip'}
available = _available_dir + list(_available_zip.keys())

def load_tindex_dataset_to_gdf(dataset):


    return gpd.read_file(get_src_tile_index(dataset))

def load_tindex_dataset_to_gdf(dataset):


    return gpd.read_file(get_train_tile_index(dataset))

def get_src_tile_index(dataset):
    """
    Get the path to the data file.
    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``geopandas.datasets.available`` for
        all options.
    """
    if dataset in _available_dir:
        return os.path.abspath(
            os.path.join(_module_path, dataset, '{}_SrcTindexex.geojson'.format(dataset)))
    elif dataset in _available_zip:
        fpath = os.path.abspath(
            os.path.join(_module_path, _available_zip[dataset]))
        return 'zip://' + fpath
    else:
        msg = "The dataset '{data}' is not available".format(data=dataset)
        raise ValueError(msg)


def get_train_tile_index(dataset):
    """
    Get the path to the data file.
    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``geopandas.datasets.available`` for
        all options.
    """
    if dataset in _available_dir:
        return os.path.abspath(
            os.path.join(_module_path, dataset, '{}_Train_Tindex.geojson'.format(dataset)))
    else:
        msg = "The dataset '{data}' is not available".format(data=dataset)
        raise ValueError(msg)

