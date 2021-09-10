import csv
import os
import warnings

import h5py
import numpy as np
import tifffile
import yaml
from scipy.ndimage import zoom

from plantcelltype.utils.axis_transforms import AxisTransformer

TIFF_FORMATS = ['.tiff', '.tif']
H5_FORMATS = ['.h5', '.hdf']
LIF_FORMATS = ['.lif']


def read_tiff_voxel_size(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size
        return [z, y, x]


def read_h5_voxel_size(f, h5key):
    ds = f[h5key]

    # parse voxel_size
    if 'element_size_um' in ds.attrs:
        voxel_size = ds.attrs['element_size_um']
    else:
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]

    return voxel_size


def load_h5(path, key, slices=None, safe_mode=False):

    with h5py.File(path, 'r') as f:
        if key is None:
            key = list(f.keys())[0]

        if safe_mode and key not in list(f.keys()):
            return None, (1, 1, 1)

        if slices is None:
            file = f[key][...]
        else:
            file = f[key][slices]

        voxel_size = read_h5_voxel_size(f, key)

    return file, voxel_size


def load_tiff(path):
    file = tifffile.imread(path)
    try:
        voxel_size = read_tiff_voxel_size(path)

    except:
        # ZeroDivisionError could happen while reading the voxel size
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]

    return file, voxel_size


def load_lif(path):
    raise NotImplementedError


def smart_load(path, key=None, default=load_tiff):
    _, ext = os.path.splitext(path)
    if ext in H5_FORMATS:
        return load_h5(path, key)

    elif ext in TIFF_FORMATS:
        return load_tiff(path)

    elif ext in LIF_FORMATS:
        return load_lif(path)

    else:
        print(f"No default found for {ext}, reverting to default loader")
        return default(path)


def open_full_stack(path, keys=None):
    with h5py.File(path, 'r') as f:
        stacks = {'attributes': {}}
        for _key, _value in f.attrs.items():
            stacks['attributes'][_key] = _value

        if keys is None:
            keys = f.keys()

        for _key in keys:
            if isinstance(f[_key], h5py.Group):
                stacks[_key] = {}
                for __keys in f[_key].keys():
                    stacks[_key][__keys] = f[_key][__keys][...]

            elif isinstance(f[_key], h5py.Dataset):
                stacks[_key] = f[_key][...]

    return stacks, _load_axis_transformer(stacks['attributes'])


def export_full_stack(path, stack):
    for key, value in stack.items():
        if isinstance(value, dict):
            for group_key, group_value in value.items():
                if key == "attributes":
                    create_h5_attrs(path, group_value, group_key)
                else:
                    create_h5(path, group_value, key=f"{key}/{group_key}", voxel_size=None)

        elif isinstance(value, np.ndarray):
            if value.ndim == 3:
                voxel_size = stack['attributes'].get('element_size_um', [1.0, 1.0, 1.0])
            else:
                voxel_size = None
            create_h5(path, value, key=key, voxel_size=voxel_size)


def compute_scaling_factor(input_voxel_size, output_voxel_size):
    scaling = [i_size/o_size for i_size, o_size in zip(input_voxel_size, output_voxel_size)]
    return scaling


def scale_image(image, input_voxel_size, output_voxel_size, order=0):
    scaling = compute_scaling_factor(input_voxel_size, output_voxel_size)
    return zoom(image, scaling, order=order)


def import_segmentation(segmentation_path, extra_attr=None, out_voxel_size=None, key='segmentation'):
    segmentation, voxel_size = smart_load(segmentation_path, key=key)
    if out_voxel_size is not None:
        segmentation = scale_image(segmentation, voxel_size, out_voxel_size)
    else:
        out_voxel_size = voxel_size

    attributes = {'element_size_um': out_voxel_size, 'original_element_size_um': voxel_size}

    if extra_attr is not None:
        attributes.update(extra_attr)

    stack = {'segmentation': segmentation, 'attributes': attributes}
    return stack


def _load_axis_transformer(attributes):
    axis = attributes.get('global_reference_system_axis', ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    center = attributes.get('global_reference_system_origin', (0, 0, 0))
    voxel_size = attributes.get('element_size_um', (1., 1., 1.))
    return AxisTransformer(axis, center, voxel_size=voxel_size)


def check_keys(path, group=None):
    with h5py.File(path, 'r') as f:
        if group is None:
            keys = list(f.keys())
        else:
            keys = list(f[group].keys())
        attrs = list(f['attributes'].keys())

    return keys, attrs


def create_h5(path, stack, key, voxel_size=(1.0, 1.0, 1.0), mode='a'):
    del_h5_key(path, key)
    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        if voxel_size is not None:
            f[key].attrs['element_size_um'] = voxel_size


def create_h5_attrs(path, value, key, mode='a'):
    with h5py.File(path, mode) as f:
        f.attrs[key] = value


def del_h5_key(path, key, mode='a'):
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]


def export_labels_csv(cell_ids, cell_labels, path, csv_columns=('cell_ids', 'cell_labels')):
    label_data = []
    for c_ids, c_l in zip(cell_ids, cell_labels):
        label_data.append({csv_columns[0]: c_ids, csv_columns[1]: c_l})

    with open(path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for data in label_data:
            writer.writerow(data)


def import_labels_csv(path, csv_columns=('cell_ids', 'cell_labels')):
    cell_ids, cell_labels = [], []
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=csv_columns)
        reader = list(reader)
        for row in reader[1:]:
            cell_ids.append(row[csv_columns[0]])
            cell_labels.append(row[csv_columns[1]])

    return np.array(cell_ids, dtype='int32'), np.array(cell_labels, dtype='int32')


def _update(template_dict, up_dict):
    for key, value in up_dict.items():
        if isinstance(up_dict[key], dict) and key in template_dict:
            template_dict[key] = _update(template_dict[key], up_dict[key])
        else:
            template_dict[key] = up_dict[key]

    return template_dict


def load_yaml(config_path):
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def update(loader, node):
        with open(node.value, 'r') as _f:
            return yaml.full_load(_f)

    def home_path(loader, node):
        return os.path.expanduser('~')

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!home_path', home_path)
    yaml.add_constructor('!update', update)
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)

    if '_internal_variables' in config and 'template' in config['_internal_variables']:
        template_config = config['_internal_variables']['template']
        del template_config['_internal_variables']
        del config['_internal_variables']
        config = _update(template_config, config)
    else:
        if '_internal_variables' in config:
            del config['_internal_variables']

    return config
