import h5py
from plantcelltype.utils.axis_transforms import AxisTransformer


def load_full_stack(path, keys=None):
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

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in label_data:
            writer.writerow(data)


def import_labels_csv(cell_ids, cell_labels, path, csv_columns=('cell_ids', 'cell_labels')):
    label_data = []
    for c_ids, c_l in zip(cell_ids, cell_labels):
        label_data.append({csv_columns[0]: c_ids, csv_columns[1]: c_l})

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in label_data:
            writer.writerow(data)