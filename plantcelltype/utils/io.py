import h5py


def load_full_stack(path, keys=None):
    with h5py.File(path, 'r') as f:
        stacks = {'attributes': {}}
        for _key, _value in f.attrs.items():
            stacks['attributes'][_key] = _value

        if keys is None:
            keys = f.keys()

        for _key in keys:
            stacks[_key] = f[_key][...]

    return stacks


def check_keys(path):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
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
