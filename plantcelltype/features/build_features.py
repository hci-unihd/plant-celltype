import numpy as np
from plantcelltype.utils.io import import_labels_csv
from plantcelltype.utils import create_cell_mapping, map_cell_features2segmentation


def validate_dict(stack, mandatory_keys, feature_name='feature name'):
    for key in stack.keys():
        assert key in mandatory_keys, f'{key} missing, can not create {feature_name}'


def init_cell_labels(stack):
    stack['cell_ids'] = np.unique(stack['segmentation'])[1:]
    stack['cell_labels'] = np.zeros_like(stack['cell_ids'])
    return stack


def update_labels_from_csv(stack, csv_path):
    cell_ids, cell_labels = stack['cell_ids'], stack['cell_labels']

    csv_cell_ids, csv_cell_labels = import_labels_csv(csv_path)
    csv_labels_mapping = create_cell_mapping(csv_cell_ids, csv_cell_labels)

    for i, c_ids in enumerate(cell_ids):
        if c_ids in csv_labels_mapping.keys():
            label = csv_labels_mapping[c_ids]
        else:
            print(f"{c_ids} not found in csv file")
            label = 0

        cell_labels[i] = label

    stack['cell_labels'] = cell_labels
    stack['labels'] = map_cell_features2segmentation(stack['segmentation'], cell_ids, cell_labels)
    return stack

