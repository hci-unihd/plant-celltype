from celltype.utils import load_full_stack

import numpy as np
from ipywidgets import widgets

import plotly.graph_objects as go
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class OvulesPlotter:
    def __init__(self, path):
        self.path = path
        self.stack = load_full_stack(path)
        self.voxel_size = self.stack['attributes']['element_size_um']
        
        self.cell_fig = go.FigureWidget()
        
        for _label in np.unique(self.stack['cell_labels']):
            mask = self.stack['cell_labels'] == _label
            self.cell_fig.add_scatter3d(x=[],
                                        y=[],
                                        z=[],
                                        meta=[],
                                        mode='markers',
                                        showlegend=True,
                                        name=str(_label),
                                        marker={'color': None,
                                                'size': 10})
        
        self._update_get_sizes()
        self._scale_coo()
        #size_widget = widgets.FloatSlider(value=1.0, min=0.1, max=10, step=0.5, description='Size:')
        shortest_distance = widgets.Checkbox(value=False)
        container = widgets.HBox(children=[shortest_distance])
        shortest_distance.observe(self._shortest_distance, names="value")
        
        self.fig = widgets.VBox([container, self.cell_fig])
        
        
    def _update_get_sizes(self, a=0.25):
        for i, _label in enumerate(np.unique(self.stack['cell_labels'])):
            mask = self.stack['cell_labels'] == _label
            _size = ((self.stack['cell_size_voxels'][mask].astype(np.float64) * (3/(4*np.pi)))**(1/3))
            self.cell_fig.data[i].marker['size'] = _size * a
            
    def _scale_coo(self):
        for i, _label in enumerate(np.unique(self.stack['cell_labels'])):
            mask = self.stack['cell_labels'] == _label
            
            coox = self.stack['cell_com_voxels'][:, 0][mask]
            coox = 0
            
            self.cell_fig.data[i].x = self.stack['cell_com_voxels'][:, 0][mask]*self.voxel_size[0]
            self.cell_fig.data[i].y = self.stack['cell_com_voxels'][:, 1][mask]*self.voxel_size[1]
            self.cell_fig.data[i].z = self.stack['cell_com_voxels'][:, 2][mask]*self.voxel_size[2]
            
    def _shortest_distance(self, value):
        print(value)
        if value['new']:
            shprtest_distance = self._compute_shortst_distance()
            for i, _label in enumerate(np.unique(self.stack['cell_labels'])):
                mask = self.stack['cell_labels'] == _label
                self.cell_fig.data[i].marker['color'] = shprtest_distance[mask]
        else:
            for i, _label in enumerate(np.unique(self.stack['cell_labels'])):
                self.cell_fig.data[i].marker['color'] = _label
    
    def _compute_shortst_distance(self, label=0):
        adj = csr_matrix((np.ones(self.stack['edges_ids'].shape[0]),
                         (self.stack['edges_ids'][:, 0], self.stack['edges_ids'][:, 1])),
                         shape=(self.stack['cell_ids'].max() + 1, self.stack['cell_ids'].max() + 1))
        adj = adj + adj.T
        distance = shortest_path(adj, indices=label)
        return distance[self.stack['cell_ids']]
        
        
    def _update_size_slider(self, value):
        #self.cell_fig.data[0].x = [0, 0]
        pass
