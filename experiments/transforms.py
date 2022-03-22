from ctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train
from torch_geometric.transforms import LocalDegreeProfile, Compose
from ctg_benchmark.transforms.basics import RandomNormalNoise
from ctg_benchmark.transforms.graph import RandomAdjDropout
template_config_paths = ('./transforms/gcn.yaml',
                         './transforms/deeper_gcn.yaml',
                         )

for template_config_path in template_config_paths:
    for p, sigma in zip([0.5, 0.5, 0.7, 0.7], [0.1, 0.25, 0.1, 0.25]):
        template_config = load_yaml(template_config_path)
        name = f'_features_noise:{sigma}_adj_dropout:{p}'
        template_config['logs']['name'] += name
        template_config['loader']['train_dataset']['transform'] = Compose([RandomNormalNoise(noise_sigma=sigma)])
        train(config=template_config)  # perturbed training

    for p in [0.1, 0.5, 0.7]:
        template_config = load_yaml(template_config_path)
        name = f'_adj_dropout:{p}'
        template_config['logs']['name'] += name
        template_config['loader']['train_dataset']['transform'] = Compose([RandomAdjDropout(p=p)])
        train(config=template_config)  # perturbed training

    for sigma in [0.1, 0.25, 0.5]:
        template_config = load_yaml(template_config_path)
        name = f'_features_noise:{sigma}'
        template_config['logs']['name'] += name
        template_config['loader']['train_dataset']['transform'] = Compose([RandomNormalNoise(noise_sigma=sigma)])
        train(config=template_config)  # perturbed training

    template_config = load_yaml(template_config_path)
    name = f'_base'
    template_config['logs']['name'] += name
    train(config=template_config)  # not perturbed training
