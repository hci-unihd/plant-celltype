from ctg_benchmark.loaders.torch_loader import get_cross_validation_loaders
from ctg_benchmark.evaluation.metrics import NodeClassificationMetrics, aggregate_class
import torch
import numpy as np
from collections import OrderedDict
import pathlib
from plantcelltype.graphnn.pl_models import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = pathlib.Path('baseline_checkpoints')
best_models = ['TgGIN_lr_1e-2_wd_1e-5_num_layers_2_hidden_feat_64_dropout_0.1',
               'TgGCN_lr_1e-2_wd_1e-5_num_layers_2_hidden_feat_128_dropout_0.5',
               'GAT2_lr_1e-3_wd_0_hidden_feat_256_heads_3_concat_True_batch_norm_False_drop_out_0.5',
               'GAT2v2_lr_1e-3_wd_1e-5_hidden_feat_256_heads_3_concat_True_batch_norm_False_drop_out_0.5',
               'TgGraphSAGE_lr_1e-3_wd_1e-5_num_layers_4_hidden_feat_128_dropout_0.1',
               'GCNII_lr_1e-2_wd_1e-5_hidden_feat_128_num_layers_4_dropout_0.0_shared_weights_False',
               'TransformerGCN2_lr_1e-3_wd_1e-5_hidden_feat_128_heads_3_concat_True_batch_norm_False_drop_out_0.5',
               'NoEdgesTransformerGCN2_lr_1e-3_wd_0_hidden_feat_128_heads_5_concat_True_batch_norm_False_drop_out_0.5',
               'NoEdgesDeeperGCN_lr_1e-3_wd_0_hidden_feat_128_num_layers_32_dropout_0',
               'DeeperGCN_lr_1e-3_wd_1e-5_hidden_feat_128_num_layers_16_dropout_0',
               ]


def load_model_from_checkpoint(model_name='TgGCN_lr_1e-2_wd_1e-5_num_layers_2_hidden_feat_128_dropout_0.5', split=0):
    # load full check point
    path = base_path / model_name / f'split{split}' / 'version_0' / 'checkpoints'
    path = path.glob('best_acc*.ckpt').__next__()
    pl_model = torch.load(path)

    # get model config and load model Module
    model_config = pl_model['hyper_parameters']['model']
    model = load_model(model_config['name'], model_kwargs=model_config['kwargs'])

    state_dict = pl_model['state_dict']
    # renaming keys to remove pl_module extra layer
    state_dict = OrderedDict([(key.replace('net.', ''), value) for key, value in state_dict.items()])
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def validation(validation_loader, model):
    # set up evaluation
    eval_metrics = NodeClassificationMetrics(num_classes=9)

    accuracy_records, accuracy_class_records = [], []
    model.eval()
    with torch.no_grad():
        for val_batch in validation_loader:
            val_batch = val_batch.to(device)
            val_batch = model.forward(val_batch)
            logits = torch.log_softmax(val_batch.out, 1)
            pred = logits.max(1)[1]

            # results is a dictionary containing a large number of classification metrics
            results = eval_metrics.compute_metrics(pred.cpu(), val_batch.y.cpu())
            acc = results['accuracy_micro']
            # aggregate class average the single class accuracy and ignores the embryo sack class (7)
            acc_class, _ = aggregate_class(results['accuracy_class'], index=7)

            accuracy_records.append(acc)
            accuracy_class_records.append(acc_class)
    return accuracy_records, accuracy_class_records


def cv_validation(loader, model_checkpoint_name):
    accuracy_records, accuracy_class_records = [], []
    for split, split_loader in loader.items():
        training_loader, validation_loader = split_loader['train'], split_loader['val']

        model = load_model_from_checkpoint(model_name=model_checkpoint_name,
                                           split=split)
        split_accuracy_records, split_accuracy_class_records = validation(validation_loader, model)
        accuracy_records += split_accuracy_records
        accuracy_class_records += split_accuracy_class_records

    # report results
    print(f'{model_checkpoint_name} results:')
    print(f'  Accuracy:               {np.mean(accuracy_records):.3f} '
          f'\u00b1 {np.std(accuracy_records):.3f}')
    print(f'  Class Average Accuracy: {np.mean(accuracy_class_records):.3f} '
          f'\u00b1 {np.std(accuracy_class_records):.3f}')


def main():
    # create data loader
    loader = get_cross_validation_loaders(root='./ctg_data', batch_size=1, shuffle=True, grs=('label_grs_surface',))
    for model_checkpoint_name in best_models:
        cv_validation(loader, model_checkpoint_name)


if __name__ == '__main__':
    main()
