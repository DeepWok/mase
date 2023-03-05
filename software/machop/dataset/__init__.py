import torch

from torch.utils.data import DataLoader

from .nlp import dataset_factory as nlp_dataset_factory
from .vision import build_dataset
from .dataloader import MyDataLoader
from machop.models import nlp_models


def get_dataset(name, nlp_task_args={}):
    if name in ['cifar10', 'cifar100', 'imagenet']:
        path = f'data/{name}'
        train_dataset, info = build_dataset(dataset_name=name,
                                            path=path,
                                            train=True)
        val_dataset, _ = build_dataset(dataset_name=name,
                                       path=path,
                                       train=False)
        test_dataset, _ = build_dataset(dataset_name=name,
                                        path=path,
                                        train=False)
    elif name in nlp_dataset_factory:
        # get dataset cls
        dataset_cls = nlp_dataset_factory[name]
        args = nlp_task_args
        # Classification style tasks
        if name in ['sst2', 'SST2']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        elif name in ['mnli', 'MNLI']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation_matched', **args)
            test_dataset = dataset_cls(split='test_matched', **args)
        elif name in ['qnli', 'QNLI']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        # Translation tasks
        elif name in [
                'iwslt2017_en_de', 'IWSLT2017_EN_DE', 'iwslt2017_de_en', 'IWSLT2017_DE_EN',
                'iwslt2017_en_fr', 'IWSLT2017_EN_FR', 'iwslt2017_en_ch', 'IWSLT2017_EN_CH',
                'wmt19_de_en', 'WMT19_DE_EN', 'wmt19_zh_en', 'WMT19_ZH_EN']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        elif name in ['opus_en_fr', 'OPUS_EN_FR']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        elif name in ['multi30k', 'MULTI30K']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        elif name in ['wmt16_ro_en', 'WMT16_RO_EN']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        # Language Modeling
        elif name in ['wikitext2', 'WIKITEXT2', 'wikitext103', 'WIKITEXT103']:
            train_dataset = dataset_cls(split='train', **args)
            val_dataset = dataset_cls(split='validation', **args)
            test_dataset = dataset_cls(split='test', **args)
        info = {'num_classes': train_dataset.num_classes}
    else:
        raise ValueError(f"Dataset {name} is not supported.")
    return train_dataset, val_dataset, test_dataset, info


def get_dataloader(name,
                   model,
                   train_dataset,
                   val_dataset,
                   test_dataset,
                   workers=4,
                   batch_size=64,
                   max_token_len=512):
    if name in nlp_models:
        tokenizer = model['tokenizer']
        train_dataset.setup_tokenizer(tokenizer, max_token_len)
        val_dataset.setup_tokenizer(tokenizer, max_token_len)
        test_dataset.setup_tokenizer(tokenizer, max_token_len)

    dataloader = MyDataLoader(dataset_name=name,
                              workers=workers,
                              train_dataset=train_dataset,
                              val_dataset=val_dataset,
                              test_dataset=test_dataset,
                              batch_size=batch_size)

    return dataloader
