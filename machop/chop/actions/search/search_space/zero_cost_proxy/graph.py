# This is the search space for Zero Cost Proxies

from ..base import SearchSpaceBase

from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.utils import get_train_val_loaders, get_project_root
from fvcore.common.config import CfgNode
from tqdm import tqdm

from .utils import sample_arch_dataset, evaluate_predictions, eval_zcp

DEFAULT_ZERO_COST_PROXY_CONFIG = {
    "config": {
        'benchmark': 'nas-bench-201',
        'dataset': 'cifar10',
        'how_many_archs': 10,
        'zc_proxy': 'synflow'
    }
}

class ZeroCostProxy(SearchSpaceBase):
    """
    Zero Cost Proxy search space.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_ZERO_COST_PROXY_CONFIG
        self.scores = {}
        self.zcp_results = []
    
    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """

        # Create configs required for get_train_val_loaders
        config_dict = {
            'dataset': self.config["zc"]["dataset"], # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
            'data': str(get_project_root()) + '/data', # path to naslib/data where cifar is saved
            'search': {
                'seed': self.config["zc"]["seed"], # Seed to use in the train, validation and test dataloaders
                'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
                'batch_size': 32, # batch size of the dataloaders
            }
        }
        config = CfgNode(config_dict)

        # Get the dataloaders
        train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)

        seed = self.config["zc"]["seed"]
        
        pred_dataset = self.config["zc"]["dataset"]
        pred_api = get_dataset_api(search_space=self.config["zc"]["benchmark"], dataset=self.config["zc"]["dataset"])
        train_size = self.config["zc"]["num_archs_train"]
        test_size = self.config["zc"]["num_archs_test"]
        
        train_sample, train_hashes = sample_arch_dataset(NasBench201SearchSpace(), pred_dataset, pred_api, data_size=train_size, shuffle=True, seed=seed)
        test_sample, test_hashes = sample_arch_dataset(NasBench201SearchSpace(), pred_dataset, pred_api, arch_hashes=train_hashes, data_size=test_size, shuffle=True, seed=seed+1)
        
        xtrain, ytrain, _ = train_sample
        xtest, ytest, _ = test_sample

        for zcp_name in self.config["zc"]["zc_proxies"]:
            # train and query expect different ZCP formats
            zcp_train = [{'zero_cost_scores': eval_zcp(t_arch, zcp_name, train_loader)} for t_arch in tqdm(xtrain)]
            zcp_test = [{'zero_cost_scores': eval_zcp(t_arch, zcp_name, train_loader)} for t_arch in tqdm(xtest)]            

            zcp_pred_test = [s['zero_cost_scores'][zcp_name] for s in zcp_test]
            zcp_pred_train = [s['zero_cost_scores'][zcp_name] for s in zcp_train]
            
            train_metrics = evaluate_predictions(ytrain, zcp_pred_train)
            test_metrics = evaluate_predictions(ytest, zcp_pred_test)

            results = []
            for i, t_arch in tqdm(enumerate(xtest)):
                results.append({
                    "test_hash": f'{t_arch}',
                    "test_accuracy": ytest[i],
                    "zc_metric": zcp_pred_test[i],
                })

            self.zcp_results.append({
                zcp_name: {
                    "test_spearman": test_metrics['spearmanr'],
                    "train_spearman": train_metrics['spearmanr'],
                    "results": results
                }
            })

            # import pdb
            # pdb.set_trace()

        print("zcp_results: ", self.zcp_results)