from scipy.stats import kendalltau, spearmanr
import random
from tqdm import tqdm
from naslib.search_spaces.core.query_metrics import Metric
from naslib.predictors import ZeroCost
from naslib.search_spaces import NasBench201SearchSpace


def evaluate_predictions(y_true, y_pred):
    res = {}
    res["kendalltau"] = kendalltau(y_true, y_pred)[0]
    res["spearmanr"] = spearmanr(y_true, y_pred)[0]

    return res


def iterate_whole_searchspace(search_space, dataset_api, seed=None, shuffle=False):
    arch_iter = search_space.get_arch_iterator(dataset_api)
    if shuffle:
        arch_iter = list(arch_iter)
        rng = random if seed is None else random.Random(seed)
        rng.shuffle(arch_iter)

    for arch_str in arch_iter:
        yield arch_str


def sample_arch_dataset(
    search_space,
    dataset,
    dataset_api,
    data_size=None,
    arch_hashes=None,
    seed=None,
    shuffle=False,
):
    xdata = []
    ydata = []
    train_times = []
    arch_hashes = arch_hashes if arch_hashes is not None else set()

    # get all architecture hashes and accuracies in a searchspace.
    search_space = search_space.clone()
    search_space.instantiate_model = False
    arch_iterator = iterate_whole_searchspace(
        search_space, dataset_api, shuffle=shuffle, seed=seed
    )

    # iterate over architecture hashes
    for arch in tqdm(arch_iterator):
        if data_size is not None and len(xdata) >= data_size:
            break

        if arch in arch_hashes:
            continue

        arch_hashes.add(arch)
        search_space.set_spec(arch)

        # query metric for the current architecture hash
        accuracy = search_space.query(
            metric=Metric.TEST_ACCURACY, dataset=dataset, dataset_api=dataset_api
        )
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api
        )

        xdata.append(arch)
        ydata.append(accuracy)
        train_times.append(train_time)

    return [xdata, ydata, train_times], arch_hashes


def encode_archs(search_space, arch_ops, encoding=None, verbose=True):
    encoded = []

    for arch_str in tqdm(arch_ops, disable=not verbose):
        arch = search_space.clone()
        arch.set_spec(arch_str)

        arch = arch.encode(encoding) if encoding is not None else arch
        encoded.append(arch)

    return encoded


def eval_zcp(model, zc_name, data_loader):
    model = encode_archs(NasBench201SearchSpace(), [model], verbose=False)[0]
    model.parse()
    zc_pred = ZeroCost(method_type=zc_name)
    res = zc_pred.query(graph=model, dataloader=data_loader)

    return {zc_name: res}
