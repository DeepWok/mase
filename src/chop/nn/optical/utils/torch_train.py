"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 03:15:06
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 03:15:06
"""

import csv
import os
import random
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
from scipy import interpolate
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from torchsummary import summary
except:
    print("[W] Cannot import torchsummary")
from .general import ensure_dir

__all__ = [
    "DeterministicCtx",
    "set_torch_deterministic",
    "set_torch_stochastic",
    "get_random_state",
    "summary_model",
    "save_model",
    "BestKModelSaver",
    "load_model",
    "count_parameters",
    "check_converge",
    "ThresholdScheduler",
    "ThresholdScheduler_tf",
    "ValueRegister",
    "ValueTracer",
    "EMA",
    "SWA",
    "export_traces_to_csv",
    "set_learning_rate",
    "get_learning_rate",
    "apply_weight_decay",
    "disable_bn",
    "enable_bn",
]


class DeterministicCtx:
    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def __enter__(self):
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        self.torch_cuda_random_state = torch.cuda.get_rng_state()
        set_torch_deterministic(self.random_state)
        return self

    def __exit__(self, *args):
        random.setstate(self.random_state)
        np.random.seed(self.numpy_random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        torch.cuda.set_rng_state(self.torch_cuda_random_state)


def set_torch_deterministic(random_state: int = 0) -> None:
    random_state = int(random_state) % (2**32)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)


def set_torch_stochastic():
    seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed_all(seed)


def get_random_state():
    return np.random.get_state()[1][0]


def summary_model(model, input):
    summary(model, input)


def save_model(model, path="./checkpoint/model.pt", print_msg=True):
    """Save PyTorch model in path

    Args:
        model (PyTorch model): PyTorch model
        path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
        print_msg (bool, optional): Control of message print. Defaults to True.
    """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    try:
        torch.save(model.state_dict(), path)
        if print_msg:
            print(f"[I] Model saved to {path}")
    except Exception as e:
        if print_msg:
            print(f"[E] Model failed to be saved to {path}")
        traceback.print_exc(e)


class BestKModelSaver(object):
    def __init__(
        self,
        k: int = 1,
        descend: bool = True,
        truncate: int = 2,
        metric_name: str = "acc",
        format: str = "{:.2f}",
    ):
        super().__init__()
        self.k = k
        self.descend = descend
        self.truncate = truncate
        self.metric_name = metric_name
        self.format = format
        self.epsilon = 0.1**truncate
        self.model_cache = OrderedDict()

    def better_op(self, a, b):
        if self.descend:
            return a >= b + self.epsilon
        else:
            return a <= b - self.epsilon

    def __insert_model_record(self, metric, dir, checkpoint_name, epoch=None):
        metric = round(metric * 10**self.truncate) / 10**self.truncate
        if len(self.model_cache) < self.k:
            new_checkpoint_name = (
                f"{checkpoint_name}_{self.metric_name}-"
                + self.format.format(metric)
                + f"{'' if epoch is None else '_epoch-'+str(epoch)}"
            )
            path = os.path.join(dir, new_checkpoint_name + ".pt")
            self.model_cache[path] = (metric, epoch)
            return path, None
        else:
            worst_metric, worst_epoch = sorted(
                list(self.model_cache.values()),
                key=lambda x: x[0],
                reverse=False if self.descend else True,
            )[0]
            if self.better_op(metric, worst_metric):
                del_checkpoint_name = (
                    f"{checkpoint_name}_{self.metric_name}-"
                    + self.format.format(worst_metric)
                    + f"{'' if epoch is None else '_epoch-'+str(worst_epoch)}"
                )
                del_path = os.path.join(dir, del_checkpoint_name + ".pt")
                try:
                    del self.model_cache[del_path]
                except:
                    print(
                        "[W] Cannot remove checkpoint: {} from cache".format(del_path),
                        flush=True,
                    )
                new_checkpoint_name = (
                    f"{checkpoint_name}_{self.metric_name}-"
                    + self.format.format(metric)
                    + f"{'' if epoch is None else '_epoch-'+str(epoch)}"
                )
                path = os.path.join(dir, new_checkpoint_name + ".pt")
                self.model_cache[path] = (metric, epoch)
                return path, del_path
            # elif(acc == min_acc):
            #     new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
            #     path = os.path.join(dir, new_checkpoint_name+".pt")
            #     self.model_cache[path] = (acc, epoch)
            #     return path, None
            else:
                return None, None

    def get_topk_model_path(self, topk: int = 1):
        if topk <= 0:
            return []
        if topk > len(self.model_cache):
            topk = len(self.model_cache)
        return [
            i[0]
            for i in sorted(
                self.model_cache.items(), key=lambda x: x[1][0], reverse=self.descend
            )[:topk]
        ]

    def save_model(
        self,
        model,
        metric,
        epoch=None,
        path="./checkpoint/model.pt",
        other_params=None,
        save_model=False,
        print_msg=True,
    ):
        """Save PyTorch model in path

        Args:
            model (PyTorch model): PyTorch model
            acc (scalar): accuracy
            epoch (scalar, optional): epoch. Defaults to None
            path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
            other_params (dict, optional): Other saved params. Defaults to None
            save_model (bool, optional): whether save source code of nn.Module. Defaults to False
            print_msg (bool, optional): Control of message print. Defaults to True.
        """
        dir = os.path.dirname(path)
        ensure_dir(dir)
        checkpoint_name = os.path.splitext(os.path.basename(path))[0]
        if isinstance(metric, torch.Tensor):
            metric = metric.data.item()
        new_path, del_path = self.__insert_model_record(
            metric, dir, checkpoint_name, epoch
        )

        if del_path is not None:
            try:
                os.remove(del_path)
                print(f"[I] Model {del_path} is removed", flush=True)
            except Exception as e:
                if print_msg:
                    print(f"[E] Model {del_path} failed to be removed", flush=True)
                traceback.print_exc(e)

        if new_path is None:
            if print_msg:
                if self.descend:
                    best_list = list(reversed(sorted(list(self.model_cache.values()))))
                else:
                    best_list = list(sorted(list(self.model_cache.values())))
                print(
                    f"[I] Not best {self.k}: {best_list}, skip this model ("
                    + self.format.format(metric)
                    + f"): {path}",
                    flush=True,
                )
        else:
            try:
                # torch.save(model.state_dict(), new_path)
                if other_params is not None:
                    saved_dict = other_params
                else:
                    saved_dict = {}
                if save_model:
                    saved_dict.update(
                        {"model": model, "state_dict": model.state_dict()}
                    )
                    torch.save(saved_dict, new_path)
                else:
                    saved_dict.update({"model": None, "state_dict": model.state_dict()})
                    torch.save(saved_dict, new_path)
                if print_msg:
                    if self.descend:
                        best_list = list(
                            reversed(sorted(list(self.model_cache.values())))
                        )
                    else:
                        best_list = list(sorted(list(self.model_cache.values())))

                    print(
                        f"[I] Model saved to {new_path}. Current best {self.k}: {best_list}",
                        flush=True,
                    )
            except Exception as e:
                if print_msg:
                    print(f"[E] Model failed to be saved to {new_path}", flush=True)
                traceback.print_exc(e)
        return new_path


def load_model(
    model,
    path="./checkpoint/model.pt",
    ignore_size_mismatch: bool = False,
    print_msg=True,
):
    """Load PyTorch model in path

    Args:
        model (PyTorch model): PyTorch model
        path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
        ignore_size_mismatch (bool, optional): Whether ignore tensor size mismatch. Defaults to False.
        print_msg (bool, optional): Control of message print. Defaults to True.
    """
    try:
        raw_data = torch.load(path, map_location=lambda storage, location: storage)
        if isinstance(raw_data, OrderedDict) and "state_dict" not in raw_data:
            ### state_dict: OrderedDict
            state_dict = raw_data
        else:
            ### {"state_dict": ..., "model": ...}
            state_dict = raw_data["state_dict"]
        load_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        common_dict = load_keys & model_keys
        diff_dict = load_keys ^ model_keys
        extra_keys = load_keys - model_keys
        lack_keys = model_keys - load_keys
        cur_state_dict = model.state_dict()
        if ignore_size_mismatch:
            size_mismatch_dict = set(
                key
                for key in common_dict
                if model.state_dict()[key].size() != state_dict[key].size()
            )
            print(
                f"[W] {size_mismatch_dict} are ignored due to size mismatch", flush=True
            )
            common_dict = common_dict - size_mismatch_dict

        cur_state_dict.update({key: state_dict[key] for key in common_dict})
        if len(diff_dict) > 0:
            print(
                f"[W] Warning! Model is not the same as the checkpoint. not found keys {lack_keys}. extra unused keys {extra_keys}"
            )

        model.load_state_dict(cur_state_dict)
        if print_msg:
            print(f"[I] Model loaded from {path}")
    except Exception as e:
        traceback.print_exc(e)
        if print_msg:
            print(f"[E] Model failed to be loaded from {path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_converge(trace, epsilon=0.002):
    if len(trace) <= 1:
        return False
    if np.abs(trace[-1] - trace[-2]) / (np.abs(trace[-1]) + 1e-8) < epsilon:
        return True
    return False


class ThresholdScheduler(object):
    """Intepolation between begin point and end point. step must be within two endpoints"""

    def __init__(self, step_beg, step_end, thres_beg, thres_end, mode="tanh"):
        assert mode in {
            "linear",
            "tanh",
        }, "Threshold scheduler only supports linear and tanh modes"
        self.mode = mode
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        self.func = self.createFunc()

    def normalize(self, step, factor=2):
        return (step - self.step_beg) / (self.step_end - self.step_beg) * factor

    def createFunc(self):
        if self.mode == "linear":
            return lambda x: (self.thres_end - self.thres_beg) * x + self.thres_beg
        elif self.mode == "tanh":
            x = self.normalize(
                np.arange(self.step_beg, self.step_end + 1).astype(np.float32)
            )
            y = np.tanh(x) * (self.thres_end - self.thres_beg) + self.thres_beg
            return interpolate.interp1d(x, y)

    def __call__(self, x):
        return self.func(self.normalize(x)).tolist()


class ThresholdScheduler_tf(object):
    """smooth increasing threshold with tensorflow model pruning scheduler"""

    def __init__(self, step_beg, step_end, thres_beg, thres_end):
        import tensorflow as tf
        import tensorflow_model_optimization as tfmot

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        if thres_beg < thres_end:
            self.thres_min = thres_beg
            self.thres_range = thres_end - thres_beg
            self.descend = False

        else:
            self.thres_min = thres_end
            self.thres_range = thres_beg - thres_end
            self.descend = True

        self.pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0,
            final_sparsity=0.9999999,
            begin_step=self.step_beg,
            end_step=self.step_end,
        )

    def __call__(self, x):
        if x < self.step_beg:
            return self.thres_beg
        elif x > self.step_end:
            return self.thres_end
        res_norm = self.pruning_schedule(x)[1].numpy()
        if self.descend == False:
            res = res_norm * self.thres_range + self.thres_beg
        else:
            res = self.thres_beg - res_norm * self.thres_range

        if np.abs(res - self.thres_end) <= 1e-6:
            res = self.thres_end
        return res


class ValueRegister(object):
    def __init__(self, operator, name="", show=True):
        self.op = operator
        self.cache = None
        self.show = show
        self.name = name if len(name) > 0 else "value"

    def register_value(self, x):
        self.cache = self.op(x, self.cache) if self.cache is not None else x
        if self.show:
            print(f"Recorded {self.name} is {self.cache}")


class ValueTracer(object):
    def __init__(self, show=True):
        self.cache = {}
        self.show = show

    def add_value(self, name, value, step):
        if name not in self.cache:
            self.cache[name] = {}
        self.cache[name][step] = value
        if self.show:
            print(f"Recorded {name}: step = {step}, value = {value}")

    def get_trace_by_name(self, name):
        return self.cache.get(name, {})

    def get_all_traces(self):
        return self.cache

    def __len__(self):
        return len(self.cache)

    def get_num_trace(self):
        return len(self.cache)

    def get_len_trace_by_name(self, name):
        return len(self.cache.get(name, {}))

    def dump_trace_to_file(self, name, file):
        if name not in self.cache:
            print(f"[W] Trace name '{name}' not found in tracer")
            return
        torch.save(self.cache[name], file)
        print(f"[I] Trace {name} saved to {file}")

    def dump_all_traces_to_file(self, file):
        torch.save(self.cache, file)
        print(f"[I] All traces saved to {file}")

    def load_all_traces_from_file(self, file):
        self.cache = torch.load(file)
        return self.cache


class EMA(object):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone().data

    def __call__(self, name, x, mask=None):
        if name not in self.shadow:
            self.register(name, x)
            return x.data

        old_average = self.shadow[name]
        new_average = (1 - self.mu) * x + self.mu * old_average
        if mask is not None:
            new_average[mask].copy_(old_average[mask])
        self.shadow[name] = new_average.clone()
        return new_average.data


class SWA(torch.nn.Module):
    """Stochastic Weight Averging.

    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407

    # Arguments
        start_epoch:   integer, epoch when swa should start.
        lr_schedule:   string, type of learning rate schedule.
        swa_lr:        float, learning rate for swa.
        swa_lr2:       float, upper bound of cyclic learning rate.
        swa_freq:      integer, length of learning rate cycle.
        batch_size     integer, batch size (for batch norm with generator)
        verbose:       integer, verbosity mode, 0 or 1.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        start_epoch: int,
        epochs: int,  # total epochs
        steps,  # total steps per epoch
        lr_schedule="manual",
        swa_lr="auto",
        swa_lr2="auto",
        swa_freq=1,
        batch_size=None,
        verbose=0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.start_epoch = start_epoch - 1
        self.epochs = epochs
        self.steps = steps
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr

        # if no user determined upper bound, make one based off of the lower bound
        self.swa_lr2 = swa_lr2 if swa_lr2 is not None else 10 * swa_lr
        self.swa_freq = swa_freq
        self.batch_size = batch_size
        self.verbose = verbose

        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')

        schedules = ["manual", "constant", "cyclic"]

        if self.lr_schedule not in schedules:
            raise ValueError(
                '"{}" is not a valid learning rate schedule'.format(self.lr_schedule)
            )

        if self.lr_schedule == "cyclic" and self.swa_freq < 2:
            raise ValueError('"swa_freq" must be higher than 1 for cyclic schedule.')

        if self.swa_lr == "auto" and self.swa_lr2 != "auto":
            raise ValueError(
                '"swa_lr2" cannot be manually set if "swa_lr" is automatic.'
            )

        if (
            self.lr_schedule == "cyclic"
            and self.swa_lr != "auto"
            and self.swa_lr2 != "auto"
            and self.swa_lr > self.swa_lr2
        ):
            raise ValueError('"swa_lr" must be lower than "swa_lr2".')

    def on_train_begin(self):
        self.lr_record = []

        if self.start_epoch >= self.epochs - 1:
            raise ValueError('"swa_start" attribute must be lower than "epochs".')

        self.init_lr = self.optimizer.param_groups[0]["lr"]

        # automatic swa_lr
        if self.swa_lr == "auto":
            self.swa_lr = 0.1 * self.init_lr

        if self.init_lr < self.swa_lr:
            raise ValueError('"swa_lr" must be lower than rate set in optimizer.')

        # automatic swa_lr2 between initial lr and swa_lr
        if self.lr_schedule == "cyclic" and self.swa_lr2 == "auto":
            self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr) * 0.25

        self._check_batch_norm()

        if self.has_batch_norm and self.batch_size is None:
            raise ValueError(
                '"batch_size" needs to be set for models with batch normalization layers.'
            )

    def on_epoch_begin(self, epoch):
        # input epoch is from 0 to epochs-1

        self.current_epoch = epoch
        self._scheduler(epoch)

        # constant schedule is updated epoch-wise
        if self.lr_schedule == "constant":
            self._update_lr(epoch)

        if self.is_swa_start_epoch:
            # self.swa_weights = self.model.get_weights()
            self.swa_weights = {
                name: p.data.clone() for name, p in self.model.named_parameters()
            }

            if self.verbose > 0:
                print(
                    "\nEpoch %05d: starting stochastic weight averaging" % (epoch + 1)
                )

        if self.is_batch_norm_epoch:
            self._set_swa_weights(epoch)

            if self.verbose > 0:
                print(
                    "\nEpoch %05d: reinitializing batch normalization layers"
                    % (epoch + 1)
                )

            self._reset_batch_norm()

            if self.verbose > 0:
                print(
                    "\nEpoch %05d: running forward pass to adjust batch normalization"
                    % (epoch + 1)
                )

    def on_batch_begin(self, batch):
        # update lr each batch for cyclic lr schedule
        if self.lr_schedule == "cyclic":
            self._update_lr(self.current_epoch, batch)

        if self.is_batch_norm_epoch:
            batch_size = self.batch_size
            # this is for tensorflow momentum, applied to the running stat
            # momentum = batch_size / (batch * batch_size + batch_size)

            # we need to convert it to torch momentum, applied to the batch stat
            momentum = 1 - batch_size / (batch * batch_size + batch_size)

            for layer in self.batch_norm_layers:
                layer.momentum = momentum

    def on_epoch_end(self, epoch):
        if self.is_swa_start_epoch:
            self.swa_start_epoch = epoch

        if self.is_swa_epoch and not self.is_batch_norm_epoch:
            self.swa_weights = self._average_weights(epoch)

    def on_train_end(self):
        if not self.has_batch_norm:
            self._set_swa_weights(self.epochs)
        else:
            self._restore_batch_norm()

        ## TODO: what is meaning here?
        # for batch_lr in self.lr_record:
        #     self.model.history.history.setdefault("lr", []).append(batch_lr)

    def _scheduler(self, epoch):
        swa_epoch = epoch - self.start_epoch

        self.is_swa_epoch = epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
        self.is_swa_start_epoch = epoch == self.start_epoch
        self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm

    def _average_weights(self, epoch):
        # return [
        #     (swa_w * ((epoch - self.start_epoch) / self.swa_freq) + w)
        #     / ((epoch - self.start_epoch) / self.swa_freq + 1)
        #     for swa_w, w in zip(self.swa_weights, self.model.get_weights())
        # ]
        out = {}
        with torch.no_grad():
            for name, w in self.model.named_parameters():
                swa_w = self.swa_weights[name]
                out[name] = (
                    swa_w * ((epoch - self.start_epoch) / self.swa_freq) + w.data
                ) / ((epoch - self.start_epoch) / self.swa_freq + 1)
        return out

    def _update_lr(self, epoch, batch=None):
        if self.is_batch_norm_epoch:
            lr = 0
            # K.set_value(self.model.optimizer.lr, lr)
            set_learning_rate(lr, self.optimizer)
        elif self.lr_schedule == "constant":
            lr = self._constant_schedule(epoch)
            # K.set_value(self.model.optimizer.lr, lr)
            set_learning_rate(lr, self.optimizer)
        elif self.lr_schedule == "cyclic":
            lr = self._cyclic_schedule(epoch, batch)
            # K.set_value(self.model.optimizer.lr, lr)
            set_learning_rate(lr, self.optimizer)
        self.lr_record.append(lr)

    def _constant_schedule(self, epoch):
        t = epoch / self.start_epoch
        lr_ratio = self.swa_lr / self.init_lr
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.init_lr * factor

    def _cyclic_schedule(self, epoch, batch):
        """Designed after Section 3.1 of Averaging Weights Leads to
        Wider Optima and Better Generalization(https://arxiv.org/abs/1803.05407)
        """
        # steps are mini-batches per epoch, equal to training_samples / batch_size
        steps = self.steps

        swa_epoch = (epoch - self.start_epoch) % self.swa_freq
        cycle_length = self.swa_freq * steps

        # batch 0 indexed, so need to add 1
        i = (swa_epoch * steps) + (batch + 1)
        if epoch >= self.start_epoch:
            t = (((i - 1) % cycle_length) + 1) / cycle_length
            return (1 - t) * self.swa_lr2 + t * self.swa_lr
        else:
            return self._constant_schedule(epoch)

    def _set_swa_weights(self, epoch):
        # self.model.set_weights(self.swa_weights)
        for name, p in self.model.named_parameters():
            p.data.copy_(self.swa_weights[name])

        if self.verbose > 0:
            print(
                "\nEpoch %05d: final model weights set to stochastic weight average"
                % (epoch + 1)
            )

    def _check_batch_norm(self):
        self.batch_norm_momentums = []
        self.batch_norm_layers = []
        self.has_batch_norm = False
        self.running_bn_epoch = False

        for layer in self.model.modules():
            if isinstance(layer, _BatchNorm):
                self.has_batch_norm = True
                self.batch_norm_momentums.append(layer.momentum)
                self.batch_norm_layers.append(layer)

        if self.verbose > 0 and self.has_batch_norm:
            print(
                "Model uses batch normalization. SWA will require last epoch "
                "to be a forward pass and will run with no learning rate"
            )

    def _reset_batch_norm(self):
        for layer in self.batch_norm_layers:
            # initialized moving mean and
            # moving var weights
            layer.reset_running_stats()

    def _restore_batch_norm(self):
        for layer, momentum in zip(self.batch_norm_layers, self.batch_norm_momentums):
            layer.momentum = momentum


def export_traces_to_csv(trace_file, csv_file, fieldnames=None):
    traces = torch.load(trace_file)

    with open(csv_file, "w", newline="") as csvfile:
        if fieldnames is None:
            fieldnames = list(traces.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        max_len = max([len(traces[field]) for field in fieldnames])

        for idx in range(max_len):
            row = {}
            for field in fieldnames:
                value = traces[field][idx] if idx < len(traces[field]) else ""
                row[field] = (
                    value.data.item() if isinstance(value, torch.Tensor) else value
                )
            writer.writerow(row)


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def apply_weight_decay(W, decay_rate, learning_rate, mask=None):
    # in mask, 1 represents fixed variables, 0 represents trainable variables
    if mask is not None:
        W[~mask] -= W[~mask] * decay_rate * learning_rate
    else:
        W -= W * decay_rate * learning_rate


def disable_bn(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(
            m,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            m.eval()


def enable_bn(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(
            m,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            m.train()
