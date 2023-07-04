# ---------------------------------------
# This script specifies the command line args.
# ---------------------------------------
import logging
import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
import random
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from pprint import pprint

import ipdb
import numpy as np
import torch

# TODO: Refactor Search
# from chop.actions import train, test, validate, search, transform
from chop.actions import test, train, transform, validate
from chop.dataset import MyDataModule, available_datasets, get_dataset, get_dataset_info
from chop.models import (
    manual_nlp_models,
    manual_vision_models,
    model_map,
    nlp_models,
    vision_models,
)

# MASE imports
from chop.tools import getLogger, load_model, post_parse_load_config
from chop.tools.get_input import get_cf_args

# TODO: This whole file needs refactoring
# from chop.models.patched_nlp_models.custom_nlp_modules import get_custom_modify_sw_kwargs


logger = getLogger("chop")


class ChopCLI:
    def _parse(self):
        parser = ArgumentParser("Chop CLI")

        # Option naming rules:
        # 1. Action: arg = "--action", dest = "to_action"
        # 2. Boolean option: arg = "--condition", dest = "is_condition"
        # 3. General option: arg = "--option", dest = "option"

        # Chop action
        parser.add_argument(
            "action",
            type=str,
            help="The action to be performed. Must be one of ['train', 'eval', 'transform', 'search']",
        )

        # Housekeeping functionalities
        parser.add_argument(
            "--github-ci",
            action="store_true",
            dest="github_ci",
            default=False,
            help="Run in GitHub CI. Default=False",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            dest="to_debug",
            default=False,
            help="Run in debug mode. Default=False",
        )
        parser.add_argument(
            "--log-level",
            dest="log_level",
            default="info",
            choices=["debug", "info", "warning", "error", "critical"],
            help="The logging level. Default='info'. Note that this option is only effective when --debug is disabled.",
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            dest="is_interactive",
            default=False,
            help="Run in interactive mode. Default=False",
        )
        parser.add_argument(
            "--ls",
            dest="ls_target",
            default=None,
            help=(
                "List available models or datasets. Must be one of ['model', 'dataset', 'all']. "
                "Default=None will not list anything."
            ),
        )
        parser.add_argument(
            "--load",
            "--load-name",
            dest="load_name",
            default=None,
            help="The path to load the input model.",
        )
        parser.add_argument(
            "--load-type",
            dest="load_type",
            default="mz",
            choices=["pt", "pl", "mz", "hf"],
            help=(
                "The checkpoint type to be loaded. "
                "If --load is not specified, --load-type is a don't-care, "
                "Else if --load is specified, --load-type must be one of 'pt', 'pl', 'mz', 'hf', "
                "respectively representing pytorch model state dict, "
                "pytorch lightning checkpoint ,"
                "fx.GraphModule saved by Mase, "
                "and HuggingFace's checkpoint directory saved by 'save_pretrained'. Default='mz'"
            ),
        )
        parser.add_argument(
            "--project-dir",
            dest="project_dir",
            default=None,
            help="The directory to save the project. Default='${mase-tools}/mase_output'",
        )
        parser.add_argument(
            "--project",
            dest="project",
            default=None,
            help="The name of the project. Default='${mase-tools}/mase_output/${args.model}@${timestamp}'",
        )

        # args for actions
        ## Training
        parser.add_argument(
            "--model",
            dest="model",
            default=None,
            help="The name of the existing model for training.",
        )
        parser.add_argument(
            "--dataset",
            dest="dataset",
            default=None,
            help="The name of the existing dataset for training.",
        )
        parser.add_argument(
            "--training-optimizer",
            dest="training_optimizer",
            default="adam",
            choices=["adam", "sgd", "adamw"],  # TODO: add adafactor which saves gpu mem
            help="The name of the existing training optimizer for training. Default=Adam",
        )
        parser.add_argument(
            "--trainer-precision",
            dest="trainer_precision",
            default="32",
            choices=["16", "32", "64", "bf16"],
            help="Trainer precision. Default='32'",
        )
        parser.add_argument(
            "--seed",
            dest="seed",
            default=0,
            type=int,
            help="The number of steps for model optimisation. Default=0",
        )
        parser.add_argument(
            "--learning-rate",
            dest="learning_rate",
            default=1e-5,
            type=float,
            help="The initial learning rate for training. Default=1e-5",
        )
        parser.add_argument(
            "--max-epochs",
            dest="max_epochs",
            default=20,
            type=int,
            help="The maximum number of epochs for training. Default=100",
        )
        parser.add_argument(
            "--max-steps",
            dest="max_steps",
            default=-1,
            type=int,
            help="The maximum number of steps for training. Default=-1 disable this option",
        )
        parser.add_argument(
            "--batch-size",
            dest="batch_size",
            default=128,
            type=int,
            help="The batch size for training and evaluation. Default=128",
        )
        parser.add_argument(
            "--accumulate-grad-batches",
            dest="accumulate_grad_batches",
            default=1,
            type=int,
            help="The number of batches to accumulate gradients. Default=1",
        )

        ## Language model related
        parser.add_argument(
            "--pretrained",
            action="store_true",
            dest="is_pretrained",
            default=False,
            help="load pretrained checkpoint from HuggingFace/Torchvision when initialising models. Default=False",
        )
        parser.add_argument(
            "--task",
            dest="task",
            default="classification",
            choices=[
                "classification",
                "cls",
                "translation",
                "tran",
                "language_modeling",
                "lm",
            ],
            help="The task to perform. Default=classification",
        )
        parser.add_argument(
            "--max-token-len",
            dest="max_token_len",
            default=512,
            type=int,
            help="The maximum number of tokens. Default=None will use tokenizer.model_max_length",
        )

        ## CPU/GPU setup
        parser.add_argument(
            "--cpu",
            dest="num_workers",
            default=0,
            type=int,
            help="The number of CPU workers. Default=1",
        )
        parser.add_argument(
            "--gpu",
            dest="num_devices",
            default=1,
            type=int,
            help="The number of GPU devices. Default=1",
        )
        parser.add_argument(
            "--nodes",
            dest="num_nodes",
            default=1,
            type=int,
            help="The number of nodes. Default=1",
        )
        parser.add_argument(
            "--accelerator",
            dest="accelerator",
            default="auto",
            help="The accelerator type for training.",
        )
        parser.add_argument(
            "--strategy",
            dest="strategy",
            default="ddp",
            choices=[
                "ddp",
            ],
            help="The strategy type. Default=ddp",
        )
        parser.add_argument(
            "--auto-requeue",
            dest="is_to_auto_requeue",
            default=False,
            action="store_true",
            help="Whether automatic job resubmission is enabled or not on SLURM managed cluster",
        )

        ## FPGA setup for hardware generation
        parser.add_argument(
            "--target",
            dest="target",
            default="xcu250-figd2104-2L-e",
            help="The target FPGA for hardware synthesis. Default=xcu250-figd2104-2L-e",
        )
        parser.add_argument(
            "--num-targets",
            dest="num_targets",
            default=100,
            type=int,
            help="The number of FPGA devices. Default=1",
        )

        ## Config from toml
        parser.add_argument(
            "--config",
            dest="config",
            default=None,
            help="toml config file. Note that by default CLI args will override config file if additionally specified.",
        )
        parser.add_argument(
            "--force-config",
            dest="is_force_config",
            default=False,
            action="store_true",
            help="Force to use everything from config, not cli.",
        )

        return parser.parse_args()

    def __init__(self):
        super().__init__()
        args = self._parse()
        if args.to_debug:
            sys.excepthook = self._excepthook
            logger.setLevel(logging.DEBUG)
            logger.debug("Enabled debug mode.")
        else:
            match args.log_level:
                case "debug":
                    logger.setLevel(logging.DEBUG)
                case "info":
                    logger.setLevel(logging.INFO)
                case "warning":
                    logger.setLevel(logging.WARNING)
                case "error":
                    logger.setLevel(logging.ERROR)
                case "critical":
                    logger.setLevel(logging.CRITICAL)

        # Initialise seeds
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        args = post_parse_load_config(args)
        self.args = args

        # Model specifications
        self.model = None
        self.data_module = None
        self.info = None

        self.output_dir = self.output_dir_sw = self.output_dir_hw = None

    # Debugging configuration
    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb

        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        ipdb.post_mortem(etb)

    # Main process
    def run(self):
        if self.args.ls_target is not None:
            self.list_available()
            return

        self.init_model_and_dataset()
        self.create_output_dir()

        if self.args.action == "transform":
            self.transform()
        elif self.args.action == "train":
            self.train()
        elif self.args.action == "test":
            self.test()
        else:
            raise ValueError(f"{self.args.action} is not supported!")

    def list_available(self):
        args = self.args
        tgt = args.ls_target
        assert tgt in [
            "model",
            "dataset",
            "all",
        ], "The param of --ls must be one of ['model', 'dataset', 'all']"

        if tgt in ["model", "all"]:
            logger.info("Available models")
            pprint(list(model_map.keys()))
        if tgt in ["dataset", "all"]:
            logger.info("Available datasets")
            pprint(available_datasets)

    # Setup model and data for training
    def init_model_and_dataset(self):
        args = self.args
        assert args.dataset, f"Dataset name (--dataset) not specified: {args.dataset!r}"
        assert args.model, f"Model name (--model) not specified: {args.model!r}"

        logger.info(f"Initialising model {args.model!r}...")
        # Get dataset info
        dataset_info = get_dataset_info(args.dataset)

        # Get model
        model_inst_fn = model_map[args.model]

        checkpoint = None
        if args.load_name is not None and args.load_type == "hf":
            checkpoint = args.load_name

        if args.model in nlp_models:
            if args.model in manual_nlp_models:
                model_dict = model_inst_fn(
                    name=args.model,
                    task=args.task,
                    info=dataset_info,
                    checkpoint=checkpoint,
                    pretrained=args.is_pretrained,
                    config=args.custom_config,
                )
            else:
                model_dict = model_inst_fn(
                    name=args.model,
                    task=args.task,
                    info=dataset_info,
                    checkpoint=checkpoint,
                    pretrained=args.is_pretrained,
                )
        elif args.model in vision_models:
            if args.model in manual_vision_models:
                # create manual model from custom config
                model_dict = model_inst_fn(info=dataset_info, config=args.custom_config)
            else:
                model_dict = model_inst_fn(
                    info=dataset_info, pretrained=args.is_pretrained
                )
        else:
            raise NotImplementedError(f"Unknown model {args.model!r}.")

        # Get data module
        logger.info(f"Initialising dataset {args.dataset!r}...")
        data_module = MyDataModule(
            model_name=args.model,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            workers=args.num_workers,
            tokenizer=model_dict["tokenizer"] if args.model in nlp_models else None,
            max_token_len=args.max_token_len,
        )
        self.model, self.data_module, self.info = (
            model_dict,
            data_module,
            dataset_info,
        )

    def create_output_dir(self):
        args = self.args

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        project_dir = os.path.join(root, "mase_output")
        if args.project_dir is not None:
            project_dir = (
                args.project_dir
                if os.path.isabs(args.project_dir)
                else os.path.join(os.getcwd(), args.project_dir)
            )
        project = (
            args.project
            if args.project is not None
            else "{}_{}_{}_".format(
                args.model.replace("/", "-"), args.task, args.dataset
            )
            + time.strftime("%Y-%m-%d")
        )

        self.output_dir = os.path.join(project_dir, project)
        self.output_dir_sw = os.path.join(self.output_dir, "software")
        self.output_dir_hw = os.path.join(self.output_dir, "hardware")
        os.makedirs(self.output_dir_sw, exist_ok=True)
        os.makedirs(self.output_dir_hw, exist_ok=True)
        if args.project_dir is None or args.project is None:
            logger.warning(f"Project will be created at {self.output_dir}")
        else:
            logger.info(f"Project will be created at {self.output_dir}")

    def transform(self):
        args = self.args
        assert (
            args.config is not None
        ), "--config must be provided if using action=transform"
        logger.info(f"Transforming model {args.model!r}...")

        self.data_module.prepare_data()
        self.data_module.setup()
        transform_params = {
            "model_name": args.model,
            "model": self.model["model"]
            if isinstance(self.model, dict)
            else self.model,
            "is_nlp_model": args.model in nlp_models,
            "task": args.task,
            "data_module": self.data_module,
            "config": args.config,
            "save_dir": os.path.join(self.output_dir_sw, "transformed_ckpts"),
            "load_name": args.load_name,
            "load_type": args.load_type,
        }
        transform(**transform_params)

    def train(self):
        args = self.args
        logger.info(f"Training model {args.model!r}...")

        plt_trainer_args = {
            "max_epochs": args.max_epochs,
            "max_steps": args.max_steps,
            "devices": args.num_devices,
            "num_nodes": args.num_nodes,
            "accelerator": args.accelerator,
            "strategy": args.strategy,
            "fast_dev_run": args.to_debug,
            "precision": args.trainer_precision,
            "accumulate_grad_batches": args.accumulate_grad_batches,
        }

        load_name = None
        if args.load_name is not None and args.load_type in ["pt", "pl", "mz"]:
            load_name = args.load_name
        train_params = {
            "model_name": args.model,
            "model": self.model,
            "info": self.info,
            "task": args.task,
            "data_module": self.data_module,
            "optimizer": args.training_optimizer,
            "learning_rate": args.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "auto_requeue": args.is_to_auto_requeue,
            "save_path": os.path.join(self.output_dir_sw, "training_ckpts"),
            "load_name": load_name,
            "load_type": args.load_type,
        }
        train(**train_params)
        logger.info("Training is completed")

    def test(self):
        args = self.args
        logger.info(f"Testing model {args.model!r}...")

        plt_trainer_args = {
            "devices": args.num_devices,
            "num_nodes": args.num_nodes,
            "accelerator": args.accelerator,
            "strategy": args.strategy,
            "precision": args.trainer_precision,
        }
        load_name = args.load_name if self.when_to_load == "train_val_or_test" else None
        # assert load_name is not None, "load name must not be None for test-sw."
        test_params = {
            "model_name": args.model,
            "model": self.model,
            "info": self.info,
            "task": args.task,
            "data_module": self.data_module,
            "optimizer": args.training_optimizer,
            "learning_rate": args.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "auto_requeue": args.is_to_auto_requeue,
            "save_path": os.path.join(self.output_dir_sw, "checkpoints"),
            "load_name": load_name,
            "load_type": self.how_to_load,
        }
        test(**test_params)

        logger.info("Testing is completed")
