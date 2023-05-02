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

from .dataset import MyDataModule, available_datasets, get_dataset_info

# from .estimate_sw.flop_estimator import run_flop_estimator
from .estimate_sw import run_sw_estimator
from .evaluate_hw.mase_hardware_evaluator import get_synthesis_results
from .graph.dummy_inputs import get_dummy_inputs
from .graph.mase_graph import MaseGraph
from .graph.mase_tracer import mase_symbolic_trace
from .graph.passes import (
    create_and_save_common_metadata,
    fuse_conv_bn_pass,
    remove_nonsynthesizable_nodes_pass,
)
from .models import (
    manual_nlp_models,
    manual_vision_models,
    model_map,
    nlp_models,
    vision_models,
)
from .models.patched_nlp_models.custom_nlp_modules import get_custom_modify_sw_kwargs
from .modify.modifier import Modifier
from .session import search, test, train, validate
from .session.search import search
from .synthesize.mase_verilog_emitter import MaseVerilogEmitter
from .utils import (
    check_when_to_load_and_how_to_load,
    getLogger,
    load_pt_pl_or_pkl_checkpoint_into_pt_model,
)

logger = getLogger("machop", log_file="machop.log")


class Machop:
    def _parse(self):
        parser = ArgumentParser("Machop CLI")

        # Option naming rules:
        # 1. Action: arg = "--action", dest = "to_action"
        # 2. Boolean option: arg = "--condition", dest = "is_condition"
        # 3. General option: arg = "--option", dest = "option"

        # Global args
        # Jianyi 26/02/2023: It is unclear what does end-to-end process
        # mean to an arbitrary model for now. Let's keep it disabled until
        # we have more models.
        # parser.add_argument(
        #     '-a',
        #     '--run-all',
        #     action='store_true',
        #     dest='to_run_all',
        #     default=False,
        #     help='Run the whole end-to-end process, Default=False')
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
            dest="load_name",
            default=None,
            help="The path to load the input model.",
        )
        parser.add_argument(
            "--load-type",
            dest="load_type",
            default=None,
            help=(
                "The checkpoint type to be loaded. "
                "If --load is not specified, --load-type is a don't-care, "
                "Else if --load is specified, --load-type must be one of 'pt', 'pl', 'pkl', 'hf', "
                "referring to pytorch model state dict, "
                "pytorch lightning's LightningModule saved by Trainer's checkpoint callback, "
                "pkl file saved by MASE's modify-sw, "
                "and HuggingFace's checkpoint directory saved by 'save_pretrained'. Default=None"
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

        ## Intermediate model args

        # Actions for model
        parser.add_argument(
            "--train",
            action="store_true",
            dest="to_train",
            default=False,
            help="Train the model. Default=False",
        )
        parser.add_argument(
            "--test-sw",
            action="store_true",
            dest="to_test_sw",
            default=False,
            help="Test the model in software (accuracy). Default=False",
        )
        parser.add_argument(
            "--validate-sw",
            action="store_true",
            dest="to_validate_sw",
            default=False,
            help="Evaluate the model as software (performance). Default=False",
        )

        parser.add_argument(
            "--synthesize",
            dest="to_synthesize",
            default=None,
            help="Synthesize the model to hardware. Default=False",
        )
        parser.add_argument(
            "--test-hw",
            dest="to_test_hw",
            default=None,
            help="Test the hardware design of the model. Default=False",
        )
        parser.add_argument(
            "--evaluate-hw",
            dest="to_evaluate_hw",
            default=None,
            help="Evaluate the hardware design of the model. Default=False",
        )

        # parser.add_argument(
        #     "--modify-sw",
        #     dest="modify_sw",
        #     default=None,
        #     help="Modify the model in software from a configuration file",
        # )
        parser.add_argument(
            "--modify-sw",
            dest="to_modify_sw",
            action="store_true",
            default=False,
            help="Whether to modify the model in software from a configuration file.",
        )
        parser.add_argument(
            "--modify-sw-config",
            dest="modify_sw_config",
            default=None,
            help="The path to software modification config file if --modify-sw or --synthesis is specified. Default=None",
        )
        parser.add_argument(
            "--modify-hw",
            dest="modify_hw",
            default=None,
            help="Modify the model in hardware from a configuration file.",
        )

        # parser.add_argument(
        #     "--estimate-sw",
        #     action="store_true",
        #     dest="to_estimate_sw",
        #     default=False,
        #     help="Estimate the resource consumption of the model in software, such as FLOPs and memory footprints. Default=False",
        # )
        parser.add_argument(
            "--estimate-sw",
            dest="estimate_sw",
            default=None,
            choices=["stat", "flop"],
            help="Estimate the model in software, such as FLOPs, memory consumption, and statistical profile",
        )
        parser.add_argument(
            "--estimate-sw-config",
            dest="estimate_sw_config",
            default=None,
            help="The path to software estimation config file if --estimated-sw is specified. Default=None will use default estimation config.",
        )
        parser.add_argument(
            "--search-sw",
            dest="to_search_sw",
            action="store_true",
            default=False,
            help="Whether to search configurations.",
        )
        parser.add_argument(
            "--search-sw-config",
            dest="search_sw_config",
            default=None,
            help="The path to load software search config file if --search-sw is specified. Default=None",
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
            help="The name of the existing training optimizer for training. Default=Adam",
        )
        parser.add_argument(
            "--trainer-precision",
            dest="trainer_precision",
            default=32,
            type=int,
            help="PyTorchLightning Trainer precision, 16 or 32. Default=32",
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

        ## Language model related
        parser.add_argument(
            "--pretrained",
            action="store_true",
            dest="is_pretrained",
            default=False,
            help="Use pretrained model from HuggingFace. Default=False",
        )
        parser.add_argument(
            "--task",
            dest="task",
            default="classification",
            help="The task to perform. Default=classification",
        )
        parser.add_argument(
            "--max-token-len",
            dest="max_token_len",
            default=512,
            type=int,
            help="The maximum number of tokens. Default=None will use tokenizer.model_max_length",
        )

        ## CPU/GPU setup for lightning
        parser.add_argument(
            "--cpu",
            dest="num_workers",
            default=1,
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

        # Develop and test only. Should not be used by users, otherwise the
        # option must be defined as a seperate one like the ones above.
        parser.add_argument(
            "--config",
            dest="custom_config",
            default=None,
            help="Additional args for quick development and testing.",
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
            logger.setLevel(logging.INFO)

        logger.info(f"Arguments:\n{vars(args)}")

        # Initialise seeds
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args

        # Model specifications
        self.model = None
        self.modified_model = None
        self.data_module = None
        self.info = None

        self.output_dir = self.output_dir_sw = self.output_dir_hw = None
        self.when_to_load, self.how_to_load = check_when_to_load_and_how_to_load(
            to_modify_sw=args.to_modify_sw,
            # to_modify_sw=args.to_modify_sw,
            to_train=args.to_train,
            to_validate_sw=args.to_validate_sw,
            to_test_sw=args.to_test_sw,
            is_pretrained=args.is_pretrained,
            load_name=args.load_name,
            load_type=args.load_type,
        )
        logger.debug(
            f"when_to_load={self.when_to_load}, how_to_load={self.how_to_load}"
        )

    # Debugging configuration
    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb

        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        # pdb.post_mortem(etb)
        ipdb.post_mortem(etb)

    # Main process
    def run(self):
        if self.args.ls_target is not None:
            self.list_available()
            return
        assert not (
            (self.args.to_modify_sw)
            and (
                self.args.to_synthesize is not None or self.args.to_test_hw is not None
            )
        ), "--modify-sw and --synthesize/--test-hw cannot both be specified"
        self.init_model_and_dataset()
        self.create_output_dir()
        if self.args.to_modify_sw:
            self.modify_sw()
        if self.args.to_train:
            self.train()
        if self.args.to_test_sw:
            self.test_sw()
        if self.args.to_validate_sw:
            self.validate_sw()
        if self.args.estimate_sw is not None:
            self.estimate_sw()
        if self.args.to_search_sw:
            self.search_sw()
        if self.args.to_synthesize is not None or self.args.to_test_hw is not None:
            self.pre_synthesis()
        if self.args.to_synthesize is not None:
            self.synthesize()
        if self.args.to_test_hw is not None:
            self.test_hw()
        if self.args.to_evaluate_hw:
            to_evaluate = ["synth", "impl"]
            assert (
                self.args.to_evaluate_hw in to_evaluate
            ), f"Unsupported mode: {self.args.to_evaluate_hw}. Support: {to_evaluate}"
            self.evaluate_hw(self.args.to_evaluate_hw)

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
        logger.info(
            f"Loading DataModule of {args.dataset!r} for model {args.model!r}..."
        )
        assert args.dataset, f"Dataset name (--dataset) not specified: {args.dataset!r}"
        assert args.model, f"Model name (--model) not specified: {args.model!r}"

        # Get dataset info
        dataset_info = get_dataset_info(args.dataset)

        # Get model
        model_inst_fn = model_map[args.model]

        checkpoint = args.load_name if self.when_to_load == "init" else None
        if args.model in nlp_models:
            # *: here the model is a dict whose keys consist of "model", "tokenizer", "classifier"
            # *: here the load name only works for HuggingFace model load name/ config load name
            if args.model in manual_nlp_models:
                model = model_inst_fn(
                    name=args.model,
                    task=args.task,
                    info=dataset_info,
                    checkpoint=checkpoint,
                    pretrained=args.is_pretrained,
                    config=args.custom_config,
                )
            else:
                model = model_inst_fn(
                    name=args.model,
                    task=args.task,
                    info=dataset_info,
                    checkpoint=checkpoint,
                    pretrained=args.is_pretrained,
                )
        elif args.model in vision_models:
            if args.model in manual_vision_models:
                # create manual model from custom config
                model = model_inst_fn(info=dataset_info, config=args.custom_config)
            else:
                model = model_inst_fn(info=dataset_info, pretrained=args.is_pretrained)
        else:
            raise NotImplementedError(f"Unknown model {args.model!r}.")
        logger.info("Model is created")
        # Get data module
        data_module = MyDataModule(
            model_name=args.model,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            workers=args.num_workers,
            tokenizer=model["tokenizer"] if args.model in nlp_models else None,
            max_token_len=args.max_token_len,
        )
        logger.info("DataModule is created")
        # import ipdb

        # ipdb.set_trace()
        self.model, self.data_module, self.info = model, data_module, dataset_info

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
        if not os.path.isdir(self.output_dir_sw):
            os.makedirs(self.output_dir_sw)
        if not os.path.isdir(self.output_dir_hw):
            os.makedirs(self.output_dir_hw)
        if args.project_dir is None or args.project is None:
            logger.warning(f"Project will be created at {self.output_dir}")
        else:
            logger.info(f"Project will be created at {self.output_dir}")

    def modify_sw(self):
        args = self.args
        assert (
            args.modify_sw_config is not None
        ), "--modify-sw-config must be provided if --modify-sw is True"
        logger.info(f"Modifying model {args.model!r}...")

        if self.when_to_load == "modify-sw":
            if args.model in nlp_models:
                self.model["model"] = load_pt_pl_or_pkl_checkpoint_into_pt_model(
                    load_name=args.load_name,
                    load_type=args.load_type,
                    model=self.model["model"],
                )
            else:
                self.model = load_pt_pl_or_pkl_checkpoint_into_pt_model(
                    load_name=args.load_name,
                    load_type=args.load_type,
                    model=self.model,
                )

        dummy_inputs = get_dummy_inputs(
            model_name=args.model,
            task=args.task,
            model=self.model["model"] if args.model in nlp_models else self.model,
        )
        if args.model in nlp_models:
            custom_modify_sw_kwargs = get_custom_modify_sw_kwargs(
                model_name=args.model, config_path=args.modify_sw_config
            )
        else:
            custom_modify_sw_kwargs = {}
        modifier_kwargs = {
            "model": self.model["model"] if args.model in nlp_models else self.model,
            "config_path": args.modify_sw_config,
            "dummy_inputs_for_fx": dummy_inputs,
            "save_dir": os.path.join(self.output_dir_sw, "modify-sw"),
        }
        modifier_kwargs |= custom_modify_sw_kwargs
        Modifier.create_empty_config_template(
            model=self.model["model"] if args.model in nlp_models else self.model,
            dummy_inputs=dummy_inputs,
            save_path=os.path.join(
                self.output_dir_sw, "modify-sw", "modify-sw_template.toml"
            ),
        )
        m = Modifier(**modifier_kwargs)
        m.modify()
        if args.model in nlp_models:
            self.model["model"] = m.graph_module
        else:
            self.model = m.graph_module
        logger.info("Modify-sw is completed")

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
        }

        load_name = args.load_name if self.when_to_load == "train_val_or_test" else None
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
            "save_path": os.path.join(self.output_dir_sw, "checkpoints"),
            "load_name": load_name,
            "load_type": self.how_to_load,
        }
        train(**train_params)
        logger.info("Training is completed")

    def test_sw(self):
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

    def validate_sw(self):
        args = self.args
        logger.info(f"Validating model {args.model!r}...")

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
            "auto_requeue": args.is_to_auto_requeue,
            "plt_trainer_args": plt_trainer_args,
            "save_path": os.path.join(self.output_dir_sw, "checkpoints"),
            "load_name": load_name,
            "load_type": self.how_to_load,
        }
        validate(**test_params)

        logger.info("Validating is completed")

    def estimate_sw(self):
        args = self.args
        logger.info(f"Estimating model {args.model!r}...")
        save_dir = os.path.join(self.output_dir_sw, "estimate-sw")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dummy_inputs = get_dummy_inputs(
            model_name=args.model,
            task=args.task,
            model=self.model["model"] if args.model in nlp_models else self.model,
        )

        estimate_sw_kwargs = {
            "estimate_sw": args.estimate_sw,
            "model_name": args.model,
            "task": args.task,
            "info": self.info,
            "model": self.model["model"] if args.model in nlp_models else self.model,
            "data_module": self.data_module,
            "config_path": args.estimate_sw_config,
            "dummy_inputs_for_fx": dummy_inputs,
            "save_dir": save_dir,
        }
        # run_flop_estimator(**estimate_sw_kwargs)
        run_sw_estimator(**estimate_sw_kwargs)
        logger.info("Estimate-sw is completed")

    def search_sw(self):
        args = self.args
        logger.info(f"Searching model {args.model!r}...")
        save_dir = os.path.join(self.output_dir_sw, "search-sw")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        dummy_inputs = get_dummy_inputs(
            model_name=args.model,
            task=args.task,
            model=self.model["model"] if args.model in nlp_models else self.model,
        )
        modifier_kwargs = {}
        modifier_kwargs["dummy_inputs_for_fx"] = dummy_inputs
        # if args.model in nlp_models:
        #     modifier_kwargs |= get_custom_modify_sw_kwargs(
        #         model_name=args.model, config_path=args.modify_sw_config
        #     )

        # search_args = {
        #     "model_name": args.model,
        #     "info": self.info,
        #     "model": self.model,
        #     "task": args.task,
        #     "modifier_kwargs": modifier_kwargs,
        #     "data_module": self.data_module,
        #     "search_config": args.search_sw_config,
        #     "save_dir": save_dir,
        #     "accelerator": args.accelerator,
        # }
        # search(**search_args)
        search_args = {
            "model_name": args.model,
            "model": self.model,
            "is_nlp_model": args.model in nlp_models,
            "task": args.task,
            "info": self.info,
            "modifier_kwargs": modifier_kwargs,
            "data_module": self.data_module,
            "search_config": args.search_sw_config,
            "save_dir": save_dir,
            "accelerator": args.accelerator,
        }
        search(**search_args)

    # ------------------------------------------------------
    # HW actions
    # ------------------------------------------------------
    def pre_synthesis(self):
        args = self.args
        assert (
            args.modify_sw_config
        ) is not None, "--modify-sw-config must be provided for hardware passes"
        assert (
            os.path.isfile(args.modify_sw_config)
        ) is not None, "Invalid modify-sw-config. Cannot find the file."

        # In softwatre, the model might be modified by external module components.
        # These components need to be sync-ed in the hardware model. In this case,
        # a modified model is created from the original model where the original model
        # is also kept, which provides a mapping between the orignal model and modified model.
        logger.info(f"Modifying model {args.model!r} for hw-gen...")
        model = self.model["model"] if args.model in nlp_models else self.model

        # Create a graph module
        dummy_inputs = get_dummy_inputs(
            model_name=args.model, task=args.task, model=model
        )
        if args.model in nlp_models:
            custom_modify_sw_kwargs = get_custom_modify_sw_kwargs(
                model_name=args.model, config_path=args.modify_sw_config
            )
        else:
            custom_modify_sw_kwargs = {}

        model.eval()
        # graph_module = mase_symbolic_trace(model, concrete_args=dummy_inputs)

        # Preprocess graph module for both models
        if args.model in vision_models:
            graph_module = mase_symbolic_trace(model, concrete_args=dummy_inputs)
            graph_module = fuse_conv_bn_pass(graph_module)
        else:
            graph_module = model  # !: dirty but works

        # Create a modified graph module for interpreting
        model_to_modify = deepcopy(graph_module)

        modifier_kwargs = {
            "model": model_to_modify,
            "config_path": args.modify_sw_config,
            "dummy_inputs_for_fx": dummy_inputs,
            "save_dir": os.path.join(self.output_dir_sw, "modify-sw"),
        }
        modifier_kwargs |= custom_modify_sw_kwargs

        m = Modifier(**modifier_kwargs)
        modified_graph_module = m.modify()
        if args.model in nlp_models:
            self.model["model"] = graph_module
            self.modified_model = deepcopy(self.model)
            self.modified_model["model"] = modified_graph_module
        else:
            self.model = graph_module
            self.modified_model = modified_graph_module
        logger.info("Modify-sw-for-hw-gen is completed")

        logger.info(f"Updating metadata for synthesis...")
        create_and_save_common_metadata(
            modified_graph_model=modified_graph_module,
            model_name=args.model,
            task=args.task,
            data_module=self.data_module,
            save_dir=os.path.join(self.output_dir_sw, "modify-sw"),
        )
        logger.info(f"Metadata update is completed")

    def synthesize(self):
        args = self.args
        to_synthesize = ["hls", "auto"]
        assert (
            args.to_synthesize in to_synthesize
        ), f"Unsupported mode: {args.to_synthesize}. Support: {to_synthesize}"
        mode = args.to_synthesize
        logger.info(f"Generating hardware for {args.model!r}...")

        model = self.model["model"] if args.model in nlp_models else self.model
        quantized_model = (
            self.modified_model["model"]
            if args.model in nlp_models
            else self.modified_model
        )
        mve = MaseVerilogEmitter(
            github_ci=args.github_ci,
            model=model,
            quantized_model=quantized_model,
            project_dir=self.output_dir,
            to_debug=args.to_debug,
            target=args.target,
            mode=mode,
            num_targets=args.num_targets,
            args=args,
            common_param=os.path.join(
                self.output_dir,
                "software",
                "modify-sw",
                "common_meta.toml",
            ),
        )
        mve = remove_nonsynthesizable_nodes_pass(mve)
        mve.export("./graph.ir")
        # mve.optimise()
        mve.emit_verilog()

        if args.to_debug:
            mve.save_parameters(
                os.path.join(
                    args.project_dir,
                    args.project,
                    "hardware",
                    f"{args.project}_hw.toml",
                )
            )

    def test_hw(self):
        args = self.args
        to_test_hw = ["hls", "auto"]
        assert (
            args.to_test_hw in to_test_hw
        ), f"Unsupported mode: {args.to_test_hw}. Support: {to_test_hw}"
        mode = args.to_test_hw
        logger.info(f"Testing hardware for {args.model!r}...")

        model = self.model["model"] if args.model in nlp_models else self.model
        quantized_model = (
            self.modified_model["model"]
            if args.model in nlp_models
            else self.modified_model
        )

        mve = MaseVerilogEmitter(
            github_ci=args.github_ci,
            mase_graph=self.mase_graph,
            project_dir=self.output_dir,
            to_debug=args.to_debug,
            target=args.target,
            mode=mode,
            num_targets=args.num_targets,
            args=args,
            common_param=os.path.join(
                self.output_dir,
                "software",
                "modify-sw",
                # "hw_quantize.toml",
                "common_meta.toml",
            ),
        )
        mve.emit_tb()
        mve.run_cosim()

    def evaluate_hw(self, mode):
        logger.info(f"Evaluating hardware for {self.args.model!r}...")
        if mode == "synth":
            mase_graph = MaseGraph(
                model=self.model,
                args=args,
                common_param=os.path.join(
                    self.output_dir,
                    "software",
                    "modify-sw",
                    "common_meta.toml",
                ),
            )
            get_synthesis_results(
                self.args.model, mase_graph, self.args.target, self.output_dir
            )
        else:
            raise NotImplementedError(f"Place and Route script not implemented yet.")
