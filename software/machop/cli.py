# ---------------------------------------
# This script specifies the command line args.
# ---------------------------------------
import functools
import logging
import os
import random
import sys
import time
from argparse import ArgumentParser

import numpy as np
import toml
import torch

from .dataset import get_dataloader, get_dataset
from .estimate_sw.estimate_sw import estimate_sw_single_gpu
from .graph.dummy_inputs import get_dummy_inputs
from .models import manual_models, model_map, nlp_models, vision_models
from .modify.modifier import Modifier
from .session import test, train
from .utils import check_conda_env

try:
    import torch_mlir

    from .synthesize.mase_verilog_emitter import MaseVerilogEmitter

    is_sw_env = False
except ModuleNotFoundError:
    is_sw_env = True

logging.getLogger().setLevel(logging.INFO)


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
            "--load",
            dest="load_name",
            default=None,
            help="The path to load the input model.",
        )
        parser.add_argument(
            "--project-dir",
            dest="project_dir",
            default=None,
            help="The directory to save the project. Default='../mase_output'",
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
            "--evaluate-sw",
            action="store_true",
            dest="to_evaluate_sw",
            default=False,
            help="Evaluate the model as software (performance). Default=False",
        )

        parser.add_argument(
            "--synthesize",
            action="store_true",
            dest="to_synthesize",
            default=False,
            help="Synthesize the model to hardware. Default=False",
        )
        parser.add_argument(
            "--test-hw",
            action="store_true",
            dest="to_test_hw",
            default=False,
            help="Test the hardware design of the model. Default=False",
        )
        parser.add_argument(
            "--evaluate-hw",
            action="store_true",
            dest="to_evaluate_hw",
            default=False,
            help="Evaluate the hardware design of the model. Default=False",
        )
        parser.add_argument(
            "--modify-sw",
            dest="modify_sw",
            default=None,
            help="Modify the model in software from a configuration file.",
        )
        parser.add_argument(
            "--modify-hw",
            dest="modify_hw",
            default=None,
            help="Modify the model in hardware from a configuration file.",
        )

        parser.add_argument(
            "--estimate-sw",
            action="store_true",
            dest="to_estimate_sw",
            default=False,
            help="Estimate the resource consumption of the model in software, such as FLOPs and memory footprints. Default=False",
        )
        parser.add_argument(
            "--estimate-sw-config",
            dest="estimate_sw_config",
            default=None,
            help="The path to software estimation config file if --estimated-sw is specified. Default=None will use default estimation config.",
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
            "--seed",
            dest="seed",
            default=0,
            type=int,
            help="The number of steps for model optimisation. Default=0",
        )
        parser.add_argument(
            "--learning_rate",
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
            help="The maximum number of tokens. Default=512",
        )

        ## CPU/GPU setup for lightning
        parser.add_argument(
            "--cpu",
            dest="num_workers",
            default=0,
            type=int,
            help="The number of CPU workers. Default=0",
        )
        parser.add_argument(
            "--gpu",
            dest="num_devices",
            default=1,
            type=int,
            help="The number of GPU devices. Default=1",
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

        ## FPGA setup for hardware generation
        parser.add_argument(
            "--target",
            dest="target",
            default="xc7z020clg484-1",
            help="The target FPGA for hardware synthesis. Default=xc7z020clg484-1",
        )
        parser.add_argument(
            "--num-targets",
            dest="num_targets",
            default=1,
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
        print(f"Arguments:\n{vars(args)}")
        if args.to_debug:
            sys.excepthook = self._excepthook
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Enabled debug mode.")

        # Initialise seeds
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args

        # Model specifications
        self.model = None
        self.loader = None
        self.info = None

        self.output_dir = self.output_dir_sw = self.output_dir_hw = None

    # Debugging configuration
    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb

        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        import ipdb

        ipdb.post_mortem(etb)

    # Main process
    def run(self):
        self.init_model_and_dataset()
        self.create_output_dir()
        if self.args.modify_sw:
            self.modify_sw()
        if self.args.to_train:
            self.train()
        if self.args.to_test_sw:
            self.test_sw()
        if self.args.to_evaluate_sw:
            self.evaluate_sw()
        if self.args.to_estimate_sw:
            self.estimate_sw()
        if self.args.to_synthesize:
            check_conda_env(is_sw_env, False, "synthesize")
            self.synthesize()
        if self.args.to_test_hw:
            self.test_hw()
        if self.args.to_evaluate_hw:
            # TODO: this check may not be required if the profiler is based on FX
            check_conda_env(is_sw_env, True, "evaluate-sw")
            self.evaluate_hw()

    # Setup model and data for training
    def init_model_and_dataset(self):
        args = self.args
        logging.info(f"Loading dataset {args.dataset!r} for model {args.model!r}...")
        assert args.dataset, f"Dataset not specified: {args.dataset!r}"
        assert args.model, f"Model not specified: {args.model!r}"

        # Get dataset
        train_dataset, val_dataset, test_dataset, info = get_dataset(name=args.dataset)
        logging.debug(f"Dataset loaded.")
        # get model
        model_inst_fn = model_map[args.model]

        if args.model in nlp_models:
            # *: here the model is a dict whose keys consist of "model", "tokenizer", "classifier"
            # *: here the load name only works for HuggingFace model load name/ config load name
            model = model_inst_fn(
                name=args.model,
                task=args.task,
                info=info,
                checkpoint=args.load_name,
                pretrained=args.is_pretrained,
            )
        elif args.model in vision_models:
            if args.model in manual_models:
                # Jianyi 26/02/2023: need to fix this config. Is it for quantization?
                # Cheng 01/03/2023: No. This is for creating a quantized model
                #                   using a .toml file for configuring quantisation scheme
                #                   instead of layer replacement
                # here model is a nn.Module
                model = model_inst_fn(info=info, config=args.custom_config)
            else:
                model = model_inst_fn(info=info)
        else:
            raise NotImplementedError(f"Unknown model {args.model!r}.")

        # Get data loader from the datasets
        loader = get_dataloader(
            args.model,
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=args.batch_size,
            workers=args.num_workers,
            max_token_len=args.max_token_len,
        )
        self.model, self.loader, self.info = model, loader, info

    def create_output_dir(self):
        args = self.args
        project_dir = (
            args.project_dir if args.project_dir is not None else "../mase_output"
        )
        project = (
            args.project
            if args.project is not None
            else "{}@".format(args.model.replace("/", "-"))
            + time.strftime("%Y_%m_%d-%H_%M_%S")
        )

        self.output_dir = os.path.join(project_dir, project)
        self.output_dir_sw = os.path.join(self.output_dir, "software")
        self.output_dir_hw = os.path.join(self.output_dir, "hardware")
        if not os.path.isdir(self.output_dir_sw):
            os.makedirs(self.output_dir_sw)
        if not os.path.isdir(self.output_dir_hw):
            os.makedirs(self.output_dir_hw)

    def modify_sw(self):
        args = self.args
        logging.info(f"Modifying model {args.model!r}...")

        dummy_inputs = get_dummy_inputs(
            model_name=args.model,
            task=args.task,
            model=self.model["model"] if args.model in nlp_models else self.model,
            data_loader=self.loader,
        )
        # breakpoint()
        modifier_kwargs = {
            "model": self.model["model"] if args.model in nlp_models else self.model,
            "config": args.modify_sw,
            "dummy_inputs": dummy_inputs,
            "save_dir": os.path.join(self.output_dir_sw, "modify-sw"),
            "load_name": args.load_name,
        }

        m = Modifier(**modifier_kwargs)
        if args.model in nlp_models:
            self.model["model"] = m.graph_module
        else:
            self.model = m.graph_module

    def train(self):
        args = self.args
        logging.info(f"Training model {args.model!r}...")
        if args.project_dir is None:
            logging.warning(
                "--project_dir not specified. Your model will be saved in ${mase-tools}/mase_output/${args.model}@${timestamp}."
            )

        plt_trainer_args = {
            "max_epochs": args.max_epochs,
            "devices": args.num_devices,
            "accelerator": args.accelerator,
            "strategy": args.strategy,
            "fast_dev_run": args.to_debug,
        }

        train_params = {
            "model_name": args.model,
            "model": self.model,
            "info": self.info,
            "task": args.task,
            "data_loader": self.loader,
            "optimizer": args.training_optimizer,
            "learning_rate": args.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "save_path": os.path.join(self.output_dir_sw, "checkpoints"),
            "load_path": args.load_name,
        }
        train(**train_params)

    def test_sw(self):
        args = self.args
        logging.info(f"Testing model {args.model!r}...")

        plt_trainer_args = {
            "devices": args.num_devices,
            "accelerator": args.accelerator,
            "strategy": args.strategy,
        }
        test_params = {
            "model_name": args.model,
            "model": self.model,
            "info": self.info,
            "task": args.task,
            "data_loader": self.loader,
            "plt_trainer_args": plt_trainer_args,
            "load_path": args.load_name,
        }
        test(**test_params)

    evaluate_sw = test_sw

    def estimate_sw(self):
        args = self.args
        logging.info(f"Estimating model {args.model!r}...")
        estimate_sw_kwargs = {
            "model_name": args.model,
            "info": self.info,
            "model": self.model,
            "task": args.task,
            "data_loader": self.loader,
            "config_path": args.estimate_sw_config,
        }
        estimate_sw_single_gpu(**estimate_sw_kwargs)

    def synthesize(self):
        args = self.args
        logging.info(f"Generating hardware for {args.model!r}...")
        mve = MaseVerilogEmitter(
            model=self.model,
            project_path=args.project,
            project=args.model,
            to_debug=args.to_debug,
            target=args.target,
            num_targets=args.num_targets,
        )
        mve.emit_verilog()
        if args.to_debug:
            mve.save(os.path.join(args.project, args.model, f"{args.model}_hw.toml"))
        logging.warning("synthesis is only partially implemented.")

    def test_hw(self):
        args = self.args
        logging.info(f"Testing hardware for {args.model!r}...")
        # TODO: Generate cocotb testbench for a given model
        raise NotImplementedError(f"Hardware testing not implemented yet.")

    def evaluate_hw(self):
        args = self.args
        logging.info(f"Evaluating hardware for {args.model!r}...")
        # TODO: Run simulation and implementation for evaluating area and performance
        raise NotImplementedError(f"Hardware evaluation not implemented yet.")
