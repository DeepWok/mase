"""
Chop (ch): Machop's command line interface

The brains behind the chop interface that allows users to train, test and transform (i.e
prune or quantise) a supported model. You'll find a list of the available models and
datasets in Machop README.md file.

Feel free to browse this file and please flag any issues or feature requests here:
https://github.com/JianyiCheng/mase-tools/issues

NOTE: There are three types of arguments - default ones in the parser, manual overrides
via the CLI, and those specified in the configuration file. We establish precedence as
follows: default < configuration < manual overrides.

----------------------------------------------------------------------------------------
Internal Backlog:
1. Refactor search functionality
2. Add 'adafactor' as a training optimiser, which saves GPU memory
3. Push constants into a separate file
4. Add package configuration file (incl. versioning)
5. Better file validation (i.e. checking extension) and possibly schema validation for
   certain filetypes using Cerberus.
6. Move the primitive validation routines and custom actions to a separate file.
7. Merge the valid file and directory types into a common path type; implement support
   for checking extensions. The function would return a pre-configured callable.
"""

import logging
import os
import sys
import time
import argparse
from argparse import SUPPRESS
from typing import Sequence
from pathlib import Path
from functools import partial

import ipdb
import pytorch_lightning as pl
from tabulate import tabulate
import torch

import chop.models as models
from chop.actions import test, train, transform, search
from chop.dataset import MaseDataModule, AVAILABLE_DATASETS, get_dataset_info
from chop.tools import getLogger, post_parse_load_config


# Housekeeping -------------------------------------------------------------------------
# Use the third-party IPython debugger to handle breakpoints; this debugger features
# several improvements over the built-in pdb, including syntax highlighting and better
# tracebacks.
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
torch.set_float32_matmul_precision("medium")

# This file's in root/machop/chop; there's three levels of parents to get to the root.
ROOT = Path(__file__).parent.parent.parent.absolute()
VERSION = "23.07.0"

# Constants ----------------------------------------------------------------------------
LOGO = f"""
                        ,."--.
                   _.../     _""-.
                 //  ,'    ,"      :
                .'  /   .:'      __|
                || ||  /,    _."   '.
                || ||  ||  ,'        `.
               /|| ||  ||,'            .
              /.`| /` /`,'  __          '
             j /. " `"  ' ,' /`.        |
             ||.|        .  | . .       |
             ||#|        |  | #'|       '
            /'.||        |  \." |      -.
           /    '        `.----"      ".
           \  `.    ,'                .
            `._____           _,-'  '/
              `".  `'-..__..-'   _,."
                 ^-._      _,..-'
                     '""'""

    Chop (ch): Machop's Command Line Interface
                VERSION {VERSION}

          Maintained by the DeepWok Lab

     For comprehensive information on usage,
            please refer to the wiki:
   https://github.com/JianyiCheng/mase-tools/wiki
"""
TASKS = ["classification", "cls", "translation", "tran", "language_modeling", "lm"]
ACTIONS = ["train", "test", "transform", "search"]
INFO_TYPE = ["all", "model", "dataset"]
LOAD_TYPE = [
    "pt",  # PyTorch module state dictionary
    "pl",  # PyTorch Lightning checkpoint
    "mz",  # fx.GraphModule saved by MASE
    "hf",  # HuggingFace's checkpoint directory saved by 'save_pretrained'
]
OPTIMIZERS = ["adam", "sgd", "adamw"]
LOG_LEVELS = ["debug", "info", "warning", "error", "critical"]
ISSUES_URL = "https://github.com/JianyiCheng/mase-tools/issues"
STRATEGIES = [
    "ddp",
    "fsdp",
    "fsdp_native",
    "fsdp_custom",
    "deepspeed_stage_3_offload",
]
ACCELERATORS = ["auto", "cpu", "gpu"]
TRAINER_PRECISION = ["16", "32", "64", "bf16"]

# NOTE: Any new argument (either required or optional) must have their default values
# listed in this dictionary; this is critical in establishing argument precedence. :)
CLI_DEFAULTS = {
    # Main program arguments
    # NOTE: The following two are required if a configuration file isn't specified.
    "model": None,
    "dataset": None,
    # General options
    "config": None,
    "task": TASKS[0],
    "load_name": None,
    "load_type": LOAD_TYPE[2],
    "batch_size": 128,
    "to_debug": False,
    "log_level": LOG_LEVELS[1],
    "seed": 0,
    # Trainer options
    "training_optimizer": OPTIMIZERS[0],
    "trainer_precision": TRAINER_PRECISION[1],
    "learning_rate": 1e-5,
    "max_epochs": 20,
    "max_steps": -1,
    "accumulate_grad_batches": 1,
    # Runtime environment options
    "num_workers": os.cpu_count(),
    "num_devices": 1,
    "num_nodes": 1,
    "accelerator": ACCELERATORS[0],
    "strategy": STRATEGIES[0],
    "is_to_auto_requeue": False,
    "github_ci": False,
    "disable_dataset_cache": False,
    # Hardware generation options
    "target": "xcu250-figd2104-2L-e",
    "num_targets": 100,
    # Language model options
    "is_pretrained": False,
    "max_token_len": 512,
    # Project options,
    "project_dir": os.path.join(ROOT, "mase_output"),
    "project": None,
}


# Main ---------------------------------------------------------------------------------
class ChopCLI:
    def __init__(self, argv: Sequence[str] | None = None):
        super().__init__()

        self.logger = getLogger("chop")
        parser = self._setup_parser()
        args = parser.parse_intermixed_args(argv)

        # Housekeeping
        pl.seed_everything(args.seed)
        if args.to_debug:
            sys.excepthook = self._excepthook
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Enabled debug mode.")
        else:
            match args.log_level:
                case "debug":
                    self.logger.setLevel(logging.DEBUG)
                case "info":
                    self.logger.setLevel(logging.INFO)
                case "warning":
                    self.logger.setLevel(logging.WARNING)
                case "error":
                    self.logger.setLevel(logging.ERROR)
                case "critical":
                    self.logger.setLevel(logging.CRITICAL)

        # Merge arguments from the configuration file (if one exists) and print
        # NOTE: The project name is set later on (if no configuration is provided), so
        # the merged argument table may show None, but this is not the case.
        self.args = post_parse_load_config(args, CLI_DEFAULTS)

        # Sanity check
        if not self.args.model or not self.args.dataset:
            raise ValueError("No model and/or dataset provided! These are required.")

        (
            self.model,
            self.tokenizer,
            self.data_module,
            self.dataset_info,
            self.model_info,
        ) = self._setup_model_and_dataset()
        self.output_dir, self.output_dir_sw, self.output_dir_hw = self._setup_folders()

    def run(self):
        match self.args.action:
            case "transform":
                self._run_transform()
            case "train":
                self._run_train()
            case "test":
                self._run_test()
            case "search":
                self._run_search()

    # Actions --------------------------------------------------------------------------
    def _run_train(self):
        self.logger.info(f"Training model {self.args.model!r}...")

        plt_trainer_args = {
            "max_epochs": self.args.max_epochs,
            "max_steps": self.args.max_steps,
            "devices": self.args.num_devices,
            "num_nodes": self.args.num_nodes,
            "accelerator": self.args.accelerator,
            "strategy": self.args.strategy,
            "fast_dev_run": self.args.to_debug,
            "precision": self.args.trainer_precision,
            "accumulate_grad_batches": self.args.accumulate_grad_batches,
        }

        # Load from a checkpoint!
        load_name = None
        load_types = ["pt", "pl", "mz"]
        if self.args.load_name is not None and self.args.load_type in load_types:
            load_name = self.args.load_name

        train_params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "model_info": self.model_info,
            "data_module": self.data_module,
            "dataset_info": self.dataset_info,
            "task": self.args.task,
            "optimizer": self.args.training_optimizer,
            "learning_rate": self.args.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "auto_requeue": self.args.is_to_auto_requeue,
            "save_path": os.path.join(self.output_dir_sw, "training_ckpts"),
            "load_name": load_name,
            "load_type": self.args.load_type,
        }

        train(**train_params)
        self.logger.info("Training is completed")

    def _run_test(self):
        self.logger.info(f"Testing model {self.args.model!r}...")

        plt_trainer_args = {
            "devices": self.args.num_devices,
            "num_nodes": self.args.num_nodes,
            "accelerator": self.args.accelerator,
            "strategy": self.args.strategy,
            "precision": self.args.trainer_precision,
        }

        # The checkpoint must be present, except when the model is pretrained.
        if self.args.load_name is None and not self.args.is_pretrained:
            raise ValueError("expected checkpoint via --load, got None")

        test_params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "model_info": self.model_info,
            "data_module": self.data_module,
            "dataset_info": self.dataset_info,
            "task": self.args.task,
            "optimizer": self.args.training_optimizer,
            "learning_rate": self.args.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "auto_requeue": self.args.is_to_auto_requeue,
            "save_path": os.path.join(self.output_dir_sw, "checkpoints"),
            "load_name": self.args.load_name,
            "load_type": self.args.load_type,
        }

        test(**test_params)
        self.logger.info("Testing is completed")

    def _run_transform(self):
        # A configuration is compulsory for transformation passes
        if self.args.config is None:
            raise ValueError("expected configuration via --config, got None")

        self.logger.info(f"Transforming model {self.args.model!r}...")
        self.data_module.prepare_data()
        self.data_module.setup()

        # Resolve the quirk with NLP models where the actual model is embedded in a dict

        transform_params = {
            "model": self.model,
            "model_info": self.model_info,
            "data_module": self.data_module,
            "task": self.args.task,
            "config": self.args.config,
            "save_dir": os.path.join(self.output_dir_sw, "transform"),
            "load_name": self.args.load_name,
            "load_type": self.args.load_type,
        }

        transform(**transform_params)
        self.logger.info("Transformation is completed")

    def _run_search(self):
        load_name = None
        load_types = ["pt", "pl", "mz"]
        if self.args.load_name is not None and self.args.load_type in load_types:
            load_name = self.args.load_name

        search_params = {
            "model_name": self.args.model,
            "model": self.model,
            "task": self.args.task,
            "info": self.dataset_info,
            "data_module": self.data_module,
            "accelerator": self.args.accelerator,
            "search_config": self.args.config,
            "save_path": os.path.join(self.output_dir_sw, "training_ckpts"),
            "load_name": load_name,
            "load_type": self.args.load_type,
        }

        search(**search_params)
        self.logger.info("Searching is completed")

    # Helpers --------------------------------------------------------------------------
    def _setup_parser(self):
        # NOTE: For a better developer experience, it's helpful to collapse all function
        # calls by shift clicking the collapse button in the gutter on IDEs. ;) Also,
        # when creating a new argument, DO NOT use default in the function call; instead
        # add it to the CLI_DEFAULTS constant with the key set to the argument's dest.
        parser = argparse.ArgumentParser(
            description="""
                Chop is a simple utility, part of the MASE tookit, to train, test and
                transform (i.e. prune or quantise) a supported model.
            """,
            epilog=f"Maintained by the DeepWok Lab. Raise issues at {ISSUES_URL}",
            add_help=False,
        )

        # Main program arguments -------------------------------------------------------
        main_group = parser.add_argument_group("main arguments")
        main_group.add_argument(
            "action",
            choices=ACTIONS,
            help=f"action to perform. One of {'(' + '|'.join(ACTIONS) + ')'}",
            metavar="action",
        )
        main_group.add_argument(
            "model",
            nargs="?",
            default=None,
            help="name of a supported model. Required if configuration NOT provided.",
        )
        main_group.add_argument(
            "dataset",
            nargs="?",
            default=None,
            help="name of a supported dataset. Required if configuration NOT provided.",
        )

        # General options --------------------------------------------------------------
        general_group = parser.add_argument_group("general options")
        general_group.add_argument(
            "--config",
            dest="config",
            type=_valid_filepath,
            help="""
                path to a configuration file in the TOML format. Manual CLI overrides
                for arguments have a higher precedence. Required if the action is
                transform. (default: %(default)s)
            """,
            metavar="PATH",
        )
        general_group.add_argument(
            "--task",
            dest="task",
            choices=TASKS,
            help=f"""
                task to perform. One of {'(' + '|'.join(TASKS) + ')'}
                (default: %(default)s)
            """,
            metavar="TASK",
        )
        general_group.add_argument(
            "--load",
            dest="load_name",
            type=_valid_file_or_directory_path,
            help="path to load the model from. (default: %(default)s)",
            metavar="PATH",
        )
        general_group.add_argument(
            "--load-type",
            dest="load_type",
            choices=LOAD_TYPE,
            help=f"""
                the type of checkpoint to be loaded; it's disregarded if --load is NOT
                specified. It is designed to and must be used in tandem with --load.
                One of {'(' + '|'.join(LOAD_TYPE) + ')'} (default: %(default)s)
            """,
            metavar="",
        )
        general_group.add_argument(
            "--batch-size",
            dest="batch_size",
            type=int,
            help="batch size for training and evaluation. (default: %(default)s)",
            metavar="NUM",
        )
        general_group.add_argument(
            "--debug",
            action="store_true",
            dest="to_debug",
            help="""
                run the action in debug mode, which enables verbose logging, custom
                exception hook that uses ipdb, and sets the PL trainer to run in
                "fast_dev_run" mode. (default: %(default)s)
            """,
        )
        general_group.add_argument(
            "--log-level",
            dest="log_level",
            choices=LOG_LEVELS,
            help=f"""
                verbosity level of the logger; it's only effective when --debug flag is
                NOT passed in. One of {'(' + '|'.join(LOG_LEVELS) + ')'}
                (default: %(default)s)
            """,
            metavar="",
        )
        general_group.add_argument(
            "--seed",
            dest="seed",
            type=int,
            help="""
                seed for random number generators set via Pytorch Lightning's
                seed_everything function. (default: %(default)s)
            """,
            metavar="NUM",
        )

        # Trainer options --------------------------------------------------------------
        trainer_group = parser.add_argument_group("trainer options")
        trainer_group.add_argument(
            "--training-optimizer",
            dest="training_optimizer",
            choices=OPTIMIZERS,
            help=f"""
                name of supported optimiser for training. One of
                {'(' + '|'.join(OPTIMIZERS) + ')'} (default: %(default)s)
            """,
            metavar="TYPE",
        )
        trainer_group.add_argument(
            "--trainer-precision",
            dest="trainer_precision",
            choices=TRAINER_PRECISION,
            help=f"""
                numeric precision for training. One of
                {'(' + '|'.join(TRAINER_PRECISION) + ')'} (default: %(default)s)
            """,
            metavar="TYPE",
        )
        trainer_group.add_argument(
            "--learning-rate",
            dest="learning_rate",
            type=float,
            help="initial learning rate for training. (default: %(default)s)",
            metavar="NUM",
        )
        trainer_group.add_argument(
            "--max-epochs",
            dest="max_epochs",
            type=int,
            help="maximum number of epochs for training. (default: %(default)s)",
            metavar="NUM",
        )
        trainer_group.add_argument(
            "--max-steps",
            dest="max_steps",
            type=_positive_int,
            help="""
                maximum number of steps for training. A negative value disables this
                option. (default: %(default)s)
            """,
            metavar="NUM",
        )
        trainer_group.add_argument(
            "--accumulate-grad-batches",
            dest="accumulate_grad_batches",
            type=int,
            help="number of batches to accumulate gradients. (default: %(default)s)",
            metavar="NUM",
        )

        # Runtime environment options --------------------------------------------------
        runtime_group = parser.add_argument_group("runtime environment options")
        runtime_group.add_argument(
            "--cpu",
            "--num-workers",
            dest="num_workers",
            type=_positive_int,
            help="""
                number of CPU workers; the default varies across systems and is set to
                os.cpu_count(). (default: %(default)s)
            """,
            metavar="NUM",
        )
        runtime_group.add_argument(
            "--gpu",
            "--num-devices",
            dest="num_devices",
            type=_positive_int,
            help="number of GPU devices. (default: %(default)s)",
            metavar="NUM",
        )
        runtime_group.add_argument(
            "--nodes",
            dest="num_nodes",
            type=int,
            help="number of nodes. (default: %(default)s)",
            metavar="NUM",
        )
        runtime_group.add_argument(
            "--accelerator",
            dest="accelerator",
            choices=ACCELERATORS,
            help=f"""
                type of accelerator for training. One of
                {'(' + '|'.join(ACCELERATORS) + ')'} (default: %(default)s)
            """,
            metavar="TYPE",
        )
        runtime_group.add_argument(
            "--strategy",
            dest="strategy",
            choices=STRATEGIES,
            help=f"""
                type of strategy for training. One of
                {'(' + '|'.join(STRATEGIES) + ')'} (default: %(default)s)
            """,
            metavar="TYPE",
        )
        runtime_group.add_argument(
            "--auto-requeue",
            dest="is_to_auto_requeue",
            action="store_true",
            help="""
                enable automatic job resubmission on SLURM managed cluster. (default:
                %(default)s)
            """,
        )
        runtime_group.add_argument(
            "--github-ci",
            action="store_true",
            dest="github_ci",
            help="""
                set the execution environment to GitHub's CI pipeline; it's used in the
                MASE verilog emitter transform pass to skip simulations.
                (default: %(default)s)
                """,
        )
        runtime_group.add_argument(
            "--disable-dataset-cache",
            dest="disable_dataset_cache",
            action="store_true",
            help="""
                disable caching of datasets. (default: %(default)s)
            """,
        )

        # Hardware generation options --------------------------------------------------
        hardware_group = parser.add_argument_group("hardware generation options")
        hardware_group.add_argument(
            "--target",
            dest="target",
            help="target FPGA for hardware synthesis. (default: %(default)s)",
            metavar="STR",
        )
        hardware_group.add_argument(
            "--num-targets",
            dest="num_targets",
            type=int,
            help="number of FPGA devices. (default: %(default)s)",
            metavar="NUM",
        )

        # Language model options -------------------------------------------------------
        lm_group = parser.add_argument_group(title="language model options")
        lm_group.add_argument(
            "--pretrained",
            action="store_true",
            dest="is_pretrained",
            help="""
                load pretrained checkpoint from HuggingFace/Torchvision when
                initialising models. (default: %(default)s)
            """,
        )
        lm_group.add_argument(
            "--max-token-len",
            dest="max_token_len",
            type=_positive_int,
            help="""
                maximum number of tokens. A negative value will use
                tokenizer.model_max_length. (default: %(default)s)
            """,
            metavar="NUM",
        )

        # Project-level options --------------------------------------------------------
        project_group = parser.add_argument_group(title="project options")
        project_group.add_argument(
            "--project-dir",
            dest="project_dir",
            type=partial(_valid_directory_path, create_dir=True),
            help="directory to save the project to. (default: %(default)s)",
            metavar="DIR",
        )
        project_group.add_argument(
            "--project",
            dest="project",
            help="""
                name of the project.
                (default: {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP})
            """,
            metavar="NAME",
        )

        # Information flags ------------------------------------------------------------
        information_group = parser.add_argument_group("information")
        information_group.add_argument(
            "-h", "--help", action="help", help="show this help message and exit"
        )
        information_group.add_argument(
            "-V", "--version", action=ShowVersionAction, help="show version and exit"
        )
        information_group.add_argument(
            "--info",
            action=ShowInfoAction,
            const=INFO_TYPE[0],
            choices=INFO_TYPE,
            help=f"""
                list information about supported models or/and datasets and exit. One of
                {'(' + '|'.join(INFO_TYPE) + ')'} (default: %(const)s)
            """,
            metavar="TYPE",
        )

        parser.set_defaults(**CLI_DEFAULTS)
        return parser

    def _setup_model_and_dataset(self):
        self.logger.info(f"Initialising model {self.args.model!r}...")

        # Grab the dataset information and model instance functions; as evident in its
        # name, when called, the model instance function creates and returns an instance
        # of a specified model.
        # NOTE: See machop/chop/models/__init__.py for more information
        dataset_info = get_dataset_info(self.args.dataset)

        checkpoint, tokenizer = None, None
        if self.args.load_name is not None and self.args.load_type == "hf":
            checkpoint = self.args.load_name

        model_info = models.get_model_info(self.args.model)
        model = models.get_model(
            name=self.args.model,
            task=self.args.task,
            dataset_info=dataset_info,
            checkpoint=checkpoint,
            pretrained=self.args.is_pretrained,
        )

        if model_info.is_nlp_model:
            tokenizer = models.get_tokenizer(self.args.model, checkpoint=checkpoint)

        self.logger.info(f"Initialising dataset {self.args.dataset!r}...")
        data_module = MaseDataModule(
            name=self.args.dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            tokenizer=tokenizer,
            max_token_len=self.args.max_token_len,
            load_from_cache_file=not self.args.disable_dataset_cache,
            model_name=self.args.model,
        )

        return model, tokenizer, data_module, dataset_info, model_info

    def _setup_folders(self):
        project = None
        if self.args.project is not None:
            project = self.args.project
        else:
            # No project name is given; so we construct one structured as follows:
            # {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP}
            # NOTE: We set the attribute in args so that any subsequent routine has
            # access to the name of the project. :)
            project = "{}_{}_{}_{}".format(
                self.args.model.replace("/", "-"),
                self.args.task,
                self.args.dataset,
                time.strftime("%Y-%m-%d"),
            )
            setattr(self.args, "project", project)

        output_dir = Path(self.args.project_dir) / project
        output_dir_sw = Path(output_dir) / "software"
        output_dir_hw = Path(output_dir) / "hardware"
        output_dir_hw.mkdir(parents=True, exist_ok=True)
        output_dir_sw.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Project will be created at {output_dir}")

        return output_dir, output_dir_sw, output_dir_hw

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb

        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        ipdb.post_mortem(etb)


# Custom types ---------------------------------------------------------------------
# check if the path is a valid file path
def _valid_filepath(path: str):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"file not found")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"expected path to file, got {path!r}")
    return os.path.abspath(path)


# Returns the absolute path to a directory if it is indeed a valid path
def _valid_directory_path(path: str, create_dir: bool = False):
    if os.path.isfile(path):
        raise argparse.ArgumentTypeError(
            f"expected path to directory, got file {path!r}"
        )
    if (not os.path.exists(path)) and (not create_dir):
        raise argparse.ArgumentTypeError(f"directory not found")
    elif (not os.path.exists(path)) and create_dir:
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


# Returns the absolute path to a file or directory if it is indeed a valid path
def _valid_file_or_directory_path(path: str):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"file or directory not found")
    return os.path.abspath(path)


# Returns None for values less than or equal to 0
def _positive_int(s: str) -> int | None:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentError(f"expected integer, got {s!r}")

    if v <= 0:
        return None
    return v


# Custom actions -------------------------------------------------------------------
class ShowVersionAction(argparse.Action):
    def __init__(self, option_strings, dest, help):
        """
        Pretty print the version number with the Machop logo and exit.
        For more details: https://docs.python.org/3/library/argparse.html#action
        """
        super().__init__(option_strings, dest, nargs=0, help=help)

    def __call__(self, parser, *_):
        print(LOGO)
        parser.exit()


class ShowInfoAction(argparse.Action):
    def __init__(self, option_strings, dest=SUPPRESS, const=None, **kwargs):
        """
        Pretty print the version number with the Machop logo and exit.
        For more details: https://docs.python.org/3/library/argparse.html#action
        """
        super().__init__(option_strings, dest, nargs="?", const=const, **kwargs)

    def __call__(self, parser, _, values, *__):
        choice = values if values is not None else self.default

        if choice in ["model", "all"]:
            self._generate_table(list(models.model_map.keys()), "Supported Models")
        if choice in ["dataset", "all"]:
            self._generate_table(AVAILABLE_DATASETS, "Supported Datasets", cols=2)

        parser.exit()

    def _generate_table(self, data, title, cols=3):
        table = []

        rows = (len(data) + cols - 1) // cols
        for col in range(cols):
            split = data[col * rows : (col + 1) * rows]
            if len(split) < rows:
                # Pad split with empty rows to make up for the difference
                split.extend([""] * (rows - len(split)))
            table.append(split)

        table = list(zip(*table))
        print(title)
        print(tabulate(table, tablefmt="pretty"))
