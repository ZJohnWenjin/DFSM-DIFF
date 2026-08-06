import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from models.DFMM_pretrain import FMM_Pretrain
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default='config.yml', help="Path to the config file"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="log",
        action="store",
    )
    # encoders
    parser.add_argument(
        "--train_specific_en_decoder",
        default=True,
        action="store",
        help="Whether to train specific encoder and decoder",
    )
    parser.add_argument(
        "--train_mapping_encoder",
        default=True,
        action="store",
        help="Whether to train mapping encoder",
    )

    # diffusion
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--resume_training", type=int, default=False, help="Resume training the diffusion model"
    )
    parser.add_argument(
        "--sample",
        default=False,
        action="store",
        help="Whether to produce samples from the model",
    )
    
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_obj = yaml.safe_load(f)

    if not isinstance(cfg_obj, dict):
        raise ValueError(f"YAML root must be a mapping (dict), got {type(cfg_obj).__name__}")

    new_config = dict2namespace(cfg_obj)

    if os.path.exists(args.log_path):
        overwrite = False
        response = input("Folder already exists. Overwrite? (Y/N)")
        if response.upper() == "Y":
            overwrite = True

        if overwrite:
            shutil.rmtree(args.log_path)
            os.makedirs(args.log_path)
    else:
        os.makedirs(args.log_path)

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(20)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))

    return args, new_config


def dict2namespace(config):
    if isinstance(config, dict):
        ns = argparse.Namespace()
        for k, v in config.items():
            setattr(ns, k, dict2namespace(v))
        return ns
    elif isinstance(config, list):
        return [dict2namespace(v) for v in config]
    else:
        return config

def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))


    if args.train_specific_en_decoder or args.train_mapping_encoder:
        FMM_P = FMM_Pretrain(config)
        if args.train_specific_en_decoder:
            FMM_P.pretrain(specific_encoder=True)
        if args.train_mapping_encoder:
            FMM_P.pretrain(specific_encoder=False)

    runner = Diffusion(args, config)
    if args.sample:
        runner.sample()
    else:
        runner.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())
