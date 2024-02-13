import os
from argparse import ArgumentParser

import yaml

from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    run_preprocessing(config)


def run_preprocessing(config):

    if "isic" in config['dataset_name']:
        classes = [0]
    elif "imagenet" in config['dataset_name']:
        classes = [0]
    elif "celeba" in config['dataset_name']:
        classes = [0, 1]
    elif "bone" in config['dataset_name']:
        classes = [2]
    else:
        classes = range(5)
    for class_id in classes:
        run_collect_relevances_and_activations({**config,
                                                'class_id': class_id})


if __name__ == "__main__":
    main()
