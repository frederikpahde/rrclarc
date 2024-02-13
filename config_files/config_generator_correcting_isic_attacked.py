import copy
import os
import yaml

config_dir = "correcting_isic_attacked"

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

_base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'img_size': 224,
    'wandb_project_name': 'your_project_name',
    'unique_wandb_ids': True,
    'attacked_classes': ['MEL'],
    'p_artifact': 0.1
}

lsb_trigger = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut " \
              "labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores " \
              "et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem " \
              "ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et " \
              "dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea " \
              "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum " \
              "dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et " \
              "dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea " \
              "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem " \
              "vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu " \
              "feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum " \
              "zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer " \
              "adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat " \
              "volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis " \
              "nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate " \
              "velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et " \
              "iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla " \
              "facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod " \
              "mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, " \
              "sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad " \
              "minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo " \
              "consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, " \
              "vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea " \
              "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum " \
              "dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et " \
              "dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea " \
              "rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum " \
              "dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod " \
              "eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no " \
              "rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor " \
              "sit amet, consetetur"


def store_local(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = f"checkpoints/checkpoint_{model_name}_isic_attacked_lsb.pth"

    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']
    config['data_paths'] = [local_config['isic2019_dir']]
    config['batch_size'] = local_config['local_batch_size']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = f"checkpoints/checkpoint_{model_name}_isic_attacked_lsb.pth"
    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    config['data_paths'] = ["/mnt/dataset_isic2019"]
    config['batch_size'] = 64

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name, layer_name in [
    ('vgg16', 'features.28'),
    ('resnet18', 'last_conv'),
    ('efficientnet_b0', 'last_conv'),
]:

    _base_config['model_name'] = model_name

    for layer_name in [layer_name]:
        _base_config['layer_name'] = layer_name

        for artifact_type in [
            "lsb",
        ]:

            _base_config["artifact_type"] = artifact_type
            base_config = copy.deepcopy(_base_config)

            if artifact_type == "lsb":
                base_config['lsb_factor'] = 2 if model_name == "efficientnet_b0" else 3
                base_config['lsb_trigger'] = lsb_trigger
            for artifact in [
                "artificial"
            ]:
                base_config['artifact'] = artifact
                for lr in [
                    0.0001,
                ]:
                    base_config['lr'] = lr
                    optim_name = "adam" if model_name == "efficientnet_b0" else "sgd"

                    base_config['optimizer'] = optim_name

                    ### VANILLA
                    config_vanilla = copy.deepcopy(base_config)
                    method = 'Vanilla'
                    config_vanilla['method'] = method
                    config_vanilla['lamb'] = 0.0
                    config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_{artifact}_{layer_name}"
                    store_local(config_vanilla, config_name)
                    store_cluster(config_vanilla, config_name)

                    config_vanilla = copy.deepcopy(base_config)
                    method = 'Vanilla'
                    config_name = f"{model_name}_{method}_{artifact_type}-{base_config['p_artifact']}"
                    config_vanilla['num_epochs'] = 0
                    config_vanilla['method'] = method
                    store_cluster(config_vanilla, config_name)
                    store_local(config_vanilla, config_name)

                    method = 'RRR_ExpLogSum'
                    for lamb in [
                        0.05, 0.1,
                        0.5, 1.0,
                        5, 10,
                        50, 100,
                        500, 1000,
                        5000, 10000,
                        50000, 100000,
                        500000, 1000000,
                        5000000, 10000000,
                    ]:
                        base_config['method'] = method
                        base_config['lamb'] = lamb
                        # base_config['num_epochs'] = 5
                        config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                        store_local(base_config, config_name)
                        store_cluster(base_config, config_name)

                    for direction_mode in [
                        'signal',
                        "svm",
                        "lasso",
                        "ridge",
                        "logistic"
                    ]:
                        base_config['direction_mode'] = direction_mode

                        for method in [
                            'AClarc',
                            'PClarc'
                        ]:
                            base_config['method'] = method
                            lamb = 1.0
                            base_config['lamb'] = lamb
                            config_name = f"{model_name}_{method}_{direction_mode}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                            store_local(base_config, config_name)
                            store_cluster(base_config, config_name)

                    for direction_mode in [
                        'signal',
                        # 'svm',
                        # 'lasso',
                        # 'ridge',
                        # 'logistic',
                    ]:
                        base_config['direction_mode'] = direction_mode
                        for lamb2 in [
                            # "cosine_mean",
                            "l2_mean",
                            # "l1_mean",
                        ]:

                            base_config['compute'] = lamb2

                            for lamb in [
                                0.05, 0.1,
                                0.5, 1.0,
                                10,
                                50, 100,
                                500, 1000,
                                5000, 10000,
                                50000, 100000,
                                500000, 1000000,
                                5000000, 10000000,
                            ]:
                                base_config['lamb'] = lamb
                                for method in [
                                    'RRClarc'
                                ]:
                                    base_config['method'] = method
                                    for criterion in [
                                        'all',
                                        'allrand',
                                        'logprobs',
                                    ]:
                                        base_config['criterion'] = criterion
                                        base_config['cav_type'] = "cavs_max"
                                        config_name = f"{model_name}_{method}_{direction_mode}_{optim_name}_lr{lr}_lamb{lamb}_comp{lamb2}_crit{criterion}_{artifact}_{layer_name}"
                                        store_local(base_config, config_name)
                                        store_cluster(base_config, config_name)
