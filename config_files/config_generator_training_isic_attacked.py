import os
import yaml

config_dir = "training_isic"

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 150,
    'device': 'cuda',
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 20,
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'wandb_project_name': 'your_project_name',
    'img_size': 224,
    'attacked_classes': ['MEL']
}


def store_local(config, config_name):
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['isic2019_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def store_cluster(config, config_name):
    config['batch_size'] = 64 if config['model_name'] == 'efficientnet_b4' else 128
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/dataset_isic2019"]

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

lsb_trigger = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur"
for p_artifact in [
    0.05,
    0.1,
    0.125,
]:
    base_config["p_artifact"] = p_artifact

    for model_name in [
        'vgg16',
        'efficientnet_b0',
        'resnet18',
    ]:
        for artifact_type in [
            "lsb",
        ]:
            
            base_config['model_name'] = model_name
            base_config['artifact_type'] = artifact_type
            base_config['artifact_type'] = artifact_type

            base_config["lsb_trigger"] = lsb_trigger
            lsb_factor = 2 if model_name == "efficientnet_b0" else 3
            base_config['lsb_factor'] = lsb_factor
            lrs = [0.001] if model_name == "efficientnet_b0" else [0.005]
            for lr in lrs:
                base_config['lr'] = lr
                for pretrained in [
                    True,
                ]:

                    base_config['pretrained'] = pretrained
                    optims = ["adam"] if model_name == "efficientnet_b0" else ["sgd"]
                    for optim_name in optims:
                        base_config['optimizer'] = optim_name
                        config_name = f"attacked-{p_artifact}-{artifact_type}-{lsb_factor}_{model_name}_{optim_name}_lr{lr}_pretrained-{pretrained}"
                        store_local(base_config, config_name)
                        store_cluster(base_config, config_name)
