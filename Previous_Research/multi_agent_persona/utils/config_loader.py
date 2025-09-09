import os

import yaml


def load_config(config_file='./config/config.yaml'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config


def load_config_2(config_file='./config/config_2.yaml'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config


def get_model_settings(config):
    generator_model = config.get('generator_model')
    discriminator_model = config.get('discriminator_model')

    if not generator_model or not discriminator_model:
        raise ValueError("Generator model or discriminator model not specified in the configuration.")

    return generator_model, discriminator_model
