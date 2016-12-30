"""Script for fitting neural net to a training model."""
import click
import numpy as np

import data
import util
from nn import create_net

@click.command()
@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
              help='Path to the config module.')
@click.option('--weights_from', default=None, show_default=True,
              help=’Path to the weights file')
def main(cnf, weights_from):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    files = data.get_image_files(config.get('train_dir'))
    file_names = data.get_names(files)
    file_labels = data.get_labels(names).astype(np.float32)

    net = create_net(config)

    try:
        net.load_params_from(weights_from)
        print("using weights from {}".format(weights_from))
    except IOError:
        print("error loading weights from file")

    print("fitting model ...")
    net.fit(file_names, file_labels)

if __name__ == '__main__':
    main()
