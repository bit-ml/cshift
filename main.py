import os
import sys
from datetime import datetime

import numpy as np
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from experts.experts import Experts
from graph.edges.graph_edges import Edge
from graph.graph import MultiDomainGraph
from utils.utils import DummySummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import configparser


def build_space_graph(config, silent, iter_no=1):
    if config.has_option('GraphStructure', 'selector_map'):
        selector_map_str = config.get('GraphStructure',
                                      'selector_map').split(",")
        selector_map = [int(token) for token in selector_map_str]
    else:
        selector_map = None

    all_experts = Experts(dataset_name=config.get('General', 'DATASET_NAME'),
                          full_experts=False,
                          selector_map=selector_map)

    md_graph = MultiDomainGraph(
        config,
        all_experts,
        device,
        iter_no=iter_no,
        silent=silent,
    )
    return md_graph


def eval_1hop_ensembles(space_graph, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')
        datetime = config.get('Run id', 'datetime')
        writer = SummaryWriter(log_dir=f'%s/%s_1hop_ens_%s' %
                               (tb_dir, tb_prefix, datetime),
                               flush_secs=30)
    for expert in space_graph.experts.methods:
        end_id = expert.identifier

        edges_1hop = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if not edge_xk.trained:
                continue
            if edge_xk.expert2.identifier == end_id:
                edges_1hop.append(edge_xk)

        if len(edges_1hop) == 0:
            continue

        # 2. Eval each ensemble
        Edge.eval_all_1hop_ensembles(edges_1hop, device, writer, config)
    writer.close()


def save_1hop_ensembles(space_graph, config):
    writer = DummySummaryWriter()

    for expert in space_graph.experts.methods:
        end_id = expert.identifier
        edges_1hop = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if not edge_xk.trained:
                continue
            if edge_xk.expert2.identifier == end_id:
                edges_1hop.append(edge_xk)

        # 2. Eval each ensemble
        if len(edges_1hop) > 0:
            Edge.save_1hop_ensemble(edges_1hop, device, config)

    writer.close()


def train_edge_models(space_graph, start_epoch, n_epochs, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')
        datetime = config.get('Run id', 'datetime')
        writer = SummaryWriter(log_dir=f'%s/%s_train_edge_models_%s' %
                               (tb_dir, tb_prefix, datetime),
                               flush_secs=30)

    for net_idx, net in enumerate(space_graph.edges):
        print("[%2d] Train" % net_idx, net)

        net.train(start_epoch=start_epoch,
                  n_epochs=n_epochs,
                  device=device,
                  writer=writer)
        net.trained = True

    writer.close()


def load_edge_models(graph, epoch):
    print("Load nets from checkpoints.", colored("Epoch: %2d" % epoch, "red"))

    for _, edge in enumerate(graph.edges):
        path = os.path.join(edge.load_model_dir, 'epoch_%05d.pth' % (epoch))
        if os.path.exists(path):
            edge.net.load_state_dict(torch.load(path))
            edge.net.module.eval()
            edge.trained = True
        else:
            print(
                'model: %s_%s UNAVAILABLE' %
                (edge.expert1.domain_name, edge.expert2.domain_name), path)


def prepare_store_folders(config, all_experts):
    train_store_path = config.get('PathsIter', 'ITER_TRAIN_STORE_PATH')
    valid_store_path = config.get('PathsIter', 'ITER_VALID_STORE_PATH')
    test_store_path = config.get('PathsIter', 'ITER_TEST_STORE_PATH')
    assert (len(train_store_path.split('\n')) == len(
        valid_store_path.split('\n')) == len(test_store_path.split('\n')) == 1)

    for expert in all_experts.methods:
        save_to_dir = "%s/%s" % (train_store_path, expert.identifier)
        os.makedirs(save_to_dir, exist_ok=True)

        save_to_dir = "%s/%s" % (valid_store_path, expert.identifier)
        os.makedirs(save_to_dir, exist_ok=True)

        save_to_dir = "%s/%s" % (test_store_path, expert.identifier)
        os.makedirs(save_to_dir, exist_ok=True)


def main(argv):

    config = configparser.ConfigParser()
    config.read(argv[1])

    config.set('Run id', 'datetime', str(datetime.now()))
    print(config.get('Run id', 'datetime'))
    print(colored("Config file: %s" % argv[1], "red"))
    print("load_path", config.get('Edge Models', 'load_path'))

    train_flag = config.getboolean('General', 'do_train')
    test_flag = config.getboolean('General', 'do_test')
    store_predictions_flag = config.getboolean('General',
                                               'do_store_predictions')

    if not (train_flag or test_flag or store_predictions_flag):
        print(
            "No operation to do please set one of the following: do_train, do_test, or do_store_predictions"
        )
        return

    # Build graph
    silent = config.getboolean('Logs', 'silent')
    graph = build_space_graph(config, silent=silent)

    # Load models
    start_epoch = config.getint('Edge Models', 'start_epoch')
    if start_epoch > 0:
        load_edge_models(graph, epoch=start_epoch)
        print("==================")

    # Train models
    if train_flag:
        n_epochs = config.getint('Edge Models', 'n_epochs')
        train_edge_models(graph,
                          start_epoch=start_epoch,
                          n_epochs=n_epochs,
                          silent=silent,
                          config=config)

    # Test models - fixed epoch
    if test_flag:
        eval_1hop_ensembles(graph, silent=silent, config=config)

    # Save data for next iter
    if store_predictions_flag:
        if config.has_option('GraphStructure', 'selector_map'):
            selector_map_str = config.get('GraphStructure',
                                          'selector_map').split(",")
            selector_map = [int(token) for token in selector_map_str]
        else:
            selector_map = None
        all_experts = Experts(full_experts=False,
                              dataset_name=config.get('General',
                                                      'DATASET_NAME'),
                              selector_map=selector_map)
        prepare_store_folders(config=config, all_experts=all_experts)
        save_1hop_ensembles(graph, config=config)


if __name__ == "__main__":
    main(sys.argv)
