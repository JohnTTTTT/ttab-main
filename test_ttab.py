# weight_stats_ttab.py

import numpy as np
import torch

import parameters
import ttab.configs.utils as configs_utils
import ttab.loads.define_dataset as define_dataset
from ttab.benchmark import Benchmark
from ttab.loads.define_model import define_model, load_pretrained_model
from ttab.model_adaptation import get_model_adaptation_method
from ttab.model_selection import get_model_selection_method


def main(init_config):
    # 1) load config & scenario
    config, scenario = configs_utils.config_hparams(config=init_config)

    # 2) build test loader (you can skip this if you only want weights)
    test_data_cls = define_dataset.ConstructTestDataset(config=config)
    test_loader = test_data_cls.construct_test_loader(scenario=scenario)

    # 3) instantiate & load
    model = define_model(config=config)
    load_pretrained_model(config=config, model=model)

    # 4) dump per-layer stats
    print("\n>> Weight statistics per parameter:")
    for name, param in model.named_parameters():
        mean = param.detach().cpu().numpy().mean()
        print(f"{name:40s}  mean={mean: .6e}")

    # 5) (optional) run the usual TTAB eval to double-check accuracy
    model_adapt = get_model_adaptation_method(scenario.model_adaptation_method)(
        meta_conf=config, model=model
    )
    model_select = get_model_selection_method(scenario.model_selection_method)(
        meta_conf=config, model_adaptation_method=model_adapt
    )
    benchmark = Benchmark(
        scenario=scenario,
        model_adaptation_cls=model_adapt,
        model_selection_cls=model_select,
        test_loader=test_loader,
        meta_conf=config,
    )
    print("\n>> Running TTAB eval:")
    benchmark.eval()


if __name__ == "__main__":
    conf = parameters.get_args()
    main(init_config=conf)
