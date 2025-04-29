# -*- coding: utf-8 -*-

from ttab.loads.datasets.dataset_shifts import NoShiftProperty
from ttab.scenarios import HomogeneousNoMixture, Scenario, TestCase, TestDomain

default_scenarios = {
    "S1": Scenario(
        task="classification",
        model_name="vit_large_patch16_224",
        model_adaptation_method="no_adaptation",
        model_selection_method="last_iterate",
        base_data_name="affectnet",
        src_data_name="affectnet",
        test_domains=[
            TestDomain(
                base_data_name="affectnet",
                # data_name must exactly match what you trained on:
                data_name="affectnet",
                # load your in‐distribution (no shift) split
                shift_type="no_shift",
                shift_property=NoShiftProperty(has_shift=False),
                domain_sampling_name="uniform",
                domain_sampling_value=None,
                domain_sampling_ratio=1.0,
            )
        ],
        test_case=TestCase(
            inter_domain=HomogeneousNoMixture(has_mixture=False),
            batch_size=32,
            data_wise="batch_wise",
            offline_pre_adapt=False,
            episodic=False,
            # no shuffling or extra augment in‐domain
            intra_domain_shuffle=False,
        ),
    ),
}
