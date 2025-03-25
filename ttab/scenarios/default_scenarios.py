# -*- coding: utf-8 -*-

from ttab.loads.datasets.dataset_shifts import SyntheticShiftProperty
from ttab.scenarios import HomogeneousNoMixture, Scenario, TestCase, TestDomain

default_scenarios = {
    "S1": Scenario(
        task="classification",
        model_name="resnet18",
        model_adaptation_method="no_adaptation",
        model_selection_method="last_iterate",
        base_data_name="fairface",
        src_data_name="fairface",
        test_domains=[
            TestDomain(
                base_data_name="fairface",
                data_name="fairface",
                shift_type="natural",
                shift_property=SyntheticShiftProperty(
                    shift_degree=5,
                    shift_name="gaussian_noise",
                    version="deterministic",
                    has_shift=True,
                ),
                domain_sampling_name="uniform",
                domain_sampling_value=None,
                domain_sampling_ratio=1.0,
            )
        ],
        test_case=TestCase(
            inter_domain=HomogeneousNoMixture(has_mixture=False),
            batch_size=16,
            data_wise="batch_wise",
            offline_pre_adapt=False,
            episodic=False,
            intra_domain_shuffle=True,
        ),
    ),
}
