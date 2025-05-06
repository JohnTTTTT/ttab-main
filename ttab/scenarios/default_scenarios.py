# -*- coding: utf-8 -*-

from ttab.loads.datasets.dataset_shifts import SyntheticShiftProperty
from ttab.scenarios import HomogeneousNoMixture, Scenario, TestCase, TestDomain

default_scenarios = {
    "S1": Scenario(
        task="classification",
        model_name="vit",
        model_adaptation_method="no_adaptation",
        model_selection_method="last_iterate",
        base_data_name="affectnet",
        src_data_name="affectnet",
        test_domains=[
            TestDomain(
                base_data_name="affectnet",
                data_name="affectnet",
                shift_type="no_shift",
                shift_property=SyntheticShiftProperty(
                    shift_degree=5,
                    shift_name="gaussian_noise",
                    version="deterministic",
                    has_shift=False,
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
