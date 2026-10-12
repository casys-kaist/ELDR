# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Language main evaluation with load-balancing and Domain baselines."""

from eldr.experiments.runner.experiment import (
    BASELINES,
    main,
    main_worker_group,
    prepare_domain_baseline,
)

POLICIES = BASELINES
RATES = (20, 40, 60, 80, 100)
configure = main_worker_group
prepare = prepare_domain_baseline

if __name__ == "__main__":
    main("main_language")
