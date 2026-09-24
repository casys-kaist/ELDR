"""Language main evaluation with load-balancing and Domain baselines."""

from eldr.experiments.runner.experiment import (
    BASELINES,
    main,
    main_worker_group,
    prepare_domain_baseline,
)

POLICIES = BASELINES  # Same policies/protocol as Task, with Language inputs.
RATES = (20, 40, 60, 80, 100)
configure = main_worker_group
prepare = prepare_domain_baseline

if __name__ == "__main__":
    main("main_language")
