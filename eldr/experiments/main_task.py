"""Task main evaluation; add --execute to launch, --replay to redraw."""

from eldr.experiments.runner.experiment import (
    BASELINES,
    main,
    main_worker_group,
    prepare_domain_baseline,
)

POLICIES = BASELINES  # RR, Random, JSQ, P2C, Domain, ELDR.
RATES = (20, 40, 60, 80, 100)
configure = main_worker_group
prepare = prepare_domain_baseline

if __name__ == "__main__":
    main("main_task")
