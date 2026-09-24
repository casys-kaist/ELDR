"""Pure locality (tau=0) versus positive bands with at least two candidates."""

from eldr.experiments.runner.experiment import main, plan_worker_group

RATES = (20, 40, 60, 80, 100)
TAUS = (0.0, 0.1, 0.2, 0.3)


def configure(config, output, policies):
    variants = [("rr", "rr", {})] + [
        (f"tau-{tau:g}", "eldr-static", dict(tau=tau)) for tau in TAUS
    ]
    return [plan_worker_group(config, output / config["model"], variants)]


if __name__ == "__main__":
    main("locality_band")
