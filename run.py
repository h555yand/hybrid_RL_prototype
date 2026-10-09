"""Entrypoint for running an experiment."""

from tbp.monty.frameworks.run_env import setup_env

setup_env()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402


@hydra.main(
    config_path="src/tbp/hybrid_rl/conf",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    experiment = hydra.utils.instantiate(cfg.experiment)
    with experiment:
        experiment.run()


if __name__ == "__main__":
    main()
