"""Entrypoint for running an experiment with episodes executing in parallel."""

from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.run_parallel import main  # noqa: E402

if __name__ == "__main__":
    main()
