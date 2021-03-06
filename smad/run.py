import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from smad.runner import Runner


log = logging.getLogger(__name__)

config_path = str(Path(sys.argv[1]).parent)
config_name = str(Path(sys.argv[1]).stem)
sys.argv.pop(1)


@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.start_run(run_name=cfg.mlflow.run_name)
    mlflow.log_params(cfg.params)
    mlflow.log_param("cwd", os.getcwd())

    runner = Runner(cfg)
    runner.run()

    mlflow.log_artifacts(".hydra", "hydra")


if __name__ == "__main__":
    main()
