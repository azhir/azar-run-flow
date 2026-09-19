import time

from azar_run_flow import run_experiment


def experiment(run, config):
    for step in range(config["steps"]):
        loss = 1.0 / (step + 1)

        run.metric(
            "loss",
            loss,
            step=step,
        )

        time.sleep(0.1)

    return {"final_loss": loss}


config = {
    "steps": 20,
    "method": "dummy",
}


run_experiment(
    experiment,
    config=config,
    experiment_name="azar-test",
    run_name="basic-run",
    seed=42,
)