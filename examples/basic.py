from azar_run_flow import run_experiment


def experiment(run, config):
    for step in range(config["steps"]):
        loss = 1.0 / (step + 1)

        run.metric(
            "loss",
            loss,
            step=step,
        )

    return {
        "final_loss": loss,
    }


if __name__ == "__main__":
    run_experiment(
        experiment,
        config={
            "steps": 20,
            "method": "dummy",
            "seed": 42,
        },
        experiment_name="azar-test",
        run_name="basic-run",
    )