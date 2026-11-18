import os
import json
import yaml
import mlflow
from src.profiler_study import run_profiler_study
from src.workers_study import run_workers_study
from src.batch_study import run_batch_study
from src.optim_sched_study import run_optim_sched_study
from src.grad_accum_and_resume import run_grad_accum_study, run_checkpoint_resume_study


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["experiment"]["tracking_uri"])
    mlflow.set_experiment(config["experiment"]["mlflow_experiment"])

    with mlflow.start_run(run_name="Lab6_Full_Pipeline"):
        mlflow.log_params({
            "student_id": config["experiment"]["student_id"],
            "seed": config["experiment"]["seed"],
            "subset_size": config["data"]["subset_size"],
            "conv1_channels": config["model"]["conv1_channels"],
            "conv2_channels": config["model"]["conv2_channels"],
        })

        prof_shares = run_profiler_study(config)
        workers_res = run_workers_study(config)
        batch_res = run_batch_study(config)
        optim_res = run_optim_sched_study(config)
        accum_res = run_grad_accum_study(config)
        resume_res = run_checkpoint_resume_study(config)

        for k, v in prof_shares.items():
            mlflow.log_metric(f"profiler_{k}_pct", v)
        for row in batch_res:
            mlflow.log_metric(f"bs_{row['batch_size']}_acc", row["final_accuracy"])
            mlflow.log_metric(f"bs_{row['batch_size']}_throughput", row["throughput_samples_sec"])
        mlflow.log_metric("grad_accum_256_acc", accum_res["native_256"]["final_acc"])
        mlflow.log_metric("grad_accum_64x4_acc", accum_res["accum_64x4"]["final_acc"])
        mlflow.log_metric("resumed_final_acc", resume_res[-1]["acc"])

        summary = {
            "profiler_shares_pct": prof_shares,
            "workers_study": workers_res,
            "batch_study": batch_res,
            "optim_sched_study": optim_res,
            "grad_accum_study": accum_res,
            "checkpoint_resume_history": resume_res,
        }

        summary_path = "results/lab6_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        mlflow.log_artifacts("results")
        print(f"All Lab 6 studies complete. Results & MLflow artifacts saved to results/ and mlruns/.")