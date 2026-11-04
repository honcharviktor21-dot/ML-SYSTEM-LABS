import yaml
from src.verify import verify_gradients
from src.nn_moons import run_moons_benchmark
from src.execution import benchmark_pytorch_execution

if __name__ == "__main__":
    print("=== Launching Lab 5 Autograd & Framework Benchmark Pipeline ===\n")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    verify_gradients(config)
    run_moons_benchmark(config)
    benchmark_pytorch_execution(config)

    print("Pipeline Execution Complete. Results written to results/ directory.")