import sys
import os
import numpy
import pandas
import sklearn
import scipy
import matplotlib
import seaborn
import torch
import torchvision
import torchinfo
import thop
import onnx
import onnxruntime
import mlflow
import memory_profiler
import psutil
import codecarbon
import fastapi
import uvicorn
import pytest
import httpx
import locust
import requests
import pyarrow
import joblib
import tqdm

def print_and_save_versions():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "versions.txt")

    packages = [
        ("Python", sys.version.replace("\n", " ")),
        ("numpy", numpy.__version__),
        ("pandas", pandas.__version__),
        ("scikit-learn", sklearn.__version__),
        ("scipy", scipy.__version__),
        ("matplotlib", matplotlib.__version__),
        ("seaborn", seaborn.__version__),
        ("torch", torch.__version__),
        ("torchvision", torchvision.__version__),
        ("torchinfo", torchinfo.__version__),
        ("thop", thop.__version__),
        ("onnx", onnx.__version__),
        ("onnxruntime", onnxruntime.__version__),
        ("mlflow", mlflow.__version__),
        ("memory-profiler", memory_profiler.__version__),
        ("psutil", psutil.__version__),
        ("codecarbon", codecarbon.__version__),
        ("fastapi", fastapi.__version__),
        ("uvicorn", uvicorn.__version__),
        ("pytest", pytest.__version__),
        ("httpx", httpx.__version__),
        ("locust", locust.__version__),
        ("requests", requests.__version__),
        ("pyarrow", pyarrow.__version__),
        ("joblib", joblib.__version__),
        ("tqdm", tqdm.__version__),
    ]

    with open(output_file, "w") as f:
        for name, version in packages:
            line = f"{name}=={version}"
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    print_and_save_versions()
