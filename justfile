set positional-arguments

# Set up virtual environment.
setup_venv:
    #!/usr/bin/env bash
    echo "Setting up virtual environment..."
    python3.11 -m venv --system-site-packages .venv
    echo 'export MPLBACKEND=Agg' >> .venv/bin/activate

# Build and install dependencies from `extern` directory.
build_external_repos onnxrt="cpu":
    #!/usr/bin/env bash
    echo "Installing dependencies..."
    source .venv/bin/activate
    pip install -e extern/Comprehensive-Transformer-TTS[$1]
    pip install -e extern/torch-dev-utils

# Install project with its deps into the virtual environment.
build_project:
    #!/usr/bin/env bash
    echo "Installing project..."
    source .venv/bin/activate
    pip install -e .

# Install project with its dev deps into the virtual environment.
build_project_dev:
    #!/usr/bin/env bash
    echo "Installing project in development mode..."
    source .venv/bin/activate
    pip install -e .[dev]

# Run mlflow tracking server for experiment tracking and artifact storage.
setup_mlflow_server:
    #!/usr/bin/env bash
    echo "Setting up MLflow server..."
    source .venv/bin/activate
    mlflow server --backend-store-uri sqlite:///mlflow_tracking.db --default-artifact-root ./mlflow_artifacts/ --host 0.0.0.0 --port 5000 --workers 1 --cors-allowed-origins "*"

# Run pre-commit hooks on all files.
run_pre_commit:
    #!/usr/bin/env bash
    echo "Running pre-commit hooks..."
    source .venv/bin/activate
    pre-commit run --all-files

# Run an arbitrary python script inside the virtual environment.
run_python *args:
    #!/usr/bin/env bash
    source .venv/bin/activate
    python "$@"
