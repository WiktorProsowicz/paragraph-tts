
# Set up virtual environment.
setup_venv:
    #!/usr/bin/env bash
    echo "Setting up virtual environment..."
    python3.11 -m venv --system-site-packages .venv
    echo 'export MPLBACKEND=Agg' >> .venv/bin/activate

# Build and install dependencies from `extern` directory.
build_external_repos:
    #!/usr/bin/env bash
    echo "Installing dependencies..."
    source .venv/bin/activate
    pip install -e extern/torch-dev-utils
    pip install -e extern/Comprehensive-Transformer-TTS

# Install project with its deps into the virtual environment.
build_project:
    #!/usr/bin/env bash
    echo "Installing project..."
    source .venv/bin/activate
    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
    pip install -e .

# Install project with its dev deps into the virtual environment.
build_project_dev:
    #!/usr/bin/env bash
    echo "Installing project in development mode..."
    source .venv/bin/activate
    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
    pip install -e .[dev]

# Run pre-commit hooks on all files.
run_pre_commit:
    #!/usr/bin/env bash
    echo "Running pre-commit hooks..."
    source .venv/bin/activate
    pre-commit run --all-files
