
# Set up virtual environment.
setup_venv:
    #!/usr/bin/env bash
    echo "Setting up virtual environment..."
    python3.11 -m venv --system-site-packages .venv

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
    pip install -e .

# Install project with its dev deps into the virtual environment.
build_project_dev:
    #!/usr/bin/env bash
    echo "Installing project in development mode..."
    source .venv/bin/activate
    pip install -e .[dev]
