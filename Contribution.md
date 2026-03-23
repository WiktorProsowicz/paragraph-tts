This repository contains code used to process large amounts of `<text, sound>` pairs, analyze the data and train Text-to-Speech models. The repository uses the following tech stack:

- `Python`
- `Pytorch + Torch Lightning` - model building and training
- `MLFlow, Tensorboard` - experiment tracking, reproducibility

## Code guidelines

1. Use guidelines defined in the Google Python Style Guide (https://google.github.io/styleguide/pyguide.html)
2. Avoid writing too long functions. If possible, split them into several ones.
3. Avoid using typedefs to define complex data structures. Make the most out of `Pydantic` or `dataclasses`.
4. Use docstrings for modules, classes and functions. For public functions, use Google docstring format. For private functions, use just a short description. Always in 3rd person.
5. Avoid using unnecessary `try-except` blocks and, in general, too nested code.
6. Every module should have its `_logger()` function, defining a logger as `logging.getLogger(__name__)`.
7. Do not use exceptions to handle unrecoverable problems. Use just a critical log and sys exit.
8. Do not use comments, unless absolutely necessary.
9. Do not use redundant variables, unless they contribute to the readability of the code. As a rule of thumb, if a variable is asigned a short expression, just use it directly.
