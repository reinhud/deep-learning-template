# Deep Learning with Pytorch Lightning and the DVC Stack

This is a template for a deep learning project using.

Happy coding :rocket:

## Getting Started

### Dependencies

- Docker Engine - https://docs.docker.com/engine/install/

### Installing

Before starting, make sure you have the latest versions of Docker installed.

Run the following commands to pull this repo from github and get to src folder:

```
git clone https://github.com/reinhud/deep-learning-template

```

Create the `.env` files or modify the `.env.example` files:

```
touch .env
```

## Usage

### Remote Development

### Training pipeline and workflow

The experiment pipeline is set up using Pytorch Lightning, DVC and Hydra,
with parameterization of the pipeline defined in the `/config folder`.
When involking a new experiment run like `dvc exp run`,
the /config/train.yaml file is used, pulling all other configs from the
files defined there. A `/params.yaml` file is automatically created that is used by DVC to
start the run. The configs defined can be overridden easily by specifying them in the run command
like `dvc exp run --set-param "trainer.max_epochs=5"`.

Outputs of the experiment can be found within the /output folder.

### Common commands

Debug the experiment pipeline:

```bash
dvc exp run -f --set-param "debug=fdr"
```

Running multiple experiments:

```bash
dvc exp run --queue -S "experiment=A"
dvc exp run --queue -S "experiment=B"

dvc queue start
```

Running all fast tests:

```
make test
```

Running all the tests, including slow ones:

```
make test-full
```

### Testing

While you can still use the tests by involking pytest directly
for full customizability of the testing runs like

```
poetry run pytest tests/test_train.py
```

there are make command provided for quick access to the testing suite.

### Project Structure

```bash

```

## Authors

@Lukas Reinhardt

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

Inspiration, useful repos, code snippets, etc.:

- https://github.com/ashleve/lightning-hydra-template
