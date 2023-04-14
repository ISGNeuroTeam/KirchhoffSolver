#.SILENT

BASE_PTH = $(shell pwd)

GENERATE_VERSION = $(shell cat ksolver/__init__.py | grep __version__ | head -n 1 | sed -re 's/[^"]+//' | sed -re 's/"//g' )
GENERATE_BRANCH = $(shell git name-rev $$(git rev-parse HEAD) | cut -d\  -f2 | sed -re 's/^(remotes\/)?origin\///' | tr '/' '_')
SET_VERSION = $(eval VERSION=$(GENERATE_VERSION))
SET_BRANCH = $(eval BRANCH=$(GENERATE_BRANCH))


CONDA = conda/miniconda/bin/conda
ENV_PYTHON = venv/bin/python3.9


conda/miniconda.sh:
	echo Download Miniconda
	mkdir -p conda
	wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -O conda/miniconda.sh; \

conda/miniconda: conda/miniconda.sh
	bash conda/miniconda.sh -b -p conda/miniconda; \

install_conda: conda/miniconda

conda/miniconda/bin/conda-pack: conda/miniconda
	conda/miniconda/bin/conda install conda-pack -c conda-forge  -y

install_conda_pack: conda/miniconda/bin/conda-pack

clean_conda:
	rm -rf ./conda

venv: conda/miniconda
	$(CONDA) create --copy -p ./venv -y
	$(CONDA) install -p ./venv python==3.9.7 -y;
	$(ENV_PYTHON) -m pip  install --no-input  -r requirements.txt


clean_venv:
	echo "Cleaning venv..."
	rm -rf ./venv

test: venv
	echo "Testing..."
	# export PYTHONPATH=./ot_simple_connector/:./tests/; ./venv/bin/python -m unittest

clean_test: clean_venv
	echo "Cleaning after test..."

publish: venv
	./venv/bin/python3 ./setup.py sdist bdist_wheel

clean_dist:
	rm -rf dist/ KSolver.egg-info

clean: clean_venv clean_dist clean_conda
