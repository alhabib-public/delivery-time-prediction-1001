# Food delivery time prediction — everything runs inside the Docker image.
IMAGE  ?= delivery-time-prediction
MOUNTS := -v $(CURDIR):/app
RUN    := docker run --rm $(MOUNTS) $(IMAGE)
RUN_IT := docker run --rm -it $(MOUNTS) $(IMAGE)

# what `make run` executes inside the container, e.g. make run FILE="-m delivery.train --n-rows 3000 --cv 3"
FILE   ?= -m delivery.train --help
# experiment module under delivery/exp/ and candidate subset for `make train` / `make evaluate`
EXP    ?= v001_hgb_vs_mlp_kfold5
MODELS ?= median,ridge,hgb,mlp
CV     ?= 5
SEED   ?= 0
PORT   ?= 8899
# the report notebook: `make run-report` re-executes it in place, `make report` exports it to ./COMPLETED_ASSIGNMENT.html
NB       ?= REPORT.ipynb
NO_INPUT ?= 1

.PHONY: help build run shell notebook test lint evaluate train predict experiments \
        run-report report readme all clean

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## build the docker image (rerun after touching requirements.txt)
	docker build -t $(IMAGE) .

run: ## run python inside the container: make run FILE="-m delivery.train --cv 3"
	$(RUN) python $(FILE)

shell: ## drop into a shell in the container
	$(RUN_IT) bash

notebook: ## JupyterLab at http://localhost:8899 (no token; PORT=<n> to override)
	@echo "jupyter lab -> http://localhost:$(PORT)"
	docker run --rm -it -p $(PORT):8888 $(MOUNTS) $(IMAGE) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token=''

test: ## run the unit tests (pytest)
	$(RUN) python -m pytest -q tests

lint: ## byte-compile the package and tests
	$(RUN) python -m compileall -q delivery tests

evaluate: ## cross-validate the experiment's candidates, print the leaderboard (no submission)
	$(RUN) python -m delivery.train --exp $(EXP) --models $(MODELS) --cv $(CV) --seed $(SEED) --evaluate-only

train: ## run experiment EXP end-to-end: CV, pick best, refit, write COMPLETED_PREDICTIONS.csv
	$(RUN) python -m delivery.train --exp $(EXP) --models $(MODELS) --cv $(CV) --seed $(SEED)

predict: ## write COMPLETED_PREDICTIONS.csv from the saved model (outputs/model.joblib)
	$(RUN) python -m delivery.train --predict-only

experiments: ## run the experiment ladder behind the report -> outputs/experiments.json (~7 min)
	$(RUN) python -m delivery.experiments --cv $(CV) --seed $(SEED)

run-report: ## re-execute NB (default REPORT.ipynb) in place so its outputs are fresh; fails on any error output
	$(RUN) jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$(NB)"
	$(RUN) python -c "import json,sys; nb=json.load(open('$(NB)')); \
		errs=[o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']; \
		sys.exit(f'{len(errs)} error output(s) in $(NB): '+', '.join(e.get('ename','?') for e in errs) if errs else 0)"

HTML     ?= COMPLETED_ASSIGNMENT.html
report: ## export NB with its saved outputs to ./$(HTML) (code hidden unless NO_INPUT=0); print to PDF from the browser
	$(RUN) jupyter nbconvert --to html $(if $(filter 1,$(NO_INPUT)),--no-input,) \
		--output-dir . --output "$(HTML)" "$(NB)"
	@echo "-> $(HTML)  (open it and use the browser's Print -> Save as PDF)"

readme: ## README.md = the markdown cell(s) of NB, verbatim
	$(RUN) python -c "import json; nb=json.load(open('$(NB)')); \
		md='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown'); \
		open('README.md','w').write(md.strip('\n')+'\n'); print('README.md <-', '$(NB)')"

all: build test train run-report report readme ## build, test, train, re-execute + export the report, refresh README

clean: ## remove generated outputs (keeps metrics.json, experiments.json, figures, the submission) and the image
	rm -rf outputs/model.joblib outputs/best_model.json .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	docker rmi -f $(IMAGE) 2>/dev/null || true
