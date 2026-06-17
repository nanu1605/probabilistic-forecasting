.PHONY: setup lint format test data eda baselines deepar calibration recalibrate figures report reproduce clean

UV := uv
RUN := uv run

setup:  ## install deps + init DVC
	$(UV) sync --extra dev
	[ -d .dvc ] || $(RUN) dvc init

lint:  ## ruff check + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## apply ruff formatting + autofix
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test:  ## pytest with coverage
	$(RUN) pytest --cov=probforecast --cov-report=term-missing

data:  ## download/generate -> preprocess -> split -> DVC track
	$(RUN) python -m probforecast.data.download
	$(RUN) python -m probforecast.data.preprocess
	$(RUN) python -m probforecast.data.split

eda:  ## run EDA notebook
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

baselines:  ## train + evaluate Seasonal Naive, ARIMA, ETS
	$(RUN) python pipelines/run_baselines.py

deepar:  ## train + evaluate DeepAR (5 seeds)
	$(RUN) python pipelines/run_deepar.py

calibration:  ## compute calibration curves + plots for DeepAR
	$(RUN) python -m probforecast.evaluation.calibration

recalibrate:  ## fit recalibration on val, evaluate on test
	$(RUN) python pipelines/run_recalibration.py

figures:  ## regenerate all plots from saved predictions/metrics
	$(RUN) python pipelines/generate_figures.py

report:  ## render report.md -> PDF (if a LaTeX engine exists) else self-contained HTML
	pandoc report/report.md -o report/report.pdf 2>/dev/null && echo "report.pdf built" || \
	pandoc report/report.md -o report/report.html --standalone --embed-resources && echo "report.html built (no LaTeX engine; HTML fallback)" || \
	echo "pandoc unavailable; markdown-only report"

reproduce: data eda baselines deepar calibration recalibrate figures  ## full end-to-end run

clean:  ## remove generated artifacts (not raw data)
	rm -rf metrics/*.json docs/images/*.png mlruns
	find . -type d -name __pycache__ -exec rm -rf {} +
