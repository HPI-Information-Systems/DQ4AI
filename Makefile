# Setting PYTHONPATH to current directory
export PYTHONPATH=${PWD}
.DEFAULT_GOAL := help


.PHONY: jupyter-start
jupyter-start: ## Start jupyter server
	@echo "+ $@"
	@{ nohup jupyter notebook --no-browser --port=8888 & echo $$! > jupyter.pid; }


.PHONY: jupyter-stop
jupyter-stop: ## Stop jupyter server
	@echo "+ $@"
	@kill -9 `cat jupyter.pid`
	@rm jupyter.pid
	@rm nohup.out


.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
