IMAGE=azar-run-flow

.PHONY: build shell test mlflow conda-env clean

build:
	docker build -t $(IMAGE) .

shell:
	docker run --rm -it \
		-v "$(PWD)":/app \
		$(IMAGE)

test:
	docker run --rm \
		-v "$(PWD)":/app \
		$(IMAGE) \
		python examples/basic.py

mlflow:
	docker run --rm -it \
		-p 5001:5000 \
		-v "$(PWD)":/app \
		$(IMAGE) \
		mlflow ui \
			--host 0.0.0.0 \
			--port 5000 \
			--backend-store-uri sqlite:///mlflow.db

conda-env:
	conda env create -f environment.yml

clean:
	-docker image rm $(IMAGE)
	rm -f mlflow.db
	rm -rf mlartifacts mlruns