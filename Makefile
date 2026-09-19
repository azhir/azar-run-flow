IMAGE=azar-run-flow

.PHONY: build shell test mlflow clean

build:
	docker build -t $(IMAGE) .

shell:
	docker run --rm -it \
		-p 5000:5000 \
		-v "$(PWD)":/app \
		$(IMAGE)

test:
	docker run --rm \
		-v "$(PWD)":/app \
		$(IMAGE) \
		python examples/basic.py

mlflow:
	docker run --rm -it \
		-p 5000:5000 \
		-v "$(PWD)":/app \
		$(IMAGE) \
		mlflow ui --host 0.0.0.0 --port 5000

clean:
	docker image rm $(IMAGE)