IMAGE = tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04

.PHONY: build
build:
	DOCKER_BUILDKIT=1 docker build --progress=plain -t $(IMAGE) .

.PHONY: push
push:
	docker push $(IMAGE)
	docker push tandav/pitch-detectors:latest

.PHONY: test
test: build
	docker run --rm -t --gpus all \
	-e PITCH_DETECTORS_GPU=true \
	-e PITCH_DETECTORS_GPU_MEMORY_LIMIT=true \
	-v /home/tandav/docs/bhairava/libmv/data/fcnf0++.pt:/fcnf0++.pt:ro \
	-v /home/tandav/docs/bhairava/libmv/data/spice_model:/spice_model:ro \
	$(IMAGE) \
	pytest -x -v --cov pitch_detectors

.PHONY: test-no-gpu
test-no-gpu: build
	docker run --rm -t \
	-e PITCH_DETECTORS_GPU=false \
	-v /home/tandav/docs/bhairava/libmv/data/fcnf0++.pt:/fcnf0++.pt:ro \
	-v /home/tandav/docs/bhairava/libmv/data/spice_model:/spice_model:ro \
	$(IMAGE) \
	pytest -v --cov pitch_detectors

.PHONY: evaluation
evaluation: build
	eval "$$(cat .env)"; \
	docker run --rm -t --gpus all \
	-e PITCH_DETECTORS_GPU=true \
	-e REDIS_URL=$$REDIS_URL \
	-v /media/tandav/sg8tb1/downloads-archive/f0-datasets:/app/f0-datasets:ro \
	$(IMAGE) \
	python -m pitch_detectors.evaluation

.PHONY: table
table:
	eval "$$(cat .env)"; \
	REDIS_URL=$$REDIS_URL \
	python -m pitch_detectors.evaluation.table

.PHONY: bumpver
bumpver:
	# usage: make bumpver PART=minor
	bumpver update --no-fetch --$(PART)
