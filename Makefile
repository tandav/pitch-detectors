IMAGE = tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04

include .env
export

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
	$(IMAGE) \
	pytest -v --cov pitch_detectors

.PHONY: test-no-gpu
test-no-gpu: build
	docker run --rm -t \
	-e PITCH_DETECTORS_GPU=false \
	$(IMAGE) \
	pytest -v --cov pitch_detectors

.PHONY: evaluation
evaluation: build
	docker run --rm -t --gpus all \
	-e PITCH_DETECTORS_GPU=true \
	-e REDIS_URL=$$REDIS_URL \
	-v /home/tandav/Downloads/MIR-1K:/app/MIR-1K \
	$(IMAGE) \
	python pitch_detectors/evaluation.py
