.PHONY: build
build:
	# https://pythonspeed.com/articles/docker-cache-pip-downloads/
	# DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04 .
	DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04 .

.PHONY: push
push:
	docker push tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04
	docker push tandav/pitch-detectors:latest

.PHONY: test
test: build
	docker run --rm -t --gpus all \
	-e PITCH_DETECTORS_GPU=true \
	tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04 \
	pytest -v --cov pitch_detectors --cov-fail-under 90
	# -e PYTHONDONTWRITEBYTECODE=1 \
	# -v $$PWD/tests:/app/tests \
	# -v $$PWD/pitch_detectors:/app/pitch_detectors \

.PHONY: test-no-gpu
test-no-gpu: build
	docker run --rm -t \
	-e PITCH_DETECTORS_GPU=false \
	tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04 \
	pytest -v --cov pitch_detectors --cov-fail-under 90
	# -e PYTHONDONTWRITEBYTECODE=1 \
	# -v $$PWD/tests:/app/tests \
	# -v $$PWD/pitch_detectors:/app/pitch_detectors \

.PHONY: evaluation
evaluation: build
	docker run --rm -t --gpus all \
	-e PYTHONDONTWRITEBYTECODE=1 \
	-e PITCH_DETECTORS_GPU=true \
	-v /home/tandav/Downloads/MIR-1K:/app/MIR-1K \
	tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04 \
	python pitch_detectors/evaluation.py
	# -v $$PWD/pitch_detectors:/app/pitch_detectors \
