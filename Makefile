.PHONY: build
build:
	# https://pythonspeed.com/articles/docker-cache-pip-downloads/
	# DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04 .
	DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors:cuda12.0.1-cudnn8-devel-ubuntu22.04 .

.PHONY: push
push:
	# docker push tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04
	docker push tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04
	docker push tandav/pitch-detectors:latest

.PHONY: test-tensorflow-gpu
test-tensorflow-gpu: build
	docker run --rm -it --gpus all tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04 \
	python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

.PHONY: test
test: build
	docker run --rm -it --gpus all \
	-v $$PWD/pitch_detectors:/app/pitch_detectors \
	-v $$PWD/tests:/app/tests \
	tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04 \
	pytest -v
