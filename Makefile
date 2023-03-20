.PHONY: build
build:
	# https://pythonspeed.com/articles/docker-cache-pip-downloads/
	# DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04 .
	DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04 .

.PHONY: push
push:
	# docker push tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04
	docker push tandav/pitch-detectors:cuda11.7.0-cudnn8-devel-ubuntu22.04
	docker push tandav/pitch-detectors:latest

.PHONY: test
test: build
	docker run --rm -it --gpus all \
	-e PITCH_DETECTORS_ERROR_GPU_NOT_AVAILABLE=true \
	-v $$PWD/pitch_detectors:/app/pitch_detectors \
	-v $$PWD/tests:/app/tests \
	tandav/pitch-detectors:11.8.0-cudnn8-devel-ubuntu22.04 \
	pytest -v
