.PHONY: build
build:
	# https://pythonspeed.com/articles/docker-cache-pip-downloads/
	DOCKER_BUILDKIT=1 docker build --progress=plain -t tandav/pitch-detectors .

.PHONY: push
push:
	docker push tandav/pitch-detectors

.PHONY: test-tensorflow-gpu
test-tensorflow-gpu: build
	docker run --rm -it --gpus all tandav/pitch-detectors \
	python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

.PHONY: test
test: build
	docker run --rm -it --gpus all \
	-v $$PWD/pitch_detectors:/app/pitch_detectors \
	-v $$PWD/tests:/app/tests \
	tandav/pitch-detectors \
	pytest -v
