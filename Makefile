IMAGE_NAME=mlp
SCRIPT ?= process_input.py
CONFIG ?= neuron.conf
INPUT ?= input.txt

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -ti \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME) python $(SCRIPT) --config $(CONFIG) --input $(INPUT)