.PHONY: *

install:
	python3.8 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r yolov5-dev/requirements.txt


download_weights:
	mkdir weights
	wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt -O weights/yolov5l6.pt


train:
	venv/bin/python yolov5-dev/train.py \
	  --epochs 100 \
	  --data barcodes.yml \
	  --freeze 10


test:
	venv/bin/python yolov5-dev/val.py --weights weights/best.pt --data barcodes_test.yml
