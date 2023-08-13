# deepschool-cvr-bcdetect

The model is trained to detect barcodes on images.
It was train on teh data from the link https://www.kaggle.com/datasets/kniazandrew/ru-goods-barcodes .

Result of the traing are located in `results` folder. 
You can also look on the traing result by the following link: https://app.clear.ml/projects/fa9afe1ea8634490ba638e634915c224/experiments/0bf305d129e04732810e779c2c3c6ee7/output/execution

## instalation
To install model requiremants run `make install`.

## Data preparation 
To prepare data prepare config `data_prep.yml` and run `make prepare_data`.

## Train
To train the model set hyperparameters in config `barcodes.yml` and run `make train`.

## Evaluate on test
To evaluate the model on test data prepare config `barcodes_test.yml` and run `make test`

