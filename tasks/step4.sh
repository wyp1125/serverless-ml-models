#!/bin/bash
cd /scratch
aws s3 cp s3://bdx-demo-docker/bank_data.codedx bank_data.codedx
aws s3 cp s3://bdx-demo-docker/bank_data.codedy bank_data.codedy
python3 /quick_dnn_model_builder.py -i bank_data -m testmodel >dnn.metrices.txt
aws s3 cp dnn.metrices.txt s3://bdx-demo-docker/dnn.metrices.txt
