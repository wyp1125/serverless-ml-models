FROM python:3.7
COPY quick_dnn_model_builder.py /
RUN python3 -m pip install tensorflow
RUN pip3 install argparse pandas numpy sklearn
RUN pip3 install boto3 awscli
RUN mkdir /scratch
CMD ["python3","/quick_dnn_model_builder.py"]

