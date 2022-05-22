FROM python:3.7
COPY quick_ml_model_builder.py /
RUN pip3 install argparse pandas numpy sklearn
RUN pip3 install boto3 awscli
RUN mkdir /scratch
CMD ["python3","/quick_ml_model_builder.py"]

