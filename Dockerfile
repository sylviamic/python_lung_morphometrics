FROM python:latest

RUN \
	git clone https://github.com/sylviamic/python_lung_morphometrics && \
	cd python_lung_morphometrics && \
	make install && \
	make clean

ENTRYPOINT ["python-lung-morphometrics"]