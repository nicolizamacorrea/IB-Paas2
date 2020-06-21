# Move to /dash

FROM python:3.8

#COPY /exam.csv /app/exam.csv

WORKDIR /app

COPY .  /app

RUN pip install --trusted-host pypi.python.org -r app/requirements.txt

EXPOSE 8080

CMD ["python", "app/main.py"]

#ENTRYPOINT ["gunicorn","--bind=0.0.0.0:8050","main:server"]