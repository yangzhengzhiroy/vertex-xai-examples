FROM python:3.11

ARG MODEL_PATH

ENV MODEL_PATH ${MODEL_PATH}
ENV ENV dev

WORKDIR /tf_serving

RUN python -m venv venv
ENV PATH="/tf_serving/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -U -r requirements.txt

COPY app/ ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
