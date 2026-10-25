FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY qtensor ./qtensor
COPY simulator ./simulator
RUN pip install --no-cache-dir .
ENTRYPOINT ["qtensor"]
CMD ["bench", "--quick"]
