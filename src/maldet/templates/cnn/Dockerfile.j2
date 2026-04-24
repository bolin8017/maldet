FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml maldet.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

ARG MALDET_NAME
ARG MALDET_VERSION
ARG MALDET_FRAMEWORK
ARG MALDET_MANIFEST_B64
ARG GIT_COMMIT

LABEL org.opencontainers.image.title="${MALDET_NAME}"
LABEL org.opencontainers.image.version="${MALDET_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL io.maldet.manifest.schema="1"
LABEL io.maldet.manifest="${MALDET_MANIFEST_B64}"
LABEL io.maldet.framework="${MALDET_FRAMEWORK}"

ENTRYPOINT ["maldet"]
