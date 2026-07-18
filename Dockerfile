# Build Stage

ARG PYTHON_VERSION=3.13
ARG DISTROLESS_PY=python3-debian13
ARG FUNCTION_DIR="/app/function/"

FROM python:${PYTHON_VERSION}-slim AS build

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    unzip \
    libcurl4-openssl-dev

ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}

COPY ./src/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt --target ${FUNCTION_DIR}

# Deploy Stage

FROM gcr.io/distroless/${DISTROLESS_PY} AS deploy

ARG FUNCTION_DIR

WORKDIR ${FUNCTION_DIR}
ENV ROOT_DIR=${FUNCTION_DIR}

COPY --from=build ${FUNCTION_DIR} ${FUNCTION_DIR}
COPY ./src ./

ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]