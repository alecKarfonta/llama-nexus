# syntax=docker/dockerfile:1
# [2026-08-24 | INFO | Dockerfile.ninfer | Builds Neroued/ninfer (RTX 5090 sm_120a) with curl for healthchecks
# Upstream: https://github.com/Neroued/ninfer (Apache-2.0)
#
# Requires a host driver reporting CUDA 13.1+ (upstream nvcc gate).
# CUDA_VERSION is overridable if a future pin changes.

ARG CUDA_VERSION=13.1.2

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# [2026-09-28 | INFO | Dockerfile.ninfer | Pin the fetch ref to the same commit as NINFER_SHA
ARG NINFER_REF=3d9fda2229eb28553768ce54ed86a0deeb57b733
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# [2026-08-31 | INFO | Dockerfile.ninfer | SHA assert so a moved default ref cannot silently ship
ARG NINFER_SHA=3d9fda2
RUN git init \
    && git remote add origin https://github.com/Neroued/ninfer.git \
    && git fetch --depth 1 origin "${NINFER_REF}" \
    && git checkout --detach FETCH_HEAD \
    && echo "ninfer HEAD=$(git rev-parse --short=7 HEAD) want=${NINFER_SHA}" \
    && test "$(git rev-parse --short=7 HEAD)" = "${NINFER_SHA}"

# [2026-08-30 | INFO | Dockerfile.ninfer | Optional adaptive MTP window (shrink to 2
# when rolling accept < 40%). Off by default; enable with --build-arg NINFER_APPLY_ADAPTIVE_DRAFT=1.
ARG NINFER_APPLY_ADAPTIVE_DRAFT=0
COPY patches/adaptive_mtp_draft.patch /tmp/adaptive_mtp_draft.patch
RUN if [ "${NINFER_APPLY_ADAPTIVE_DRAFT}" = "1" ]; then \
        git apply /tmp/adaptive_mtp_draft.patch; \
    fi

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
