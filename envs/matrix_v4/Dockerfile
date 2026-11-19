# Build context: envs/matrix_v4/ only. The host supplies NVIDIA graphics/compute drivers.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04@sha256:e869de8a68e95272e51c4c7071b1d39941aaab707f9bc051a99062a3791448b1
ENV MATRIX_V4_ROOT=/opt/matrix-v4 \
    MATRIX_INBOX=/inputs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
COPY colab_setup.sh requirements-*.txt /opt/matrix-v4-env/
RUN bash /opt/matrix-v4-env/colab_setup.sh system \
    && install -d -m 0700 -o matrixv4 -g matrixv4 /opt/matrix-v4 \
    && rm -rf /var/lib/apt/lists/*
USER matrixv4
WORKDIR /opt/matrix-v4
RUN bash /opt/matrix-v4-env/colab_setup.sh envs
ENTRYPOINT ["bash", "/opt/matrix-v4-env/colab_setup.sh"]
CMD ["all"]
