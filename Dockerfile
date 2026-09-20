FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch is smaller and avoids multi-hundred-MB CUDA wheels timing out on slow mirrors.
RUN pip install --upgrade pip \
    && pip install --retries 10 --timeout 1000 \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --retries 10 --timeout 1000 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r requirements.txt

COPY scripts/ ./scripts/

CMD ["python", "-m", "scripts.traffic_analysis"]
