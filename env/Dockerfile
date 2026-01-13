FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -r env/requirements.txt

# 기본 실행은 inference (운영진이 inference.py 실행할 확률 높음)
CMD ["python", "inference.py"]
