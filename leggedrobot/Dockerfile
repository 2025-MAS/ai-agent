# Python base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 업데이트 및 필수 패키지 설치
# OpenCV와 비디오 처리에 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY workspace/ /workspace
# 포트 노출
# 5005: UDP 비디오 스트림 수신
# 7860: Gradio 웹 인터페이스
EXPOSE 5005/udp 7860/tcp

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 애플리케이션 실행
CMD ["tail", "-f", "/dev/null"]
