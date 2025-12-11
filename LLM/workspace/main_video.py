# ========== Vision Agent 설정 ==========
FRAME_INTERVAL_SECONDS = 2  # 프레임 추출 간격 (초) - 이 값을 조정하여 다양한 결과를 확인할 수 있습니다
VIDEO_PATH = "input/sample.mp4"  # 분석할 영상 파일 경로
# ========================================

from services.agent_runner import ask_vision_interactive
import os
from dotenv import load_dotenv

load_dotenv()

# Interactive Vision Agent - 대화형 영상 분석
ask_vision_interactive(
    video_path=VIDEO_PATH,
    frame_interval=FRAME_INTERVAL_SECONDS
)
