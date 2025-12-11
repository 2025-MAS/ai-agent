# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np
import socket
import cv2
import time

print("=== 프로그램 시작 ===")
print("모든 라이브러리 import 완료")

# UDP 수신 소켓 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))
print("UDP 서버가 0.0.0.0:5005에서 시작되었습니다.")

# 전역 변수
last_frame = None

# UDP 프레임 수신 함수
def receive_udp_frame():
    global last_frame
    sock.settimeout(0.1)
    
    try:
        data, addr = sock.recvfrom(65536)
        np_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            print(f"프레임 수신 성공 from {addr}")
            last_frame = frame.copy()
            return frame
    except socket.timeout:
        pass
    except Exception as e:
        print(f"프레임 수신 오류: {e}")
    
    return None

# 비디오 스트림 표시 함수
def video_stream():
    global last_frame
    
    while True:
        try:
            # 최신 프레임 수신
            current_frame = receive_udp_frame()
            
            # 화면 표시용 프레임 선택
            if current_frame is not None:
                display_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            elif last_frame is not None:
                display_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            else:
                # 표시할 프레임이 없으면 검은 화면 생성
                display_frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            
            yield display_frame_rgb
            time.sleep(0.03)  # 약 30 FPS
            
        except Exception as e:
            print(f"스트리밍 오류: {e}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            yield error_frame

# Gradio 인터페이스 구성
print("Gradio 인터페이스 생성 중...")

interface = gr.Interface(
    fn=video_stream,
    inputs=[],
    outputs=[
        gr.Image(label="UDP 비디오 스트림", width=1280, height=720)
    ],
    title="UDP 비디오 스트리밍",
    description="실시간 UDP 비디오 스트림 수신 및 표시",
    live=True
)

print("Gradio 인터페이스 시작...")
interface.launch(server_name="0.0.0.0", share=True)
