import cv2
import numpy as np
from ultralytics import YOLO
from dtaidistance import dtw
import streamlit as st
import tempfile
import os

# YOLO 모델 불러오기
model = YOLO('yolov8m-pose.pt')  # YOLOv8 포즈 모델 경로

# keypoints 좌표를 [0, 1]로 정규화하는 함수
def normalize_keypoints(keypoints, frame_width, frame_height):
    normalized_keypoints = np.copy(keypoints)
    for i in range(0, len(keypoints), 2):
        normalized_keypoints[i] = keypoints[i] / frame_width  # x 좌표
        normalized_keypoints[i + 1] = keypoints[i + 1] / frame_height  # y 좌표
    return normalized_keypoints

# Keypoints 간 상대적 거리 계산
def calculate_relative_distances(keypoints):
    num_keypoints = len(keypoints) // 2
    relative_distances = []
    
    # 각 keypoint 사이의 거리를 계산 (유클리드 거리)
    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x1, y1 = keypoints[2 * i], keypoints[2 * i + 1]
            x2, y2 = keypoints[2 * j], keypoints[2 * j + 1]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            relative_distances.append(distance)
    
    return np.array(relative_distances)

# Keypoints 시퀀스를 스무딩하는 함수
def smooth_keypoints(sequence, window_size=3):
    smoothed_sequence = []
    for i in range(sequence.shape[1]):  # 각 keypoint에 대해
        smoothed = np.convolve(sequence[:, i], np.ones(window_size)/window_size, mode='valid')
        smoothed_sequence.append(smoothed)
    smoothed_sequence = np.array(smoothed_sequence).T
    return smoothed_sequence

# 비디오에서 keypoints 추출하는 함수 (1초당 1개의 프레임만 분석)
def extract_keypoints(video_path, model):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS 가져오기
    frame_interval = fps  # 1초에 한 프레임을 가져오기 위해 interval을 FPS로 설정
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:  # 1초당 1 프레임 추출
            # YOLO로 프레임에서 포즈 추출
            results = model(frame)

            for result in results:
                if result.keypoints is not None:
                    # Keypoints 추출 (xy 좌표만 사용)
                    keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
                    xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                    
                    # 좌표 정규화
                    normalized_keypoints = normalize_keypoints(xy_keypoints, frame_width, frame_height)

                    # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
                    if len(normalized_keypoints) < max_keypoints:
                        padded_keypoints = np.zeros(max_keypoints)
                        padded_keypoints[:len(normalized_keypoints)] = normalized_keypoints
                        keypoints_sequence.append(padded_keypoints)
                    else:
                        keypoints_sequence.append(normalized_keypoints[:max_keypoints])
        
        frame_count += 1

    cap.release()

    # keypoints_sequence를 배열로 변환
    keypoints_sequence = np.array(keypoints_sequence)

    # keypoints 시퀀스에 스무딩 적용
    if len(keypoints_sequence) > 3:  # 스무딩 적용 가능한 최소 길이 확인
        keypoints_sequence = smooth_keypoints(keypoints_sequence)

    return keypoints_sequence

# 두 시퀀스 간의 DTW 거리 계산 (상대적 거리 기반)
def calculate_dtw_distance(seq1, seq2):
    # 각 시퀀스의 상대적 거리 계산
    seq1_relative = np.array([calculate_relative_distances(frame) for frame in seq1])
    seq2_relative = np.array([calculate_relative_distances(frame) for frame in seq2])

    # 시퀀스 길이 맞추기
    min_len = min(len(seq1_relative), len(seq2_relative))
    seq1_flat = seq1_relative[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
    seq2_flat = seq2_relative[:min_len]

    # 프레임별로 DTW 거리 계산
    distances = []
    for i in range(min_len):
        if np.any(np.isnan(seq1_flat[i])) or np.any(np.isnan(seq2_flat[i])):
            distances.append(np.inf)  # NaN이 있는 경우, 무한대로 처리
        else:
            distance = dtw.distance(seq1_flat[i], seq2_flat[i])
            distances.append(distance)

    return np.mean(distances)

# 두 영상의 유사도를 계산하는 메인 함수
def compare_videos(video_path1, video_path2, model):
    st.info('첫 번째 비디오의 키포인트를 추출 중입니다...')
    keypoints_seq1 = extract_keypoints(video_path1, model)
    
    st.info('두 번째 비디오의 키포인트를 추출 중입니다...')
    keypoints_seq2 = extract_keypoints(video_path2, model)

    st.info('DTW 거리를 계산 중입니다...')
    dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
    st.success(f"두 비디오 간의 DTW 거리: {dtw_distance}")

    return dtw_distance

# Streamlit 앱 UI
st.title('비디오 포즈 유사도 비교 (DTW)')

# 파일 업로드 위젯
video_file_1 = st.file_uploader('첫 번째 비디오 파일을 업로드하세요.', type=['mp4', 'mov', 'avi'])
video_file_2 = st.file_uploader('두 번째 비디오 파일을 업로드하세요.', type=['mp4', 'mov', 'avi'])

# 두 개의 파일이 업로드되었을 때 실행
if video_file_1 is not None and video_file_2 is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp1:
        temp1.write(video_file_1.read())
        video_path1 = temp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp2:
        temp2.write(video_file_2.read())
        video_path2 = temp2.name

    if st.button('비디오 유사도 비교 시작'):
        dtw_distance = compare_videos(video_path1, video_path2, model)
        st.write(f"두 비디오 간의 유사도 (DTW 거리): {dtw_distance}")

    # 임시 파일 삭제
    os.remove(video_path1)
    os.remove(video_path2)
