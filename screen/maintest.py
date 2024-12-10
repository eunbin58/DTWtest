import os
import cv2
import numpy as np
import tempfile
import streamlit as st
from ultralytics import YOLO
from dtaidistance import dtw
from openai import OpenAI  # OpenAI 임포트

# 페이지 아이콘 설정
icon_path = os.path.join(os.path.dirname(__file__), '../src/images/logo.jpg')

st.set_page_config(
    page_title="healthy_homebody",
    page_icon=icon_path,
    layout="wide"
)

# CSS 파일 로드 함수
def load_css(file_path):
    """CSS 파일 내용을 읽어 반환"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSS 파일을 찾을 수 없습니다: {file_path}")

# CSS 파일 경로
css_path = os.path.join(os.path.dirname(__file__), '../src/styles.css')

# CSS 로드 및 적용
st.markdown(f"<style>{load_css(css_path)}</style>", unsafe_allow_html=True)

# 세션 상태 초기화
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "main"

if 'selected_action' not in st.session_state:
    st.session_state.selected_action = None
    
if 'uploaded_video_path' not in st.session_state:
    st.session_state.uploaded_video_path = None

if 'description_video_path' not in st.session_state:
    st.session_state.description_video_path = None

# YOLO 모델 로드
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8m-pose.pt', verbose=False)

# 메인 페이지
def main_page():
    st.markdown('<div class="title-style">Healthy Homebody</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-style section">필라테스 동작을 선택하세요</div>', unsafe_allow_html=True)

    actions = [
        "로우 런지(Low Lunge)",
        "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)",
        "선 활 자세(Standing Split)",
        "런지 사이트 스트레칭(Lunging Side Stretch)"
    ]

    col1, col2 = st.columns(2)

    with col1:
        for action in actions[:2]:
            if st.button(action):
                st.session_state.selected_page = "page1"
                st.session_state.selected_action = action

    with col2:
        for action in actions[2:]:
            if st.button(action):
                st.session_state.selected_page = "page1"
                st.session_state.selected_action = action

    st.markdown(
        """
        <div class="description-style section">
        최근 비만율 증가와 함께 <span class="highlight">건강 관리</span>는 더욱 중요한 문제로 대두되고 있습니다. 
        특히 재택근무자, 집순이·집돌이, 은둔형 외톨이들은 신체 활동이 부족하여 건강이 악화될 위험이 큽니다. 
        이를 해결하기 위해 저희 서비스는 실내에서 손쉽게 할 수 있는 <span class="highlight">스트레칭 및 필라테스 동작</span>을 
        제공하여 체력 증진과 비만 예방을 목표로 하고 있습니다.
        <br><br>
        저희는 <span class="highlight">YOLOv8 포즈 추정 모델</span>을 활용하여 사용자의 운동 동작을 분석하고, 
        올바른 자세를 유지할 수 있도록<span class="highlight">정확한 피드백</span>을 제공합니다. 
        편리한 웹 기반 플랫폼을 통해 사용자는 언제 어디서나 스트레칭을 수행하고 자신의 건강 상태를 관리할 수 있습니다.
        </div>
        """, 
        unsafe_allow_html=True
    )

# 페이지 1: 동작 설명 페이지
def page1():
    action_info = {
        "로우 런지(Low Lunge)": {
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video1.mp4'),
            "description": [
                ("자세 설명", [
                    "제자리에서 힘을 기르는 우아한 동작이에요.",
                    "마치 춤을 추듯 부드럽게 움직이면서 몸의 균형을 잡아보세요."
                ]),
                ("효과", [
                    "고관절의 유연성을 부드럽게 확장하고 근육의 탄력성을 높여줘요."
                ]),
                ("주의사항", [
                    "자신의 신체 한계를 존중하며 천천히 진행하세요."
                ]),
                ("실행 방법", [
                    "1. 안정된 자세로 시작해 한 발을 앞으로 크게 내딛어요."
                ])
            ]
        },
        "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)": {
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video6.mp4'),
            "description": [
                ("자세 설명", [
                    "이 자세는 마치 나무가 바람에 살랑이듯 유연하게 몸을 움직이는 거예요."
                ]),
                ("효과", [
                    "척추와 근육의 깊은 이완으로 전체적인 유연성을 극대화해요."
                ]),
                ("주의사항", [
                    "허리나 관절에 통증이 있다면 전문 트레이너와 상담해주세요."
                ])
            ]
        },
        "선 활 자세(Standing Split)": {
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video3.mp4'),
            "description": [
                ("자세 설명", [
                    "한 발로 서서 다른 다리를 하늘 높이 들어올리는 멋진 동작입니다."
                ]),
                ("효과", [
                    "균형 감각을 향상시키고 하체 유연성을 강화합니다."
                ]),
                ("주의사항", [
                    "처음에는 벽을 지지대 삼아 연습하세요."
                ])
            ]
        },
        "런지 사이트 스트레칭(Lunging Side Stretch)": {
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video4.mp4'),
            "description": [
                ("자세 설명", [
                    "런지 자세에서 상체를 옆으로 기울이는 스트레칭 동작입니다."
                ]),
                ("효과", [
                    "코어 근육을 강화하고 옆구리 근육을 이완시킵니다."
                ]),
                ("주의사항", [
                    "무릎과 허리에 부담이 가지 않도록 주의하세요."
                ])
            ]
        }
    }

    selected_action = st.session_state.selected_action
    video_path = action_info[selected_action]
    st.session_state.description_video_path = video_path  # 설명 비디오 경로 저장
    st.markdown(f'<h2 class="sub-title-style">{selected_action}</h2>', unsafe_allow_html=True)

    video_path = action_info[selected_action]["video_path"]
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("목록으로"):
            st.session_state.selected_page = "main"

    with col2:
        if st.button("다음"):
            st.session_state.selected_page = "page2"

    description = action_info[selected_action]["description"]
    for section_title, section_content in description:
        st.markdown(f'<h3>{section_title}</h3>', unsafe_allow_html=True)
        for line in section_content:
            st.markdown(f'<li class="animated-section">{line}</li>', unsafe_allow_html=True)

# 페이지 2: 사용자 비디오 업로드 및 비교 페이지
def page2():
    

    # OpenAI API 키 설정 (환경 변수에서 불러오기)
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

    # GPT-4 피드백 생성 함수
    def get_advice_based_on_similarity(dtw_distance, action_name):
        user_message = (
            f"사용자와 '{action_name}' 동작을 비교한 결과, DTW 거리 값은 {dtw_distance}입니다.\n"
            "이 값에 기반하여 피드백을 제공해주세요:\n"
            "- 유사도가 낮을 경우: 자세를 교정하기 위한 구체적인 피드백 제공.\n"
            "- 유사도가 높을 경우: 칭찬과 간단한 개선점을 제안.\n"
        )
        messages = [
            {"role": "system", "content": "당신은 피트니스 전문가입니다."},
            {"role": "user", "content": user_message},
        ]
        try:
            # OpenAI API 호출
            result = client.chat.completions.create(
                model="gpt-4o",  # OpenAI API 모델명
                messages=messages,
                temperature=0.7
            )
            advice = result.choices[0].message.content  # GPT-4의 응답 추출
            return advice
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            return "피드백을 생성하는 동안 문제가 발생했습니다. 다시 시도해주세요."
        
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
        
            # 피드백 생성 요청
        st.info('피드백을 생성 중입니다...')
        action_name = "동작"  # 예시: 동작명을 지정하세요.
        advice = get_advice_based_on_similarity(dtw_distance, action_name)
        
        st.info('피드백:')
        st.write(advice)

        return dtw_distance



    # Streamlit 앱 UI
    st.title('비디오 포즈 유사도 비교 (DTW)')

    # 첫 번째 비디오는 고정된 경로 사용
    video_path1 = os.path.join(os.path.dirname(__file__), '../src/mp4/video6.mp4')

    # 파일 업로드 위젯 (두 번째 비디오)
    video_file_2 = st.file_uploader('두 번째 비디오 파일을 업로드하세요.', type=['mp4', 'mov', 'avi'])

    # 두 번째 파일이 업로드되었을 때 실행
    if video_file_2 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp2:
            temp2.write(video_file_2.read())
            video_path2 = temp2.name

        if st.button('비디오 유사도 비교 시작'):
            dtw_distance = compare_videos(video_path1, video_path2, model)
            st.write(f"두 비디오 간의 유사도 (DTW 거리): {dtw_distance}")

        # 임시 파일 삭제
        os.remove(video_path2)

# 페이지 전환 및 실행
if st.session_state.selected_page == "main":
    main_page()
elif st.session_state.selected_page == "page1":
    page1()
elif st.session_state.selected_page == "page2":
    page2()