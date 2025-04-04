import time
import cv2
import mediapipe as mp
from multiprocessing import Process, Queue
from indy_utils import indydcp_client as client

# 로봇 연결 정보
robot_ip = "192.168.0.5"  # Robot (Indy) IP 주소
name = "NRMK-Indy7"       # 로봇 이름 (Indy7)
indy = client.IndyDCPClient(robot_ip, name)  # IndyDCPClient 객체 생성

flag = False  # 전체 프로그램 종료 플래그

# 그리드를 나누는 라인 수 (x, y, z에 대한 분할)
x_line = 30
y_line = 20
z_line = 20

def open_cam(q1, q2, q3, q6):
    """
    두 대의 카메라(상단, 측면)로부터 영상을 수집하고,
    MediaPipe를 통해 손 랜드마크(좌표, 제스처)를 추출 후
    필요한 정보(x,y,z, 제스처 등)를 큐에 전달하는 함수(프로세스).
    """
    # 카메라 해상도 설정
    w, h = 1200, 900    # 첫 번째 카메라(상단)용
    w2, h2 = 600, 900   # 두 번째 카메라(측면)용

    # 첫 번째 카메라 연결
    cap = cv2.VideoCapture(0)
    cap.set(3, w)
    cap.set(4, h)
    print("bp1")  # 디버깅용 출력

    # 두 번째 카메라 연결
    cap_z = cv2.VideoCapture(1)
    cap_z.set(3, w2)
    cap_z.set(4, h2)
    print("bp0")  # 디버깅용 출력

    # MediaPipe 설정 (Hands)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 상단 카메라용 Hands 객체 (x, y 좌표)
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    print("bp2")

    # 측면 카메라용 Hands 객체 (z 좌표)
    mp_hands_z = mp.solutions.hands
    hands_z = mp_hands_z.Hands(max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
    print("bp3")

    sw_list = [0]  # (사용 여부 미미함: sw_flag 관련 임시 리스트)

    # 카메라가 열려 있는 동안 계속 루프
    while cap.isOpened() and cap_z.isOpened():
        success, img = cap.read()    # 첫 번째 카메라 프레임
        _, img_z = cap_z.read()      # 두 번째 카메라 프레임

        if not success:
            break  # 첫 번째 카메라에서 영상을 못 받으면 루프 탈출

        # --- 상단 카메라( x, y 좌표 ) 처리 ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV → RGB 변환
        result = hands.process(img)                 # MediaPipe Hands 처리
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 다시 BGR로 변환
        img = cv2.resize(img, dsize=(w, h))

        # --- 측면 카메라( z 좌표 ) 처리 ---
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)
        result_z = hands_z.process(img_z)
        img_z = cv2.cvtColor(img_z, cv2.COLOR_RGB2BGR)
        img_z = cv2.resize(img_z, dsize=(w2, h2))

        # --- 상단 카메라에서 손 랜드마크가 검출된 경우 ---
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                center = res.landmark[9]  # 중지 중심점(landmark[9])
                x1 = float(center.x)
                y1 = float(center.y)

                # q1이 비어 있으면 [x, y] 좌표 데이터 전송
                if q1.qsize() == 0:
                    q1.put([x1, y1])

                # 엄지 vs 새끼손가락 x좌표 비교(s)
                thumb = res.landmark[4]
                little_finger = res.landmark[20]
                s = thumb.x - little_finger.x
                if s < 0:
                    s_flag = True   # x,y 평면 모드
                else:
                    s_flag = False  # z 축 모드

                # q3가 비어 있으면 s_flag 전송
                if q3.qsize() == 0:
                    q3.put(s_flag)

                # 손가락 길이(인덱스, 미들, 링, 리틀) vs 손목(landmark[0]) 비교
                index = res.landmark[8]
                middle = res.landmark[12]
                ring = res.landmark[16]
                little = res.landmark[20]
                zero = res.landmark[0]

                d1 = abs(index.y - zero.y)
                d2 = abs(middle.y - zero.y)
                d3 = abs(ring.y - zero.y)
                d4 = abs(little.y - zero.y)

                # 네 손가락이 다 어느 정도 펴져 있으면 sw_flag = True
                if d1 <= 0.23 and d2 <= 0.23 and d3 <= 0.23 and d4 <= 0.23:
                    sw_flag = True
                else:
                    sw_flag = False

                # q6가 비어 있으면 sw_flag (그리퍼 열림/닫힘 여부) 전송
                if q6.qsize() == 0:
                    q6.put(sw_flag)

                # 손 랜드마크 시각화(선/점) + x,y 좌표 표시
                mp_drawing.draw_landmarks(
                    img, res, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                cv2.putText(img, f'x : {x1:.2f}',
                            org=(int(res.landmark[9].x * img.shape[1]),
                                 int(res.landmark[9].y * img.shape[0] + 10)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.putText(img, f'y : {y1:.2f}',
                            org=(int(res.landmark[9].x * img.shape[1]),
                                 int(res.landmark[9].y * img.shape[0] + 40)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.circle(img,
                           (int(res.landmark[9].x * img.shape[1]),
                            int(res.landmark[9].y * img.shape[0])),
                           8, (0, 0, 255), thickness=cv2.FILLED)

            # --- 측면 카메라에서 손 랜드마크가 검출된 경우 ---
            if result_z.multi_hand_landmarks is not None:
                for res_z in result_z.multi_hand_landmarks:
                    center_z = res_z.landmark[9]
                    z1 = float(center_z.y)

                    # q2가 비어 있으면 z 좌표 정보 전송
                    if q2.qsize() == 0:
                        q2.put([z1])

                    # z 좌표 표시
                    cv2.putText(img_z, f'z : {z1:.2f}',
                                org=(int(res_z.landmark[9].x * img.shape[1]),
                                     int(res_z.landmark[9].y * img.shape[0] + 30)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 255, 255), thickness=2)

        # --- x_line, y_line, z_line으로 나눈 그리드 표시 ---
        #   (화면에 격자 선을 그려주어 위치 가늠)
        for i in range(1, x_line + 1):
            cv2.line(img, pt1=(int(w / x_line) * i, 0),
                     pt2=(int(w / x_line) * i, int(h)),
                     color=(255, 255, 255), thickness=1)
        for j in range(1, y_line + 1):
            cv2.line(img, pt1=(0, int(h / y_line) * j),
                     pt2=(int(w), int(h / y_line) * j),
                     color=(255, 255, 255), thickness=1)
        for k in range(1, z_line + 1):
            cv2.line(img_z, pt1=(0, int(h / z_line) * k),
                     pt2=(int(w), int(h / z_line) * k),
                     color=(255, 255, 255), thickness=1)

        # --- 화면 출력 ---
        cv2.imshow('img', img)
        cv2.imshow('img_z', img_z)

        # 두 창을 서로 다른 위치로 이동
        cv2.moveWindow('img', 0, 0)
        cv2.moveWindow('img_z', 1200, 0)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) == ord('q'):
            global flag
            flag = True
            indy.disconnect()  # 로봇 연결 해제
            break

    # 종료 시 자원 해제
    cap.release()
    cap_z.release()
    cv2.destroyAllWindows()


def x_divide(x, x_line):
    """
    0~1 사이의 x값을 (f = 1/x_line) 단위로 나눠서
    실제 로봇 이동 범위에 매핑하기 위한 스케일 함수.
    """
    f = 1 / x_line
    i = 0
    # x가 (f/2)보다 클 때마다 i를 증가 -> 그리드별 위치 결정
    while x >= (f / 2):
        x = x - f
        i += 1
    # 최종 x를 실제 로봇 좌표에 반영할 때 0.6 배 scaling
    x = f * i * 0.6
    return x

def y_divide(y, y_line):
    """
    0~1 사이의 y값을 (1/y_line) 단위로 나눠서
    로봇 이동 범위에 매핑하기 위한 스케일 함수.
    """
    f = 1 / y_line
    i = 0
    while y >= (f / 2):
        y = y - f
        i += 1
    # 0.4 배로 적용 (시연에 맞춘 임의 스케일)
    y = f * i * 0.4
    return y

def z_divide(z, z_line):
    """
    0~1 사이의 z값을 (1/z_line) 단위로 나눠서
    로봇의 수직 위치(z 축)에 매핑하기 위한 스케일 함수.
    """
    f = 1 / z_line
    i = 0
    while z >= (f / 2):
        z = z - f
        i += 1
    # 0.4 배로 적용
    z = f * i * 0.4
    return z

def gripper_open():
    """
    그리퍼를 여는 동작.
    로봇 EndTool DO(디지털 출력)를 1로 설정.
    """
    indy.set_endtool_do(0, 1)
    print("Gripper Open")
    time.sleep(1)

def gripper_close():
    """
    그리퍼를 닫는 동작.
    일정 높이로 이동 후 그리퍼 닫고,
    다른 위치로 이동 후 다시 열기.
    """
    # 속도 레벨 설정 (처음엔 느리게)
    indy.set_task_vel_level(1)
    indy.task_move_by([0, 0, 0.04, 0, 0, 0])  # 조금 아래로 이동
    indy.wait_for_move_finish()
    # 속도 레벨 변경 (빠르게)
    indy.set_task_vel_level(9)
    # 그리퍼 DO 0으로 설정 (닫힘)
    indy.set_endtool_do(0, 0)
    print("Gripper Close")
    time.sleep(1)

    # 인형을 집었다고 가정 후 이동 -> 홈 포지션 또는 특정 위치
    indy.task_move_to([0, 0.46, 0.32, 0, 180, 44])
    indy.wait_for_move_finish()
    indy.task_move_to([0.1, 0.8, 0.32, -12.95, 166.19, 47.27])
    indy.wait_for_move_finish()

    # 그리퍼 다시 오픈
    gripper_open()

    # 홈 포지션 비슷한 곳으로 이동
    indy.task_move_to([0, 0.46, 0.32, 0, 180, 50])
    indy.wait_for_move_finish()

def robot_control(q1, q2, q3, q4, q5, q6, x_line, y_line, z_line):
    """
    로봇(Indy) 제어 프로세스.
    - 큐에서 (x,y,z, 제스처) 정보 수신
    - s_flag(T/F)에 따라 (x,y) 이동 또는 (z) 이동
    - sw_flag(T/F)에 따라 그리퍼 열림/닫힘
    - 타이머 프로세스와 연동(q4,q5)하여 제한 시간 관리
    """
    indy.connect()              # 로봇 연결
    indy.set_task_base(1)       # Task base 설정
    indy.set_task_blend_radius(0.02)
    # 초기 위치(홈 포지션) 이동
    indy.task_move_to([0, 0.46, 0.32, 0, 180, 44])
    print('게임을 시작하려면 흰색 버튼을 눌러주세요')

    while True:
        di_list = indy.get_di()[1]  # 로봇의 디지털 입력(DI) 확인 (0:비어있음, 1:흰색버튼?)
        e_flag = True

        # DI가 감지되면(흰색 버튼 눌림?) 게임 시작
        if di_list:
            t_flag = True
            q4.put(t_flag)  # 타이머 프로세스에 '시작' 신호

            # 게임 진행
            while t_flag and e_flag:
                s = q3.get()  # s_flag (True->xy 모드, False->z 모드)
                if s:
                    # (x, y) 이동 모드
                    indy.set_task_vel_level(9)
                    print('xy')
                    z = indy.get_task_pos()[2]  # 현재 로봇 z 좌표 유지
                    # 큐 q1에서 x,y 좌표를 가져와 로봇 이동
                    if q1.qsize() > 0:
                        x1 = x_divide(q1.get()[0], x_line)
                        y1 = y_divide(q1.get()[1], y_line)
                        # y1이 특정값보다 작으면(아마 가까우면) 로봇 자세가 살짝 달라짐
                        if y1 < 0.2:
                            indy.task_move_to([0.3 - x1, 0.26 + y1, z,
                                               6.8, -171.09, 48.43])
                            indy.wait_for_move_finish()
                            # q6에서 sw_flag(True면 그리퍼 닫힘)
                            if q6.get():
                                gripper_close()
                        else:
                            indy.task_move_to([0.3 - x1, 0.26 + y1, z,
                                               0, 180, 50])
                            indy.wait_for_move_finish()
                            if q6.get():
                                gripper_close()

                else:
                    # (z) 이동 모드
                    indy.set_task_vel_level(3)
                    print('z')
                    # 현재 로봇 x, y
                    x = indy.get_task_pos()[0]
                    y = indy.get_task_pos()[1]
                    if q2.qsize() > 0:
                        z1 = z_divide(q2.get()[0], z_line)
                        # y < 0.46이면 (상단 위치?), z 이동 제한
                        if y < 0.46:
                            if z1 > 0.3:
                                indy.task_move_to([x, y, 0.16,
                                                   6.8, -171.09, 48.43])
                                indy.wait_for_move_finish()
                                if q6.get():
                                    gripper_close()
                            else:
                                indy.task_move_to([x, y, 0.56 - z1,
                                                   6.8, -171.09, 48.43])
                                indy.wait_for_move_finish()
                                if q6.get():
                                    gripper_close()
                        else:
                            if z1 > 0.3:
                                indy.task_move_to([x, y, 0.16,
                                                   0, 180, 50])
                                indy.wait_for_move_finish()
                                if q6.get():
                                    gripper_close()
                            else:
                                indy.task_move_to([x, y, 0.56 - z1,
                                                   0, 180, 50])
                                indy.wait_for_move_finish()
                                if q6.get():
                                    gripper_close()

                # q5에 종료 신호가 들어오면 e_flag=False
                if q5.qsize() > 0:
                    e_flag = q5.get()

def countdown(num_of_secs):
    """
    지정한 초(num_of_secs) 카운트다운을 진행하면서
    콘솔에 남은 시간(분:초) 표시.
    """
    while num_of_secs:
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format, sep='\n')
        time.sleep(1)
        num_of_secs -= 1
    return 1

def timer(q4, q5):
    """
    타이머 프로세스.
    - q4에서 '게임 시작' 신호를 받으면 60초 카운트다운 수행
    - 종료 시 q5에 'False'(종료 신호) 전송
    """
    while True:
        if q4.qsize() > 0:
            t_flag = q4.get()
            if t_flag:
                a = countdown(60)  # 60초 진행
                print("hello")
                if a == 1:
                    e_flag = False
                    q5.put(e_flag)
                    time.sleep(0.1)
                    print("time over")

if __name__ == "__main__":
    # 프로세스 간 통신용 큐 생성
    q1 = Queue(maxsize=1)  # x,y
    q2 = Queue(maxsize=1)  # z
    q3 = Queue(maxsize=1)  # s_flag(xy or z 모드)
    q4 = Queue(maxsize=1)  # 타이머 시작 신호
    q5 = Queue(maxsize=1)  # 타이머 종료 신호
    q6 = Queue(maxsize=1)  # 그리퍼 열림/닫힘(sy_flag)

    # 3개의 프로세스 생성
    p1 = Process(target=open_cam, args=(q1, q2, q3, q6))
    p2 = Process(target=robot_control, args=(q1, q2, q3, q4, q5, q6,
                                             x_line, y_line, z_line))
    p3 = Process(target=timer, args=(q4, q5))

    proc_list = [p1, p2, p3]

    # 프로세스 시작
    for p in proc_list:
        p.start()

    # flag가 True이면 종료 처리
    if flag:
        indy.disconnect()
        for p in proc_list:
            p.terminate()
            p.join()
