import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from flask import Flask
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)
import cv2
import mediapipe as mp

from flask_cors import CORS, cross_origin
CORS(app, support_credentials=True)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

import jsonpickle

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

@app.route("/")
@cross_origin(supports_credentials=True)
def index():
#     side ways rep
#     cap = cv2.VideoCapture("gymvideo.mp4")
    
    up = False
    counter = 0

    while True:
        success, img = cap.read()
#         img = cv2.resize(img, (1280,720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)
        # print("-----------------------------------------------------")
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id,lm,cx,cy)
                points[id] = (cx,cy)


            cv2.circle(img, points[12], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[14], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[11], 15, (255,0,0), cv2.FILLED)
            cv2.circle(img, points[13], 15, (255,0,0), cv2.FILLED)


            if not up and points[14][1] + 40 < points[12][1]:
                print("UP")
                up = True
                counter += 1
            elif points[14][1] > points[12][1]:
                print("Down")
                up = False
            # print("----------------------",counter)

        cv2.putText(img, str(counter), (100,150),cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0),12)
        cv2.imshow("img",img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
                break

#     cv2.imshow("img",img)
    cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    return jsonpickle.encode(counter)
#     return counter


@app.route("/squat")
def index2():
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle

        return angle 

    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle), 
                               tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                # Curl counter logic
                if angle > 160:
                    stage = "up"
                if angle < 160 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    return jsonpickle.encode(counter)


# dance
@app.route("/dance")
def index3():
    moves = {
    "dab": [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_SHOULDER],
    "shuffle": [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
    "floss": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
              mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_HIP,
              mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    "moonwalk": [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
                 mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL],
    "robot": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
              mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_SHOULDER,
              mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
    "twerk": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
              mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE],
    "vogue": [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW,
              mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST,
              mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER],
    "running man": [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
                    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE],
    "the worm": [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_HIP,
                 mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                 mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                 mp_pose.PoseLandmark.RIGHT_ANKLE],
    }


    # Initialize video capture and PoseNet
#     cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_pose = None
        detected_moves = set()
    #     cv2.imshow('Mediapipe Feed', frame)
        while True:
            # Read a frame from the video capture
            ret, frame = cap.read()
            if not ret:
                break

    #         # Flip the image horizontally for a mirror effect
    #         frame = cv2.flip(frame, 1)

    #         # Convert the image to grayscale
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray.flags.writeable = False

            # Detect the pose in the image
            results = pose.process(gray)

            # Draw the pose landmarks on the image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Check for each dance move
                for move_name, move_landmarks in moves.items():
                    move_detected = True
                    for landmark in move_landmarks:
                        if results.pose_landmarks.landmark[landmark].visibility < 0.5:
                            move_detected = False
                            break

                    if move_detected:
                        if move_name not in detected_moves:
                            detected_moves.add(move_name)
                            print(f"Detected {move_name}!")
                            cv2.putText(image, str(move_name), (100,150),cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0),12)
                    else:
                        detected_moves.discard(move_name)

                # Save the current pose for comparison in the next frame
                prev_pose = results.pose_landmarks

            # Show the image with landmarks and wait for a key press
            cv2.putText(image, str(move_name), (100,150),cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0),12)
#             cv2.imshow('Deadlift Detection', image)
            cv2.imshow('Pose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()
    return jsonpickle.encode(detected_moves)
# lunges


@app.route("/lunges")
def index4():
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle

        return angle 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            # Read the frame from the video capture
            ret, image = cap.read()

            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with Mediapipe
            results = pose.process(image)

            # Draw the pose landmarks on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Detect deadlifts
            if results.pose_landmarks is not None:

                # Get the landmarks for the left hip, left knee, and left ankle
                landmarks = results.pose_landmarks.landmark
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate the angles between the legs
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Visualize the angle on the image
                cv2.putText(image, str(angle), tuple(np.multiply(knee, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Count the number of deadlift repetitions
                if right_angle > 160 and left_angle<40:
                    stage = "up"
                if right_angle < 160 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)

            # Display the image

                cv2.putText(image, str(counter), (100,150),cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0),12)
                cv2.imshow('Deadlift Detection', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Show the image with landmarks and wait for a key press
#             cv2.imshow('Pose', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

        # Release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()
    return jsonpickle.encode(counter)


@app.route("/deadlift")
def index5():
    counter=0
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle

        return angle 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            # Read the frame from the video capture
            ret, image = cap.read()

            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with Mediapipe
            results = pose.process(image)

            # Draw the pose landmarks on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Detect deadlifts
            if results.pose_landmarks is not None:

                # Get the landmarks for the left hip, left knee, and left ankle
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate the angle between the left hip, left knee, and left ankle
                angle = calculate_angle(hip, knee, ankle)

                # Visualize the angle on the image
                cv2.putText(image, str(angle), tuple(np.multiply(knee, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Count the number of deadlift repetitions
                if angle > 160:
                    stage = "up"
                if angle < 160 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)

            # Display the image
#             cv2.imshow('Deadlift Detection', image)

            # Exit on ESC
            # Display the image

            cv2.putText(image, str(counter), (100,150),cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0),12)
            cv2.imshow('Deadlift Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Show the image with landmarks and wait for a key press
#             cv2.imshow('Pose', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

        # Release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()
    return jsonpickle.encode(counter)

# app.run()
# if __name__ == '__main__':
# #     app.run(debug=True,threaded=True)
app.run()