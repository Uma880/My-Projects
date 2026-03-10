import cv2
import numpy as np

# SHOW IMAGE PROCESSING STEPS

def show_processing_steps(roi):

    original = roi.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced,(5,5),0)

    edges = cv2.Canny(blur,30,100)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    row1 = np.hstack((original, gray_bgr))
    row2 = np.hstack((enhanced_bgr, edges_bgr))

    final = np.vstack((row1,row2))

    cv2.imshow("Eye Processing Steps", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# PROBABILITY OUTPUT
def probability_output(eye_score, body_score):

    print("\n=========== RESULT ===========")

    print("\nIn Eyes:")

    if not eye_score:
        print("Healthy Eye Detected ✅")

    else:
        max_eye=max(eye_score.values())
        eye_prob={d:int((s/max_eye)*100) for d,s in eye_score.items()}
        eye_sorted=sorted(eye_prob.items(), key=lambda x:x[1], reverse=True)

        print(f"Most Probable: {eye_sorted[0][0]} ({eye_sorted[0][1]}%)")

        for d,p in eye_sorted[1:4]:
            print(f"• {d} ({p}%)")


    print("\nIn Body:")

    if not body_score:
        print("No major body risk detected")

    else:
        max_body=max(body_score.values())
        body_prob={d:int((s/max_body)*100) for d,s in body_score.items()}
        body_sorted=sorted(body_prob.items(), key=lambda x:x[1], reverse=True)

        print(f"Most Probable: {body_sorted[0][0]} ({body_sorted[0][1]}%)")

        for d,p in body_sorted[1:4]:
            print(f"• {d} ({p}%)")

# SCLERA ANALYSIS ENGINE

def sclera_engine(redness,density,yellow,brightness):

    eye_score={}
    body_score={}

    # Healthy check
    if (0.10 <= density <= 0.18 and
        90 <= redness <= 140 and
        yellow < 140 and
        100 <= brightness <= 150):

        probability_output({}, {})
        return

    # Diffuse red eye
    if redness > 150:

        eye_score["Conjunctivitis"]=3
        eye_score["Eye Allergy"]=2
        eye_score["Keratitis"]=2
        eye_score["Blepharitis"]=2

        body_score["Stress"]=2
        body_score["Fatigue"]=2


    # Dry eye
    if redness > 120 and brightness < 110:

        eye_score["Dry Eye Syndrome"]=3
        eye_score["Sjogren Syndrome"]=2

        body_score["Dehydration"]=2


    # Anemia
    if redness < 90:

        body_score["Iron Deficiency Anaemia"]=4
        body_score["Vitamin B12 Deficiency"]=3


    # Liver disease
    if yellow > 140:

        body_score["Liver Disease"]=4
        body_score["Hepatitis"]=3


    # Heart disease
    if density > 0.22:

        body_score["Hypertension"]=3
        body_score["Cardiovascular Disease"]=2
        body_score["Stroke Risk"]=2


    # Diabetes
    if density > 0.30:

        body_score["Diabetes"]=3
        body_score["High Cholesterol"]=3


    # Oxygen deficiency
    if density < 0.08:

        body_score["Hypoxia"]=3
        body_score["Sleep Apnea"]=2


    probability_output(eye_score,body_score)

# RETINA ANALYSIS ENGINE

def retina_engine(redness,density,brightness):

    eye_score={}
    body_score={}

    # Retinal bleeding
    if redness > 140:

        eye_score["Diabetic Retinopathy"]=4
        eye_score["Retinal Hemorrhage"]=3

        body_score["Diabetes"]=3


    # Narrow vessels
    if density > 0.25:

        eye_score["Retinal Vascular Disease"]=3

        body_score["Hypertension"]=3
        body_score["Stroke Risk"]=2


    # Yellow deposits
    if brightness > 150:

        eye_score["Macular Degeneration"]=3

        body_score["High Cholesterol"]=2


    # Optic nerve damage
    if brightness < 80:

        eye_score["Glaucoma"]=3


    probability_output(eye_score,body_score)

# SCLERA CAMERA DETECTION
def sclera_detection():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not detected ❌")
        return

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

    roi = None   # important fix

    print("Press Q to capture")

    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in eyes:

            roi = frame[y:y+h, x:x+w]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Sclera Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if roi is None:
        print("Eye not detected ❌ Please keep eye closer to camera.")
        return

    show_processing_steps(roi)

    redness = np.mean(roi[:,:,2])
    yellow = np.mean((roi[:,:,1]+roi[:,:,2])/2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,30,100)

    density = cv2.countNonZero(edges)/edges.size
    brightness = np.mean(gray)

    sclera_engine(redness,density,yellow,brightness)

# RETINA IMAGE UPLOAD

def retina_detection():

    path=input("Upload retinal image path: ")

    img=cv2.imread(path)

    if img is None:
        print("Invalid image path ❌")
        return

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges=cv2.Canny(gray,50,150)

    density=cv2.countNonZero(edges)/edges.size

    redness=np.mean(img[:,:,2])

    brightness=np.mean(gray)

    retina_engine(redness,density,brightness)

# MAIN MENU
while True:

    print("\n1 - Check with Sclera (Camera)")
    print("2 - Check with Retina (Upload Image)")
    print("3 - Exit")

    ch=input("Enter choice: ")

    if ch=="1":
        sclera_detection()

    elif ch=="2":
        retina_detection()

    elif ch=="3":
        break