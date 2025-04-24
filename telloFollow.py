import time, cv2
from threading import Thread
from djitellopy import Tello

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()
frame_read = tello.get_frame_read()
fly_flag = True
max_fly_time = 500


# General scheme
# Take off
#look for andy
#center on andy
#look away from other people
# if no faces, stay still?
# maybe need to boost up
# If face square too small, move closer?:

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('tellotrainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Andy', 'Kim', 'Mal','Forest']

#names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

maxW = 640
maxH = 480

# Define min window size to be recognized as a face
minW = 0.1*(maxW)
minH = 0.1*(maxH)


if fly_flag:
    tello.takeoff()
    tello.move_up(50)

start_time = time.time()

counter = 0

while True:

    hV = vV = dV = rV = 0


    counter += 1
    img = tello.get_frame_read().frame
    img = cv2.resize(img,(maxW,maxH))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )


    for(x,y,w,h) in faces:



        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        if str(id) == 'Andy':

            if True: #(counter > 20):
                counter = 0

                lrdelta = maxW//2 - (x + w//2)

                #if(lrdelta > .3*maxW):
                #    if fly_flag:
                #        #tello.rotate_counter_clockwise(20)


                if(lrdelta > .2*maxW):
                    if fly_flag:
                        #tello.rotate_counter_clockwise(10)
                        #print("CC Turn")
                        rV = -30

                #elif(lrdelta < -.3*maxW):
                #     if fly_flag:
                #        tello.rotate_clockwise(20)

                elif(lrdelta < -.2*maxW):
                    if fly_flag:
                        #tello.rotate_clockwise(10)
                        rV = 30

                    #print("Clockwise Turn")

                uddelta = maxH//2 - (y + h//2)

                if(uddelta > 0.2*maxH):
                    if fly_flag:
                        #tello.move_up(20)
                        vV = 20
                    #print("Lower")

                elif(uddelta < -.2*maxH):
                    if fly_flag:
                        #tello.move_down(20)
                        vV = -20
                    #print("Higher")

                print("width= " + str(w))

                if(w < 100):
                    if fly_flag:
                        #tello.move_forward(20)
                        dV = 15

                elif(w>200):
                    if fly_flag:
                        #tello.move_back(20)
                        dV = -15




            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        else:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)


        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    tello.send_rc_control(hV, dV, vV, rV)


    cv2.imshow("result",img)
    if cv2.waitKey(30) == ord('q'):
        break

    current_time = time.time()

    seconds_passed = current_time - start_time
    #print("Seconds = " + str(seconds_passed))

    if(seconds_passed > max_fly_time):
        if fly_flag:
            tello.land()




cam.release()
cv2.destroyAllWindows()


