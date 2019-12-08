import os
import cv2
import numpy as np
from urllib.request import urlopen
import darknet
from firebase import firebase
import time


"""
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
"""


netMain = None
metaMain = None
altNames = None


isDetected = False
inWater = False
isShaking = False


detectT = 0
waterT = 0
shakeT = 0


firebase = firebase.FirebaseApplication('https://forpet-3ae34.firebaseio.com', None)


url = ' http://192.168.4.1/stream'


def detect(detections, firebase):


    global detectT
    detect=False


    for detection in detections:

        label = detection[0].decode()

        if(label=='cat' or label=='dog' or label=='bird'):

            #print(detection)
            x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]

            xmin = int(round(x - (w / 2)))

            xmax = int(round(x + (w / 2)))

            ymin = int(round(y - (h / 2)))

            ymax = int(round(y + (h / 2)))


            if((abs(xmax-xmin)*abs(ymax-ymin))>=((darknet.network_width(netMain)*darknet.network_height(netMain))*0.4)):

                detect=True


    if(~detect):

        isDetected=False


    if(detect and ~isDetected):


        if (time.time() - detectT > 10):

            isDetected=True

            result = firebase.put('/', 'detect', 1)

            print("\ndetect\n")

            detectT = time.time()



def YOLO():


    global metaMain, netMain, altNames
    global isDetected, inWater, isShaking
    global waterT, shakeT


    configPath = "team_41.cfg"
    weightPath = "team_41.weights"
    metaPath = "./cfg/team_41.data"


    if not os.path.exists(configPath):

        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")


    if not os.path.exists(weightPath):

        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")


    if not os.path.exists(metaPath):

        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")


    if netMain is None:

        netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1


    if metaMain is None:

        metaMain = darknet.load_meta(metaPath.encode("ascii"))


    if altNames is None:

        try:

            with open(metaPath) as metaFH:

                metaContents = metaFH.read()

                import re

                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)


                if match:

                    result = match.group(1)

                else:

                    result = None


                try:

                    if os.path.exists(result):

                        with open(result) as namesFH:

                            namesList = namesFH.read().strip().split("\n")

                            altNames = [x.strip() for x in namesList]

                except TypeError:

                    pass


        except Exception:

            pass



    # CAMERA_WIDTH = 320
    # CAMERA_HEIGHT = 240


    CAMERA_BUFFER_SIZE = 4096
    stream = urlopen(url)

    bytes = b''


    while True:

        bytes += stream.read(CAMERA_BUFFER_SIZE)

        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')


        if (bytes.find(b"water") > -1):

            if (~inWater):

                if (time.time() - waterT > 10):

                    inWater = True

                    firebase.put('/', 'water', 1)
                    print("water");

                    waterT = time.time()

        else:

            inWater = False


        if (bytes.find(b"shake") > -1):

            if (~isShaking):

                if (time.time() - shakeT > 10):

                    isShaking = True

                    firebase.put('/', 'shake', 1)
                    print("shake");

                    shakeT = time.time()

        else:

            isShaking = False


        if a > -1 and b > -1:

            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]


            try:

                frame_read = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain), 3)

                frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

                frame_resized = cv2.resize(frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)

                darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

                detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

                print(detections)

                detect(detections, firebase)


            except Exception:

                print (np.BUFSIZE)

                continue



if __name__ == "__main__":
    YOLO()