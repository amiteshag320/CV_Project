import cv2 as cv
import numpy as np
from flask import Flask, render_template, request,redirect,url_for,send_file
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
@app.route('/video/processed', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['video']
        path_file=os.path.join("uploads",file.filename)
        file.save(path_file)
        cap = cv.VideoCapture(path_file)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # fps = cap.get(cv.CV_CAP_PROP_FPS)

        size = (frame_width, frame_height)
        new_path = os.path.join("static",file.filename)
        result = cv.VideoWriter(new_path, 
                         cv.VideoWriter_fourcc(*'MP4V'),
                         10, size)
        ret,frame1 = cap.read()
        ret,frame2 = cap.read()
        while cap.isOpened():
            diff = cv.absdiff(frame1,frame2)
            gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray,(5,5),0)
            _,thresh = cv.threshold(blur, 20,255,cv.THRESH_BINARY)
            dilated= cv.dilate(thresh,None,iterations =3)
            contours,_ = cv.findContours(dilated,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            for contour in contours:
                (x,y,w,h)=cv.boundingRect(contour)
                if cv.contourArea(contour) <700:
                    continue
                
                cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
                cv.putText(frame1,"MOTION : {}".format("DETECTED"),(10,20),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            result.write(frame1)
            frame1 = frame2
            ret,frame2 = cap.read()
            if ret==True:
                continue
            else:
                break
            
        cap.release()
        cv.destroyAllWindows()
        return new_path
        
if __name__ == '__main__':
    app.run(debug=True,use_reloader=True)    
      