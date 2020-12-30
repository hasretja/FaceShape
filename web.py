#!/usr/bin/env python
import os
import shutil
from flask import Flask, render_template, request, \
    Response, send_file, redirect, url_for
from camera import Camera
from send_email import Email
import face_shape_detection


app = Flask(__name__)
camera = None
mail_server = None
mail_conf = "static/mail_conf.json"

def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera

def get_mail_server():
    global mail_server
    if not mail_server:
        mail_server = Email(mail_conf)

    return mail_server

@app.route('/')
def root():
    return redirect(url_for('index'))


@app.route('/index/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_feed()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/')
def capture():
    camera = get_camera()
    stamp = camera.capture()
    return redirect(url_for('show_capture', timestamp=stamp))

def stamp_file(timestamp):
    return 'captures/' + timestamp +".jpg"

@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    path = stamp_file(timestamp)
    basedir = os.path.abspath(os.path.dirname(__file__))
    basedir=basedir.replace(r"\\","/")
    email_msg = None
    if request.method == 'POST':
        if request.form.get('email'):
            email = get_mail_server()
            email_msg = email.send_email('static/{}'.format(path), 
                request.form['email'])
        else:
            email_msg = "Email field empty!"

    faceShape=face_shape_detection.deneme(basedir + "/static/captures/" + str(timestamp) + ".jpg")
    if faceShape=='Your Face Shape Square':
        suggestionGlass1 = "browline Glasses.jpg"
        suggestionGlass2 = "catEye Glasses .PNG"
        suggestionGlass3 = "oval Glasses.jpg"
        suggestionGlass4 = "rectangle Glasses.jpg"

        suggestionHair1 = "square Woman-1.PNG"
        suggestionHair2 = "square Man-1.PNG"
        suggestionHair3 = "square Woman-2.PNG"
        suggestionHair4 = "square Man-2.PNG"
    if faceShape=='Your Face Shape Oval':
        suggestionGlass1 = "aviator Glasses.jpg"
        suggestionGlass2 = "browline Glasses.jpg"
        suggestionGlass3 = "rectangle Glasses.jpg"
        suggestionGlass4 = "square Glasses.png"

        suggestionHair1 = "oval Woman-1.jpg"
        suggestionHair2 = "oval Man-1.jpg"
        suggestionHair3 = "oval Woman-2.jpg"
        suggestionHair4 = "oval Man-2.jpg"
    if faceShape=='Your Face Shape Rectangle':
        suggestionGlass1 = "browline Glasses.jpg"
        suggestionGlass2 = "catEye Glasses .PNG"
        suggestionGlass3 = "oval Glasses.jpg"
        suggestionGlass4 = "rectangle Glasses.jpg"

        suggestionHair1 = "rectangle Woman-1.PNG"
        suggestionHair2 = "rectangle Man-1.PNG"
        suggestionHair3 = "rectangle Woman-2.PNG"
        suggestionHair4 = "rectangle Man-2.PNG"
    if faceShape=='Your Face Shape Round':
        suggestionGlass1 = "aviator Glasses.jpg"
        suggestionGlass2 = "rectangle Glasses.jpg"
        suggestionGlass3 = "round Glasses.PNG"
        suggestionGlass4 = "square Glasses.png"

        suggestionHair1 = "round Woman-1.PNG"
        suggestionHair2 = "round Man-1.PNG"
        suggestionHair3 = "round Woman-2.PNG"
        suggestionHair4 = "round Man-2.PNG"
    if faceShape=='Your Face Shape Triangle':
        suggestionGlass1 = "browline Glasses.jpg"
        suggestionGlass2 = "catEye Glasses .PNG"
        suggestionGlass3 = "oval Glasses.jpg"
        suggestionGlass4 = "round Glasses.PNG"

        suggestionHair1 = "triangle Woman-1.PNG"
        suggestionHair2 = "triangle Man-1.PNG"
        suggestionHair3 = "triangle Woman-2.PNG"
        suggestionHair4 = "triangle Man-2.PNG"

    if faceShape == 'An Error Occurred Try Again.':
        suggestionGlass1 = "browline Glasses.jpg"
        suggestionGlass2 = "catEye Glasses .PNG"
        suggestionGlass3 = "oval Glasses.jpg"
        suggestionGlass4 = "round Glasses.PNG"

        suggestionHair1 = "triangle Woman-1.PNG"
        suggestionHair2 = "triangle Man-1.PNG"
        suggestionHair3 = "triangle Woman-2.PNG"
        suggestionHair4 = "triangle Man-2.PNG"
    return render_template('capture.html',
        stamp=timestamp, path=path, email_msg=email_msg,faceShape=faceShape,
                           suggestionGlass1=suggestionGlass1,suggestionGlass2=suggestionGlass2,suggestionGlass3=suggestionGlass3,suggestionGlass4=suggestionGlass4,
                           suggestionHair1=suggestionHair1,suggestionHair2=suggestionHair2,suggestionHair3=suggestionHair3,suggestionHair4=suggestionHair4)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)