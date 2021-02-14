#!/usr/bin/env python
from __future__ import print_function
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
    faceShapeSplit=faceShape.split(":")
    hairSplit=faceShape.split(":")
    ganderAndAge=faceShapeSplit[3].split("(")
    age="Young"
    photo=""
    link=""
    if ganderAndAge[0]=="Female" and faceShapeSplit[1]=="Triangle":
        photo="kalp-kadin_300320.jpg"
        link="https://www.atasunoptik.com.tr/kadin-kalp-yuz-sekline-uygun-gunes-gozlukleri"
       
    if ganderAndAge[0]=="Female" and faceShapeSplit[1]=="Rectangle":
        photo="kare-kadin_300320.jpg"
        link="https://www.atasunoptik.com.tr/kadin-kare-yuz-sekline-uygun-gunes-gozlukleri"
    
    if ganderAndAge[0]=="Female" and faceShapeSplit[1]=="Oval":
        photo="oval-kadin_300320.jpg"
        link="https://www.atasunoptik.com.tr/kadin-oval-yuz-sekline-uygun-gunes-gozlukleri"
    
    if ganderAndAge[0]=="Female" and faceShapeSplit[1]=="Square":
        photo="kare-kadin_300320.jpg"
        link="https://www.atasunoptik.com.tr/kadin-kare-yuz-sekline-uygun-gunes-gozlukleri"    
    
    if ganderAndAge[0]=="Female" and faceShapeSplit[1]=="Round":
        link="https://www.atasunoptik.com.tr/kadin-yuvarlak-yuz-sekline-uygun-gunes-gozlukleri"
        photo="yuvarlak-kadin_300320.jpg"
        
        
        
    if ganderAndAge[0]=="Male" and faceShapeSplit[1]=="Triangle":
        photo="kalp-erkek_300320.jpg"
        link="https://www.atasunoptik.com.tr/erkek-kalp-yuz-sekline-uygun-gunes-gozlukleri"
       
    if ganderAndAge[0]=="Male" and faceShapeSplit[1]=="Rectangle":
        photo="kare-erkek_300320.jpg"
        link="https://www.atasunoptik.com.tr/erkek-kare-yuz-sekline-uygun-gunes-gozlukleri"
    
    if ganderAndAge[0]=="Male" and faceShapeSplit[1]=="Oval":
        photo="oval-erkek_300320.jpg"
        link="https://www.atasunoptik.com.tr/erkek-oval-yuz-sekline-uygun-gunes-gozlukleri"
    
    if ganderAndAge[0]=="Male" and faceShapeSplit[1]=="Square":
        photo="kare-erkek_300320.jpg"
        link="https://www.atasunoptik.com.tr/erkek-yuvarlak-yuz-sekline-uygun-gunes-gozlukleri"
    
    if ganderAndAge[0]=="Male" and faceShapeSplit[1]=="Round":
        photo="yuvarlak-erkek_300320.jpg"
        link="https://www.atasunoptik.com.tr/erkek-kare-yuz-sekline-uygun-gunes-gozlukleri"
        
        
    if faceShape[0]=="Bir Hata Oluştu":
        photo="vestes_2.png"
        link="https://www.atasunoptik.com.tr/yuz-sekillerine-uygun-gunes-gozlugu-onerileri"
        
        
    if ganderAndAge[1]=="0-2)" or ganderAndAge[1]=="4-6)" or ganderAndAge[1]=="8-12)" :
        age="child"
        
    if ganderAndAge[1]=="15-20)" or ganderAndAge[1]=="25-32)" :
        age="Young"
        
    if ganderAndAge[1]=="38-43)" or ganderAndAge[1]=="48-53)" :
        age="middle aged"
        
    if ganderAndAge[1]=="60-100)":
        age="Old"
    
    return render_template('capture.html',stamp=timestamp, path=path,faceShape=faceShapeSplit[1],gander=ganderAndAge[0],age=age,hairColor=hairSplit[5],photo=photo,link=link)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)