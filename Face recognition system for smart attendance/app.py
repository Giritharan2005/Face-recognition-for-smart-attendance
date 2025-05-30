from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from datetime import datetime
import os
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class Staff(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    department = db.Column(db.String(50), nullable=False)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    register_no = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    batch = db.Column(db.String(20), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'present' or 'absent'
    face_recognized = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return Staff.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Create default staff account if it doesn't exist
    if not Staff.query.filter_by(username='TEAM-5').first():
        default_staff = Staff(
            username='TEAM-5',
            password_hash=generate_password_hash('MLF-PROJECT'),
            department='Admin'
        )
        db.session.add(default_staff)
        db.session.commit()

# Import routes
from routes import *

if __name__ == '__main__':
    app.run(debug=True)