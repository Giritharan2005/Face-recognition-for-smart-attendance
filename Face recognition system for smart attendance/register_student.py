from app import app, db, Student
from werkzeug.security import generate_password_hash
import os
import shutil
from datetime import datetime

def register_student(register_no, name, batch, department, image_path):
    with app.app_context():
        # Check if student already exists
        existing_student = Student.query.filter_by(register_no=register_no).first()
        if existing_student:
            print(f"Student with register number {register_no} already exists.")
            return

        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Copy the image to uploads folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"{register_no}_{timestamp}.jpg"
        destination_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        try:
            shutil.copy2(image_path, destination_path)
        except Exception as e:
            print(f"Error copying image: {e}")
            return

        # Create new student
        new_student = Student(
            register_no=register_no,
            name=name,
            batch=batch,
            department=department,
            image_path=destination_path
        )

        try:
            db.session.add(new_student)
            db.session.commit()
            print(f"Successfully registered student {name}")
        except Exception as e:
            print(f"Error registering student: {e}")
            db.session.rollback()

if __name__ == '__main__':
    # Register GIRITHARAN D
    student_image_path = 'path_to_your_image.jpg'  # Replace with actual image path
    register_student(
        register_no='BAM017',
        name='GIRITHARAN D',
        batch='2023-2027',
        department='CSE(AIML)',
        image_path=student_image_path
    )