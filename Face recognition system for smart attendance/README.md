# Smart Attendance System with Face Recognition and Eye Blink Detection

A web-based attendance system that uses facial recognition and eye blink detection to mark student attendance automatically.

## Features

- Face recognition using CNN model
- Eye blink detection using dlib
- Student registration and management
- Automated attendance marking
- Staff portal for attendance management
- Export attendance reports to Excel
- Secure authentication system

## Project Structure

```
├── app/
│   ├── static/          # CSS, JavaScript, and images
│   ├── templates/       # HTML templates
│   ├── models/          # ML models and database models
│   └── routes/          # Application routes
├── database/           # Database files
├── uploads/            # Uploaded student images
└── instance/          # Instance-specific files
```

## Setup Instructions

1. Create a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```
   flask db init
   flask db migrate
   flask db upgrade
   ```

4. Run the application:
   ```
   flask run
   ```

## Technologies Used

- Frontend: HTML, CSS, JavaScript
- Backend: Flask (Python)
- Database: SQLAlchemy
- ML/DL: TensorFlow, dlib, OpenCV
- Authentication: Flask-Login

## Security Features

- Secure password hashing
- JWT-based authentication
- CSRF protection
- Secure file uploads

## License

MIT License