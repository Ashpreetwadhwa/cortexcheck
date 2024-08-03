from flask import request, Flask, render_template, redirect, session, url_for, send_from_directory
import os
import uuid
from Registerclass import *
from database import *
from  utility import *

app = Flask(__name__, static_folder='static', static_url_path='')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
username="root"
passw="As#20021103"
host="mydbms.c52iy2uoqusb.eu-north-1.rds.amazonaws.com"
port="3306"
database="client"

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{username}:{passw}@{host}/{database}'

app.secret_key='secret_key'
init_db(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method=="POST":
        email=request.form['email']
        password = request.form['pwd']
        userval=User.query.filter_by(email=email).first()
        if userval and userval.check_password(password):
            session['name']=userval.name
            session['email']=userval.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='invalid user')
    return render_template('login.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/register',methods=["GET","POST"])
def register():
    if request.method=="POST":
        name=request.form['name']
        email = request.form['email']
        password = request.form['pwd']
        image = request.files["photo-upload"]
        filename = str(uuid.uuid4()) + '.' + image.filename.rsplit('.', 1)[1].lower()
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        output, confidence = predict_with_confidence(image_path)
        new_user=User(name=name,email=email,password=password,image=filename,output=output,confidence=confidence)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/dashboard', methods=["POST", "GET"])
def dashboard():
    user = User.query.filter_by(email=session["email"]).first()
    user_image = user.image  # This should be just the filename

    print(user_image)
    return render_template('dashboard.html', user=user, user_image=user_image)

@app.route('/logout')
def logout():
    db.session.close()
    return render_template('login.html')


import os


@app.route('/delete', methods=['POST'])
def delete_account():
    # Ensure the session contains an email
    if "email" in session:
        user = User.query.filter_by(email=session["email"]).first()

        if user:
            # Remove the image file if it exists
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], user.image)
            if os.path.isfile(image_path):
                os.remove(image_path)

            # Delete the user record from the database
            db.session.delete(user)
            db.session.commit()

            # Clear the session and redirect to the registration page
            session.pop('email', None)
            session.pop('name', None)
            return redirect('/register')
        else:
            # User not found, handle accordingly
            return redirect('/dashboard')
    else:
        # No email in session, redirect to login
        return redirect('/login')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True,host='0.0.0.0')
