from database import *
from app import *
import bcrypt
from utility import *
class User(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(100),nullable=False)
    email = db.Column(db.String(100), unique=True,nullable=False)
    password=db.Column(db.String(100),nullable=False)
    image=db.Column(db.String(100), nullable=False)
    output = db.Column(db.String(100), nullable=False)
    confidence= db.Column(db.Integer, nullable=False)
    def __init__(self,email,password,name,image,output,confidence):
        self.name=name
        self.email=email
        self.password=bcrypt.hashpw(password.encode("utf-8"),bcrypt.gensalt()).decode("utf-8")
        self.image=image
        self.output=output
        self.confidence=confidence
    def check_password(self,password):
        return bcrypt.checkpw(password.encode("utf-8"),self.password.encode("utf-8"))

