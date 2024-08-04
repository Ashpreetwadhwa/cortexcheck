from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text,Numeric
db=SQLAlchemy()
def init_db(app):
    db.init_app(app)