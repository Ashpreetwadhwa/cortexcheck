from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
db=SQLAlchemy()
def init_db(app):
    db.init_app(app)