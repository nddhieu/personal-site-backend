import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    message = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///chatbot.db", echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
