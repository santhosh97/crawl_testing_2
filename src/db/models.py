from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True)
    github_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    owner = Column(String, nullable=False)
    full_name = Column(String, nullable=False, index=True)
    url = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    star_records = relationship("StarRecord", back_populates="repository")

    __table_args__ = (
        UniqueConstraint('owner', 'name', name='unique_owner_name'),
    )


class StarRecord(Base):
    __tablename__ = "star_records"

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    star_count = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    repository = relationship("Repository", back_populates="star_records")

    __table_args__ = (
        Index('idx_repository_recorded_at', 'repository_id', 'recorded_at'),
    )