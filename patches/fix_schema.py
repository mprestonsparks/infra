#!/usr/bin/env python3
"""
Patch to fix the IndexAgent schema issue with partial indexes
"""

import sys
import os

# Add IndexAgent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../IndexAgent'))

# Import the fixed versions
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Text, Boolean,
    ForeignKey, Index, CheckConstraint, UniqueConstraint, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid
from datetime import datetime

Base = declarative_base()

# Re-define PerformanceMetrics without the problematic index
class PerformanceMetrics(Base):
    """
    Performance metrics table per FR-006: Calculate and track value-per-token metrics.
    Schema: agent_evolution.performance_metrics
    """
    __tablename__ = "performance_metrics"
    __table_args__ = (
        Index('ix_performance_metrics_agent_id', 'agent_id'),
        Index('ix_performance_metrics_recorded_at', 'recorded_at'),
        Index('ix_performance_metrics_metric_name', 'metric_name'),
        # Removed the partial index that was causing issues
        CheckConstraint('metric_value >= 0', name='check_metric_value_positive'),
        {'schema': 'agent_evolution'}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_evolution.agents.id'), nullable=False)
    
    # Metric details
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    
    # Temporal tracking
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Context for the metric
    context = Column(JSON, nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="performance_metrics")

print("Schema patch loaded successfully")