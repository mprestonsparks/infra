"""Knowledge repository for meta-learning and pattern analysis."""

# Try to import full metrics database, fall back to simple version
try:
    from .metrics_database import MetricsDatabase, MetricRecord, QueryBuilder
except ImportError:
    from .simple_metrics_db import SimpleMetricsDatabase as MetricsDatabase, MetricRecord
    QueryBuilder = None

from .knowledge_base import KnowledgeBase, KnowledgeEntry, InsightType
from .meta_learner import MetaLearner, LearningStrategy
from .repository_manager import RepositoryManager

__all__ = [
    'MetricsDatabase',
    'MetricRecord',
    'QueryBuilder',
    'KnowledgeBase',
    'KnowledgeEntry',
    'InsightType',
    'MetaLearner',
    'LearningStrategy',
    'RepositoryManager'
]