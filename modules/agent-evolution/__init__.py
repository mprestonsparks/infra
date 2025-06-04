"""
DEAN Agent Evolution Module

This module provides the core FastAPI service for the Distributed Evolutionary Agent Network (DEAN)
system, enabling agent creation, evolution, and management through REST APIs.
"""

__version__ = "1.0.0"
__author__ = "DEAN Development Team"
__description__ = "Distributed Evolutionary Agent Network - Agent Evolution Service"

from .main import app

__all__ = ["app"]