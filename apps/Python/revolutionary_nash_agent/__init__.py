#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Nash Agent (RNA) - 7th Member of PADS Boardroom

This package implements the Revolutionary Nash Agent as a formal member
of the Panarchy Adaptive Decision System, bringing quantum-biological
Nash equilibrium capabilities to the existing 6-agent boardroom.

The RNA integrates:
- Temporal-Biological Nash Dynamics
- Antifragile Quantum Coalitions  
- Quantum Nash Equilibria
- Machiavellian Strategic Frameworks
- Robin Hood Protocols

Author: Revolutionary AI Team
Version: 1.0.0
"""

from .core.rna_core import RevolutionaryNashAgent
from .server.rna_server import create_rna_server
from .integration.pads_interface import PADSInterface

__version__ = "1.0.0"
__author__ = "Revolutionary AI Team"

__all__ = [
    "RevolutionaryNashAgent",
    "create_rna_server", 
    "PADSInterface"
]