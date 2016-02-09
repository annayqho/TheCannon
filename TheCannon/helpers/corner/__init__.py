#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, unicode_literals

try:
    from corner import *
except ImportError:
    # fallback to internal
    from .corner import *
