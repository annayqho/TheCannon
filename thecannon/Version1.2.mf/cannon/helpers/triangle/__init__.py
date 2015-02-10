#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, unicode_literals

try:
    from triangle import *
except ImportError:
    # fallback to internal
    from .triangle import *
