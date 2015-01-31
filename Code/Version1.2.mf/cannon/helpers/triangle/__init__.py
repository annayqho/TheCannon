#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, unicode_literals

try:
    import triangle
except ImportError:
    # fallback to internal
    from .triangle import *
