"""
Make Py3k migration easier
==========================

remap some python 2 built-ins on to py3k behavior or equivalent.
Most of them become generators.

"""
import operator
import sys

PY3 = sys.version_info[0] > 2

__all__ = ['PY3', 'map', 'filter', 'range', 'zip', 'reduce', 'zip_longest',
           'iteritems', 'iterkeys', 'itervalues', 'StringIO']

if PY3:
    map = map
    filter = filter
    range = range
    zip = zip
    from functools import reduce
    from itertools import zip_longest
    iteritems = operator.methodcaller('items')
    iterkeys = operator.methodcaller('keys')
    itervalues = operator.methodcaller('values')
    from io import StringIO
else:
    range = xrange
    reduce = reduce
    from itertools import imap as map
    from itertools import ifilter as filter
    from itertools import izip as zip
    from itertools import izip_longest as zip_longest
    iteritems = operator.methodcaller('iteritems')
    iterkeys = operator.methodcaller('iterkeys')
    itervalues = operator.methodcaller('itervalues')
    from cStringIO import StringIO
