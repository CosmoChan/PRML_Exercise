# 兼容Python2和Python3的代码

## Setup

```python
import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six
```



## print 

```python
# Python 2 only:
print 'Hello'

# Python 2 and 3:
print('Hello')

# Python 2 only:
print 'Hello', 'Guido'

# 使用这种
# Python 2 and 3:
from __future__ import print_function# (at top of module)
print('Hello', 'Guido')

# Python 2 only:
print 'Hello',

# Python 2 and 3:
from __future__ import print_function
print('Hello', end='')
```

## Division

```python
# Python 2 only:
assert 2 / 3 == 0

# Python 2 and 3:
assert 2 // 3 == 0

# Python 3 only:
assert 3 / 2 == 1.5

#使用这种
# Python 2 and 3:
from __future__ import division
# (at top of module)
assert 3 / 2 == 1.5

# Python 2 and 3:
from past.utils import old_div
a = old_div(b, c)# always same as / on Py2
```

## Unicode (text) string literals

```python
# Python 2 only
s1 = 'The Zen of Python'
s2 = u'Python宗旨\n'
# Python 2 and 3
s1 = u'The Zen of Python'
s2 = u'Python宗旨\n'

#使用这种
# Python 2 and 3
from __future__ import unicode_literals
s1 = 'The Zen of Python'
s2 = 'Python宗旨\n'
```

## unicode

```python
# Python 2 only:
templates = [u"blog/blog_post_detail_%s.html" % unicode(slug)]

#使用这种
# Python 2 and 3: alternative 1
from builtins import str
templates = [u"blog/blog_post_detail_%s.html" % str(slug)]

# Python 2 and 3: alternative 2
from builtins import str as text
templates = [u"blog/blog_post_detail_%s.html" % text(slug)]
```

## Dictionaries

### Iterating through `dict` keys/values/items

####  Iterable dict keys:

```python
# Python 2 only:
for key in heights.iterkeys():
    ...
    
# Python 2 and 3:
for key in heights:
    ...
```

#### Iterable dict values:

```python
# Python 2 only:
for value in heights.itervalues():
    ...

# Idiomatic Python 3
for value in heights.values():    # extra memory overhead on Py2
    ...
    
#使用这种
# Python 2 and 3: option 1
from builtins import dict
heights = dict(Fred=175, Anne=166, Joe=192)
for key in heights.values():    # efficient on Py2 and Py3
    ...
```

#### Iterable dict items:

```python
# Python 2 only:
for (key, value) in heights.iteritems():
    ...
    
# Python 2 and 3: option 1
for (key, value) in heights.items():    # inefficient on Py2
    ...

#使用这种
# Python 2 and 3: option 3
from future.utils import iteritems
# or
from six import iteritems

for (key, value) in iteritems(heights):
    ...
```

#### dict values as a list:

```python
# Python 2 only:
heights = {'Fred': 175, 'Anne': 166, 'Joe': 192}
valuelist = heights.values()
assert isinstance(valuelist, list)

# Python 2 and 3: option 1
valuelist = list(heights.values())    # inefficient on Py2

# Python 2 and 3: option 2
from builtins import dict

heights = dict(Fred=175, Anne=166, Joe=192)
valuelist = list(heights.values())

#使用这种
# Python 2 and 3: option 4
from future.utils import itervalues
# or
from six import itervalues

valuelist = list(itervalues(heights))

```

#### dict items as a list:

```python
# Python 2 and 3: option 1
itemlist = list(heights.items())    # inefficient on Py2

# Python 2 and 3: option 2
from future.utils import listitems

itemlist = listitems(heights)

#使用这种
# Python 2 and 3: option 3
from future.utils import iteritems
# or
from six import iteritems

itemlist = list(iteritems(heights))
```

## Lists versus iterators

### xrange

```python
# Python 2 only:
for i in xrange(10**8):
    ...

# 使用这种
# Python 2 and 3: forward-compatible
from builtins import range
for i in range(10**8):
    ...
# 使用这种
# Python 2 and 3: backward-compatible
from past.builtins import xrange
for i in xrange(10**8):
    ...
```

### range

```python
# Python 2 only
mylist = range(5)
assert mylist == [0, 1, 2, 3, 4]

# Python 2 and 3: forward-compatible: option 1
mylist = list(range(5))            # copies memory on Py2
assert mylist == [0, 1, 2, 3, 4]

# Python 2 and 3: forward-compatible: option 2
from builtins import range

mylist = list(range(5))
assert mylist == [0, 1, 2, 3, 4]

# Python 2 and 3: option 3
from future.utils import lrange

mylist = lrange(5)
assert mylist == [0, 1, 2, 3, 4]

# Python 2 and 3: backward compatible
from past.builtins import range

mylist = range(5)
assert mylist == [0, 1, 2, 3, 4]
```

## reload

```python
# Python 2 only:
reload(mymodule)

# Python 2 and 3
from imp import reload
reload(mymodule)
```

