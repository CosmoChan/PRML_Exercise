# 兼容Python2和Python3的代码

## print 

```python
from __future__ import print_function# (at top of module)
age = 20
print('Hello, I am %d years old'%age)
# Python 2 only:
#print 'Hello',
print('Hello', end='')
```

## Division

```python
from __future__ import division# (at top of module)
assert 3 / 2 == 1.5
```

## Unicode (text) string literals

```python
from __future__ import unicode_literals
s2 = '可以直接打中文不需要加u'
```

## unicode

```python
from builtins import str
templates = [u"blog/blog_post_detail_%s.html" % str(slug)]
```

