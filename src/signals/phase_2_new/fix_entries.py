import os, glob, re
for f in glob.glob('*/signal.py'):
    s=open(f).read()
    orig=s
    s=re.sub(r'(^|[^\w.])entry = yp([^\w]|$)', r'\1entry = kwargs.get("yes_ask", yp)\2', s, flags=re.MULTILINE)
    s=re.sub(r'(^|[^\w.])entry = np_val([^\w]|$)', r'\1entry = kwargs.get("no_ask", np_val)\2', s, flags=re.MULTILINE)
    s=re.sub(r'(^|[^\w.])entry_price = yp([^\w]|$)', r'\1entry_price = kwargs.get("yes_ask", yp)\2', s, flags=re.MULTILINE)
    s=re.sub(r'(^|[^\w.])entry_price = np_val([^\w]|$)', r'\1entry_price = kwargs.get("no_ask", np_val)\2', s, flags=re.MULTILINE)
    if s!=orig:
        open(f,'w').write(s)
        print('fixed',f)
