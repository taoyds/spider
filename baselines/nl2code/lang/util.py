# x is a type
def typename(x):
    if isinstance(x, basestring):
        return x
    return x.__name__

def escape(text):
    text = text \
        .replace('"', '-``-') \
        .replace('\'', '-`-') \
        .replace(' ', '-SP-') \
        .replace('\t', '-TAB-') \
        .replace('\n', '-NL-') \
        .replace('\r', '-NL2-') \
        .replace('(', '-LRB-') \
        .replace(')', '-RRB-') \
        .replace('|', '-BAR-')

    if text is None:
        return '-NONE-'
    elif text == '':
        return '-EMPTY-'

    return text

def unescape(text):
    if text == '-NONE-':
        return None

    text = text \
        .replace('-``-', '"') \
        .replace('-`-', '\'') \
        .replace('-SP-', ' ') \
        .replace('-TAB-', '\t') \
        .replace('-NL-', '\n') \
        .replace('-NL2-', '\r') \
        .replace('-LRB-', '(') \
        .replace('-RRB-', ')') \
        .replace('-BAR-', '|') \
        .replace('-EMPTY-', '')

    return text