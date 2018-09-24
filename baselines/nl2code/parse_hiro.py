import ast
import sys
import re
import inspect

def typename(x):
    return type(x).__name__

def escape(text):
    text = text \
        .replace('"', '`') \
        .replace('\'', '`') \
        .replace(' ', '-SP-') \
        .replace('\t', '-TAB-') \
        .replace('\n', '-NL-') \
        .replace('(', '-LRB-') \
        .replace(')', '-RRB-') \
        .replace('|', '-BAR-')
    return repr(text)[1:-1] if text else '-NONE-'

def makestr(node):

    #if node is None or isinstance(node, ast.Pass):
    #    return ''

    if isinstance(node, ast.AST):
        n = 0
        nodename = typename(node)
        s = '(' + nodename
        for chname, chval in ast.iter_fields(node):
            chstr = makestr(chval)
            if chstr:
                s += ' (' + chname + ' ' + chstr + ')'
                n += 1
        if not n:
            s += ' -' + nodename + '-' # (Foo) -> (Foo -Foo-)
        s += ')'
        return s

    elif isinstance(node, list):
        n = 0
        s = '(list'
        for ch in node:
            chstr = makestr(ch)
            if chstr:
                s += ' ' + chstr
                n += 1
        s += ')'
        return s if n else ''

    elif isinstance(node, str):
        return '(str ' + escape(node) + ')'

    elif isinstance(node, bytes):
        return '(bytes ' + escape(str(node)) + ')'

    else:
        return '(' + typename(node) + ' ' + str(node) + ')'


def main():
    p_elif = re.compile(r'^elif\s?')
    p_else = re.compile(r'^else\s?')
    p_try = re.compile(r'^try\s?')
    p_except = re.compile(r'^except\s?')
    p_finally = re.compile(r'^finally\s?')
    p_decorator = re.compile(r'^@.*')

    for l in ["""val = Header ( val , encoding ) . encode ( )"""]:  # val = ', ' . join ( sanitize_address ( addr , encoding )  for addr in getaddresses ( ( val , ) ) )
        l = l.strip()
        if not l:
            print()
            sys.stdout.flush()
            continue

        if p_elif.match(l): l = 'if True: pass\n' + l
        if p_else.match(l): l = 'if True: pass\n' + l

        if p_try.match(l): l = l + 'pass\nexcept: pass'
        elif p_except.match(l): l = 'try: pass\n' + l
        elif p_finally.match(l): l = 'try: pass\n' + l

        if p_decorator.match(l): l = l + '\ndef dummy(): pass'
        if l[-1] == ':': l = l + 'pass'

        parse = ast.parse(l)
        parse = parse.body[0]
        dump = makestr(parse)
        print(dump)
        sys.stdout.flush()

if __name__ == '__main__':
    main()
