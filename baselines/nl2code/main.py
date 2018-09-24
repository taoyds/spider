import ast
import re

from astnode import *

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')


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


def typename(x):
    return type(x).__name__


def get_tree_str_repr(node):
    treeStr = ''
    if type(node) == list:
        for n in node:
            treeStr += get_tree_str_repr(n)

        return treeStr

    node_name = str(type(node))
    begin = node_name.find('ast.') + len('ast.')
    end = node_name.rfind('\'')
    node_name = node_name[begin: end]
    treeStr = '(' + node_name + ' '
    for field_name in node._fields:
        field = getattr(node, field_name)
        if hasattr(field, '_fields') and len(field._fields) == 0:
            continue
        if field:
            if type(field) == list:
                fieldRepr = get_tree_str_repr(field)
                fieldRepr = '(' + field_name + ' ' + fieldRepr + ') '
            elif type(field) == str or type(field) == int:
                fieldRepr = '(' + field_name + ' ' + str(field) + ') '
            else:
                fieldRepr = get_tree_str_repr(field)
                fieldRepr = '(' + field_name + ' ' + fieldRepr + ') '

            treeStr += fieldRepr
    treeStr += ') '

    return treeStr


def get_tree(node):

    if isinstance(node, str):
        node_name = escape(node)
    elif isinstance(node, int):
        node_name = node
    else:
        node_name = typename(node)

    tree = ASTNode(node_name)

    if not isinstance(node, ast.AST):
        return tree

    for field_name, field in ast.iter_fields(node):
        # omit empty fields
        if isinstance(field, ast.AST):
            if len(field._fields) == 0:
                continue

            child = get_tree(field)

            tree.children.append(ASTNode(field_name, child))
        elif isinstance(field, str):
            field_val = escape(field)
            child = ASTNode(field_name, ASTNode(field_val))

            tree.children.append(child)
        elif isinstance(field, int):
            child = ASTNode(field_name, ASTNode(field))

            tree.children.append(child)
        elif isinstance(field, list) and field:
            child = ASTNode(field_name)

            for n in field:
                child.children.append(get_tree(n))

            tree.children.append(child)

    return tree


def parse(code):
    if p_elif.match(code): code = 'if True: pass\n' + code
    if p_else.match(code): code = 'if True: pass\n' + code

    if p_try.match(code): code = code + 'pass\nexcept: pass'
    elif p_except.match(code): code = 'try: pass\n' + code
    elif p_finally.match(code): code = 'try: pass\n' + code

    if p_decorator.match(code): code = code + '\ndef dummy(): pass'
    if code[-1] == ':': code = code + 'pass'

    root_node = ast.parse(code)

    tree = get_tree(root_node.body[0])

    return tree


def parse_django(code_file):
    line_num = 0
    error_num = 0
    parse_trees = []
    for line in open(code_file):
        code = line.strip()
        try:
            parse_tree = parse(code)
            # rule_list = parse_tree.get_rule_list(include_leaf=False)
            parse_trees.append(parse_tree)
            print parse_tree
        except Exception as e:
            error_num += 1
            pass
            # print e

        line_num += 1

    print 'total line of code: %d' % line_num
    print 'error num: %d' % error_num

    assert error_num == 0

    grammar = get_grammar(parse_trees)

    with open('grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.parent + ' -> ' + ', '.join(rule.children)
            f.write(str + '\n')

    return grammar, parse_trees


if __name__ == '__main__':
#     node = ast.parse('''
# # for i in range(1, 100):
# #  sum = sum + i
# #
# # sorted(arr, reverse=True)
# # sorted(my_dict, key=lambda x: my_dict[x], reverse=True)
# # m = dict ( zip ( new_keys , keys ) )
# # for f in sorted ( os . listdir ( self . path ) ) :
# #     pass
# for f in sorted ( os . listdir ( self . path ) ) : pass
# ''')
    # print ast.dump(node, annotate_fields=False)
    # print get_tree_str_repr(node)
    # print parse('for f in sorted ( os . listdir ( self . path ) ) : sum = sum + 1; sum = "(hello there)" ')
    # print parse('global _standard_context_processors')

    parse_django()


