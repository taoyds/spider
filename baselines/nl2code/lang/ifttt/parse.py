from astnode import ASTNode

def ifttt_ast_to_parse_tree_helper(s, offset):
    """
    adapted from ifttt codebase
    """
    if s[offset] != '(':
        raise RuntimeError('malformed string: node did not start with open paren at position ' + offset)

    offset += 1
    # extract node name(type)
    name = ''
    if s[offset] == '\"':
        offset += 1
        while s[offset] != '\"':
            if s[offset] == '\\':
                offset += 1
            name += s[offset]
            offset += 1
        offset += 1
    else:
        while s[offset] != ' ' and s[offset] != ')':
            name += s[offset]
            offset += 1

    node = ASTNode(name)
    while True:
        if s[offset] == ')':
            offset += 1
            return node, offset
        if s[offset] != ' ':
            raise RuntimeError('malformed string: node should have either had a '
                               'close paren or a space at position ' + offset)
        offset += 1
        child_node, offset = ifttt_ast_to_parse_tree_helper(s, offset)
        node.add_child(child_node)


def ifttt_ast_to_parse_tree(s, attach_func_to_channel=True):
    parse_tree, _ = ifttt_ast_to_parse_tree_helper(s, 0)
    parse_tree = strip_params(parse_tree)

    if attach_func_to_channel:
        parse_tree = attach_function_to_channel(parse_tree)

    return parse_tree


def strip_params(parse_tree):
    if parse_tree.type == 'PARAMS':
        raise RuntimeError('should not go to here!')

    parse_tree.children = [c for c in parse_tree.children if c.type != 'PARAMS' and c.type != 'OUTPARAMS']
    for i, child in enumerate(parse_tree.children):
        parse_tree.children[i] = strip_params(child)

    return parse_tree


def attach_function_to_channel(parse_tree):
    trigger_func = parse_tree['TRIGGER']['FUNC'].children
    assert len(trigger_func) == 1

    trigger_func = trigger_func[0]
    parse_tree['TRIGGER'].children[0].add_child(trigger_func)

    del parse_tree['TRIGGER']['FUNC']

    action_func = parse_tree['ACTION']['FUNC'].children
    assert len(action_func) == 1

    action_func = action_func[0]
    parse_tree['ACTION'].children[0].add_child(action_func)

    del parse_tree['ACTION']['FUNC']

    return parse_tree


if __name__ == '__main__':
    tree_code = """(ROOT (IF) (TRIGGER (Instagram) (FUNC (Any_new_photo_by_you) (PARAMS))) (THEN) (ACTION (Dropbox) (FUNC (Add_file_from_URL) (PARAMS (File_URL ({{Caption}})) (File_name ("")) (Dropbox_folder_path (ifttt/instagram))))))"""
    parse_tree = ifttt_ast_to_parse_tree(tree_code)

    print parse_tree
    print strip_params(parse_tree)
    print attach_function_to_channel(parse_tree)