# -*- coding: UTF-8 -*-

from astnode import ASTNode
from lang.sql.grammar import type_str_to_type
from lang.py.parse import parse
from collections import Counter
import re


def extract_unary_closure_helper(parse_tree, unary_link, last_node):
    if parse_tree.is_leaf:
        if unary_link and unary_link.size > 2:
            return [unary_link]
        else:
            return []
    elif len(parse_tree.children) > 1:
        unary_links = []
        if unary_link and unary_link.size > 2:
            unary_links.append(unary_link)
        for child in parse_tree.children:
            new_node = ASTNode(child.type)
            child_unary_links = extract_unary_closure_helper(child, new_node, new_node)
            unary_links.extend(child_unary_links)

        return unary_links
    else:  # has a single child
        child = parse_tree.children[0]
        new_node = ASTNode(child.type, label=child.label)
        last_node.add_child(new_node)
        last_node = new_node

        return extract_unary_closure_helper(child, unary_link, last_node)


def extract_unary_closure(parse_tree):
    root_node_copy = ASTNode(parse_tree.type)
    unary_links = extract_unary_closure_helper(parse_tree, root_node_copy, root_node_copy)

    return unary_links


def get_unary_links():
    # data_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/card_datasets/hearthstone/all_hs.out'
    data_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'
    parse_trees = []
    unary_links_counter = Counter()

    for line in open(data_file):
        code = line.replace('§', '\n').strip()
        parse_tree = parse(code)
        parse_trees.append(parse_tree)

        example_unary_links = extract_unary_closure(parse_tree)
        for link in example_unary_links:
            unary_links_counter[link] += 1

    ranked_links = sorted(unary_links_counter, key=unary_links_counter.get, reverse=True)
    for link in ranked_links:
        print str(link) + ' ||| ' + str(unary_links_counter[link])

    unary_links = ranked_links[:20]
    unary_closures = []
    for link in unary_links:
        unary_closures.append(unary_link_to_closure(link))

    unary_closures = zip(unary_links, unary_closures)

    node_nums = rule_nums = 0.
    for parse_tree in parse_trees:
        original_parse_tree = parse_tree.copy()
        for link, closure in unary_closures:
            apply_unary_closure(parse_tree, closure, link)

        # assert original_parse_tree != parse_tree
        compressed_ast_to_normal(parse_tree)
        assert original_parse_tree == parse_tree

        rules, _ = parse_tree.get_productions()
        rule_nums += len(rules)
        node_nums += len(list(parse_tree.nodes))

    print '**** after applying unary closures ****'
    print 'avg. nums of nodes: %f' % (node_nums / len(parse_trees))
    print 'avg. nums of rules: %f' % (rule_nums / len(parse_trees))



def get_top_unary_closures(parse_trees, k=20, freq=50):
    unary_links_counter = Counter()
    for parse_tree in parse_trees:
        example_unary_links = extract_unary_closure(parse_tree)
        for link in example_unary_links:
            unary_links_counter[link] += 1

    ranked_links = sorted(unary_links_counter, key=unary_links_counter.get, reverse=True)
    if k:
        print 'rank cut off: %d' % k
        unary_links = ranked_links[:k]
    else:
        print 'freq cut off: %d' % freq
        unary_links = sorted([l for l in unary_links_counter if unary_links_counter[l] >= freq], key=unary_links_counter.get, reverse=True)

    unary_closures = []
    for link in unary_links:
        unary_closures.append(unary_link_to_closure(link))

    unary_closures = zip(unary_links, unary_closures)

    for link, closure in unary_closures:
        print 'link: %s ||| closure: %s ||| freq: %d' % (link, closure, unary_links_counter[link])

    return unary_closures


def apply_unary_closures(parse_tree, unary_closures):
    unary_closures = sorted(unary_closures, key=lambda x: x[0].size, reverse=True)
    original_parse_tree = parse_tree.copy()

    # apply all unary closures
    for link, closure in unary_closures:
        apply_unary_closure(parse_tree, closure, link)

    new_tree_copy = parse_tree.copy()
    compressed_ast_to_normal(new_tree_copy)
    assert original_parse_tree == new_tree_copy


rule_regex = re.compile(r'(?P<parent>.*?) -> \((?P<child>.*?)(\{(?P<clabel>.*?)\})?\)')
def compressed_ast_to_normal(parse_tree):
    if parse_tree.label and '@' in parse_tree.label and '$' in parse_tree.label:
        label = parse_tree.label
        label = label.replace('$', ' ')
        rule_reprs = label.split('@')

        intermediate_nodes = []
        first_node = last_node = None
        for rule_repr in rule_reprs:
            m = rule_regex.match(rule_repr)
            p = m.group('parent')
            c = m.group('child')
            cl = m.group('clabel')

            # print(p)
            # p_type = type_str_to_type(p)
            c_type = type_str_to_type(c)

            node = ASTNode(c_type, label=cl)
            if last_node:
                last_node.add_child(node)
            if not first_node:
                first_node = node

            last_node = node
            intermediate_nodes.append(node)

        last_node.value = parse_tree.value
        for child in parse_tree.children:
            last_node.add_child(child)
            compressed_ast_to_normal(child)


        parent_node = parse_tree.parent
        assert len(parent_node.children) == 1
        del parent_node.children[0]
        parent_node.add_child(first_node)
        # return first_node
    else:
        new_child_trees = []
        for child in parse_tree.children[:]:
            compressed_ast_to_normal(child)
        #     new_child_trees.append(new_child_tree)
        # del parse_tree.children[:]
        # for child_tree in new_child_trees:
        #     parse_tree.add_child(child_tree)
        #
        # return parse_tree


def match_sub_tree(parse_tree, cur_match_node, is_root=False):
    cur_level_match = False
    if parse_tree.type == cur_match_node.type and (len(parse_tree.children) == 1 or cur_match_node.is_leaf) and \
            (is_root or parse_tree.label == cur_match_node.label):
        cur_level_match = True

    if cur_level_match:
        if cur_match_node.is_leaf:
            return parse_tree

        last_node = match_sub_tree(parse_tree.children[0], cur_match_node.children[0])
        return last_node
    else:
        return None


def find(parse_tree, sub_tree):
    match_results = []
    last_node = match_sub_tree(parse_tree, sub_tree, True)

    if last_node:
        match_results.append((parse_tree, last_node))

    for child in parse_tree.children:
        child_match_results = find(child, sub_tree)
        match_results.extend(child_match_results)

    return match_results


def apply_unary_closure(parse_tree, unary_closure, unary_link):
    match_results = find(parse_tree, unary_link)
    for first_node, last_node in match_results:
        closure_copy = unary_closure.copy()

        leaf = closure_copy.get_leaves()[0]
        leaf.value = last_node.value
        for child in last_node.children:
            leaf.add_child(child)

        new_node = closure_copy.children[0]
        first_node.children.remove(first_node.children[0])
        first_node.add_child(new_node)


def unary_link_to_closure(unary_link):
    closure = ASTNode(unary_link.type)
    last_node = unary_link.get_leaves()[0]
    closure_child = ASTNode(last_node.type)
    prod, _ = unary_link.get_productions()
    closure_child_label = '@'.join(str(rule).replace(' ', '$') for rule in prod)
    closure_child.label = closure_child_label

    closure.add_child(closure_child)

    return closure

if __name__ == '__main__':
#     code = """
# class Demonwrath(SpellCard):
#     def __init__(self):
#         super().__init__("Demonwrath", 3, CHARACTER_CLASS.WARLOCK, CARD_RARITY.RARE)
#
#     def use(self, player, game):
#         super().use(player, game)
#         targets = copy.copy(game.other_player.minions)
#         targets.extend(game.current_player.minions)
#         for minion in targets:
#             if minion.card.minion_type is not MINION_TYPE.DEMON:
#                 minion.damage(player.effective_spell_damage(2), self)
#     """
#     parse_tree = parse(code)
#     original_parse_tree = parse_tree.copy()
#     unary_links = extract_unary_closure(parse_tree)
#
#     for link in unary_links:
#         closure = unary_link_to_closure(link)
#         print closure, link
#         apply_unary_closure(parse_tree, closure, link)
#
#     compressed_ast_to_normal(parse_tree)
#     print parse_tree
#     print original_parse_tree
#     print parse_tree == original_parse_tree
    get_unary_links()