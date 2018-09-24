from collections import namedtuple
import cPickle
from collections import Iterable, OrderedDict, defaultdict
from cStringIO import StringIO

from lang.util import typename

class ASTNode(object):
    def __init__(self, node_type, label=None, value=None, children=None):
        self.type = node_type
        self.label = label
        self.value = value

        if type(self) is not Rule:
            self.parent = None

        self.children = list()

        if children:
            if isinstance(children, Iterable):
                for child in children:
                    self.add_child(child)
            elif isinstance(children, ASTNode):
                self.add_child(children)
            else:
                raise AttributeError('Wrong type for child nodes')

        assert not (bool(children) and bool(value)), 'terminal node with a value cannot have children'

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_leaf

    @property
    def size(self):
        if self.is_leaf:
            return 1

        node_num = 1
        for child in self.children:
            node_num += child.size

        return node_num

    @property
    def nodes(self):
        """a generator that returns all the nodes"""

        yield self
        for child in self.children:
            for child_n in child.nodes:
                yield child_n

    @property
    def as_type_node(self):
        """return an ASTNode with type information only"""
        return ASTNode(self.type)

    def __repr__(self):
        repr_str = ''
        # if not self.is_leaf:
        repr_str += '('

        repr_str += typename(self.type)

        if self.label is not None:
            repr_str += '{%s}' % self.label

        if self.value is not None:
            repr_str += '{val=%s}' % self.value

        # if not self.is_leaf:
        for child in self.children:
            repr_str += ' ' + child.__repr__()
        repr_str += ')'

        return repr_str

    def __hash__(self):
        code = hash(self.type)
        if self.label is not None:
            code = code * 37 + hash(self.label)
        if self.value is not None:
            code = code * 37 + hash(self.value)
        for child in self.children:
            code = code * 37 + hash(child)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if hash(self) != hash(other):
            return False

        if self.type != other.type:
            return False

        if self.label != other.label:
            return False

        if self.value != other.value:
            return False

        if len(self.children) != len(other.children):
            return False

        for i in xrange(len(self.children)):
            if self.children[i] != other.children[i]:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, child_type):
        return next(iter([c for c in self.children if c.type == child_type]))

    def __delitem__(self, child_type):
        tgt_child = [c for c in self.children if c.type == child_type]
        if tgt_child:
            assert len(tgt_child) == 1, 'unsafe deletion for more than one children'
            tgt_child = tgt_child[0]
            self.children.remove(tgt_child)
        else:
            raise KeyError

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_child_id(self, child):
        for i, _child in enumerate(self.children):
            if child == _child:
                return i

        raise KeyError

    def pretty_print(self):
        sb = StringIO()
        new_line = False
        self.pretty_print_helper(sb, 0, new_line)
        return sb.getvalue()

    def pretty_print_helper(self, sb, depth, new_line=False):
        if new_line:
            sb.write('\n')
            for i in xrange(depth): sb.write(' ')

        sb.write('(')
        sb.write(typename(self.type))
        if self.label is not None:
            sb.write('{%s}' % self.label)

        if self.value is not None:
            sb.write('{val=%s}' % self.value)

        if len(self.children) == 0:
            sb.write(')')
            return

        sb.write(' ')
        new_line = True
        for child in self.children:
            child.pretty_print_helper(sb, depth + 2, new_line)

        sb.write('\n')
        for i in xrange(depth): sb.write(' ')
        sb.write(')')

    def get_leaves(self):
        if self.is_leaf:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())

        return leaves

    def to_rule(self, include_value=False):
        """
        transform the current AST node to a production rule
        """
        rule = Rule(self.type)
        for c in self.children:
            val = c.value if include_value else None
            child = ASTNode(c.type, c.label, val)
            rule.add_child(child)

        return rule

    def get_productions(self, include_value_node=False):
        """
        get the depth-first, left-to-right sequence of rule applications
        returns a list of production rules and a map to their parent rules
        attention: node value is not included in child nodes
        """
        rule_list = list()
        rule_parents = OrderedDict()
        node_rule_map = dict()
        s = list()
        s.append(self)
        rule_num = 0

        while len(s) > 0:
            node = s.pop()
            for child in reversed(node.children):
                if not child.is_leaf:
                    s.append(child)
                elif include_value_node:
                    if child.value is not None:
                        s.append(child)

            # only non-terminals and terminal nodes holding values
            # can form a production rule
            if node.children or node.value is not None:
                rule = Rule(node.type)
                if include_value_node:
                    rule.value = node.value

                for c in node.children:
                    val = None
                    child = ASTNode(c.type, c.label, val)
                    rule.add_child(child)

                rule_list.append(rule)
                if node.parent:
                    child_id = node.parent.get_child_id(node)
                    parent_rule = node_rule_map[node.parent]
                    rule_parents[(rule_num, rule)] = (parent_rule, child_id)
                else:
                    rule_parents[(rule_num, rule)] = (None, -1)
                rule_num += 1

                node_rule_map[node] = rule

        return rule_list, rule_parents

    def copy(self):
        # if not hasattr(self, '_dump'):
        #     dump = cPickle.dumps(self, -1)
        #     setattr(self, '_dump', dump)
        #
        #     return cPickle.loads(dump)
        #
        # return cPickle.loads(self._dump)

        new_tree = ASTNode(self.type, self.label, self.value)
        if self.is_leaf:
            return new_tree

        for child in self.children:
            new_tree.add_child(child.copy())

        return new_tree


class DecodeTree(ASTNode):
    def __init__(self, node_type, label=None, value=None, children=None, t=-1):
        super(DecodeTree, self).__init__(node_type, label, value, children)

        # record the time step when this subtree is created from a rule application
        self.t = t
        # record the ApplyRule action that is used to expand the current node
        self.applied_rule = None

    def copy(self):
        new_tree = DecodeTree(self.type, self.label, value=self.value, t=self.t)
        new_tree.applied_rule = self.applied_rule
        if self.is_leaf:
            return new_tree

        for child in self.children:
            new_tree.add_child(child.copy())

        return new_tree


class Rule(ASTNode):
    def __init__(self, *args, **kwargs):
        super(Rule, self).__init__(*args, **kwargs)

        assert self.value is None and self.label is None, 'Rule LHS cannot have values or labels'

    @property
    def parent(self):
        return self.as_type_node

    def __repr__(self):
        parent = typename(self.type)

        if self.label is not None:
            parent += '{%s}' % self.label

        if self.value is not None:
            parent += '{val=%s}' % self.value

        return '%s -> %s' % (parent, ', '.join([repr(c) for c in self.children]))


if __name__ == '__main__':
    import ast
    t1 = ASTNode('root', children=[
        ASTNode(str, 'a1_label', children=[ASTNode(int, children=[ASTNode('a21', value=123)]),
                                            ASTNode(ast.NodeTransformer, children=[ASTNode('a21', value='hahaha')])]
                ),
        ASTNode('a2', children=[ASTNode('a21', value='asdf')])
    ])

    t2 = ASTNode('root', children=[
        ASTNode(str, 'a1_label', children=[ASTNode(int, children=[ASTNode('a21', value=123)]),
                                           ASTNode(ast.NodeTransformer, children=[ASTNode('a21', value='hahaha')])]
                ),
        ASTNode('a2', children=[ASTNode('a21', value='asdf')])
    ])

    print t1 == t2


    a, b = t1.get_productions(include_value_node=True)

    # t = ASTNode('root', children=ASTNode('sdf'))

    print t1.__repr__()
    print t1.pretty_print()