"""
Python grammar and typing system
"""
import ast
import inspect
import astor

from lang.grammar import Grammar

PY_AST_NODE_FIELDS = {
    'FunctionDef': {
        'name': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        }
    },
    'ClassDef': {
        'name': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'bases': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        }
    },
    'Return': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Delete': {
        'targets': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Assign': {
        'targets': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'AugAssign': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'op': {
            'type': ast.operator,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'Print': {
        'dest': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'nl': {
            'type': bool,
            'is_list': False,
            'is_optional': False
        }
    },
    'For': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'While': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'If': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'With': {
        'context_expr': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'optional_vars': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'Raise': {
        'type': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'inst': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'tback': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'TryExcept': {
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'handlers': {
            'type': ast.excepthandler,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'TryFinally': {
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'finalbody': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'Assert': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'msg': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Import': {
        'names': {
            'type': ast.alias,
            'is_list': True,
            'is_optional': False
        }
    },
    'ImportFrom': {
        'module': {
            'type': str,
            'is_list': False,
            'is_optional': True
        },
        'names': {
            'type': ast.alias,
            'is_list': True,
            'is_optional': False
        },
        'level': {
            'type': int,
            'is_list': False,
            'is_optional': True
        }
    },
    'Exec': {
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'globals': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'locals': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Global': {
        'names': {
            'type': str,
            'is_list': True,
            'is_optional': False
        },
    },
    'Expr': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'BoolOp': {
        'op': {
            'type': ast.boolop,
            'is_list': False,
            'is_optional': False
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'BinOp': {
        'left': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'op': {
            'type': ast.operator,
            'is_list': False,
            'is_optional': False
        },
        'right': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'UnaryOp': {
        'op': {
            'type': ast.unaryop,
            'is_list': False,
            'is_optional': False
        },
        'operand': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Lambda': {
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'IfExp': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'orelse': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Dict': {
        'keys': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Set': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'ListComp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'SetComp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'DictComp': {
        'key': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'GeneratorExp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'Yield': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Compare': {
        'left': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'ops': {
            'type': ast.cmpop,
            'is_list': True,
            'is_optional': False
        },
        'comparators': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Call': {
        'func': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'keywords': {
            'type': ast.keyword,
            'is_list': True,
            'is_optional': False
        },
        'starargs': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'kwargs': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Repr': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'Num': {
        'n': {
            'type': object,  #FIXME: should be float or int?
            'is_list': False,
            'is_optional': False
        }
    },
    'Str': {
        's': {
            'type': str,
            'is_list': False,
            'is_optional': False
        }
    },
    'Attribute': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'attr': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Subscript': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'slice': {
            'type': ast.slice,
            'is_list': False,
            'is_optional': False
        },
    },
    'Name': {
        'id': {
            'type': str,
            'is_list': False,
            'is_optional': False
        }
    },
    'List': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Tuple': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'ExceptHandler': {
        'type': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'name': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'arguments': {
        'args': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'vararg': {
            'type': str,
            'is_list': False,
            'is_optional': True
        },
        'kwarg': {
            'type': str,
            'is_list': False,
            'is_optional': True
        },
        'defaults': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'comprehension': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'ifs': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'keyword': {
        'arg': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'alias': {
        'name': {
            'type': str,
            'is_list': False,
            'is_optional': False
        },
        'asname': {
            'type': str,
            'is_list': False,
            'is_optional': True
        }
    },
    'Slice': {
        'lower': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'upper': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'step': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'ExtSlice': {
        'dims': {
            'type': ast.slice,
            'is_list': True,
            'is_optional': False
        }
    },
    'Index': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    }
}

NODE_FIELD_BLACK_LIST = {'ctx'}

TERMINAL_AST_TYPES = {
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Add,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Div,
    ast.FloorDiv,
    ast.LShift,
    ast.Mod,
    ast.Mult,
    ast.Pow,
    ast.Sub,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.Is,
    ast.IsNot,
    ast.Lt,
    ast.LtE,
    ast.NotEq,
    ast.NotIn,
    ast.Not,
    ast.USub
}


def is_builtin_type(x):
    return x == str or x == int or x == float or x == bool or x == object or x == 'identifier'


def is_terminal_ast_type(x):
    if inspect.isclass(x) and x in TERMINAL_AST_TYPES:
        return True

    return False


# def is_terminal_type(x):
#     if is_builtin_type(x):
#         return True
#
#     if x == 'epsilon':
#         return True
#
#     if inspect.isclass(x) and (issubclass(x, ast.Pass) or issubclass(x, ast.Raise) or issubclass(x, ast.Break)
#                                or issubclass(x, ast.Continue)
#                                or issubclass(x, ast.Return)
#                                or issubclass(x, ast.operator) or issubclass(x, ast.boolop)
#                                or issubclass(x, ast.Ellipsis) or issubclass(x, ast.unaryop)
#                                or issubclass(x, ast.cmpop)):
#         return True
#
#     return False


# class Node:
#     def __init__(self, node_type, label):
#         self.type = node_type
#         self.label = label
#
#     @property
#     def is_preterminal(self):
#         return is_terminal_type(self.type)
#
#     def __eq__(self, other):
#         return self.type == other.type and self.label == other.label
#
#     def __hash__(self):
#         return typename(self.type).__hash__() ^ self.label.__hash__()
#
#     def __repr__(self):
#         repr_str = typename(self.type)
#         if self.label:
#             repr_str += '{%s}' % self.label
#         return repr_str
#
#
# class TypedRule:
#     def __init__(self, parent, children, tree=None):
#         self.parent = parent
#         if isinstance(children, list) or isinstance(children, tuple):
#             self.children = tuple(children)
#         else:
#             self.children = (children, )
#
#         # tree property is not incorporated in eq, hash
#         self.tree = tree
#
#     # @property
#     # def is_terminal_rule(self):
#     #     return is_terminal_type(self.parent.type)
#
#     def __eq__(self, other):
#         return self.parent == other.parent and self.children == other.children
#
#     def __hash__(self):
#         return self.parent.__hash__() ^ self.children.__hash__()
#
#     def __repr__(self):
#         return '%s -> %s' % (self.parent, ', '.join([c.__repr__() for c in self.children]))


def type_str_to_type(type_str):
    if type_str.endswith('*') or type_str == 'root' or type_str == 'epsilon':
        return type_str
    else:
        try:
            type_obj = eval(type_str)
            if is_builtin_type(type_obj):
                return type_obj
        except:
            pass

        try:
            type_obj = eval('ast.' + type_str)
            return type_obj
        except:
            raise RuntimeError('unidentified type string: %s' % type_str)


def is_compositional_leaf(node):
    is_leaf = True

    for field_name, field_value in ast.iter_fields(node):
        if field_name in NODE_FIELD_BLACK_LIST:
            continue

        if field_value is None:
            is_leaf &= True
        elif isinstance(field_value, list) and len(field_value) == 0:
            is_leaf &= True
        else:
            is_leaf &= False
    return is_leaf


class PythonGrammar(Grammar):
    def __init__(self, rules):
        super(PythonGrammar, self).__init__(rules)

    def is_value_node(self, node):
        return is_builtin_type(node.type)
