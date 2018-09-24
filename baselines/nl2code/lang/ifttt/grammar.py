from lang.grammar import Grammar

class IFTTTGrammar(Grammar):
    def __init__(self, rules):
        super(IFTTTGrammar, self).__init__(rules)

    def is_value_node(self, node):
        return False