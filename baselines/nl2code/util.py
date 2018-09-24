def is_numeric(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()