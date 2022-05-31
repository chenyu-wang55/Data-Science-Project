
def is_blank(s):
    if s is None or s.strip() == '':
        return True
    return False


def is_not_blank(s):
    return not is_blank(s)


