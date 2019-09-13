import re

def format_composition(composition):
    """ format str to ensure weights are explicate
    example: BaCu3 -> Ba1Cu3
    """
    subst = r"\g<1>1.0"
    composition = re.sub(r'[\d.]+', lambda x: str(float(x.group())), composition.rstrip())    
    composition = re.sub(r"([A-Z][a-z](?![0-9]))", subst, composition)
    composition = re.sub(r"([A-Z](?![0-9]|[a-z]))", subst, composition)
    composition = re.sub(r"([\)](?=[A-Z]))", subst, composition)
    composition = re.sub(r"([\)](?=\())", subst, composition)
    return composition


def parenthetic_contents(string):
    """
    Generate parenthesized contents in string as (level, contents, weight).
    """
    num_after_bracket = r"[^0-9.]"

    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            num = re.split(num_after_bracket, string[i+1:])[0] or 1
            yield {'value': [string[start + 1: i], float(num), False], 'level':len(stack)+1}

    yield {'value': [string,1, False], 'level':0}


def splitout_weights(composition):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    elements = []
    weights = []
    regex3 = r"(\d+\.\d+)|(\d+)"
    try:
        parsed = [j for j in re.split(regex3, composition) if j]
    except:
        print('parsed:', composition)
    elements += parsed[0::2]
    weights += parsed[1::2]
    weights = [float(w) for w in weights]
    return elements, weights


def update_weights(composition, weight):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    regex3 = r"(\d+\.\d+)|(\d+)"
    parsed = [j for j in re.split(regex3, composition) if j]
    elements = parsed[0::2]
    weights = [float(p)*weight for p in parsed[1::2]]
    new_comp = ''
    for m,n in zip(elements,weights):
        new_comp += m+f'{n:.2f}'
    return new_comp    


class Node(object):
    """ Node class for tree data structure """
    def __init__(self, parent, val=None):
        self.value = val
        self.parent = parent
        self.children = []

    def __repr__(self):
        return '<Node %s >' % (self.value,)


def build_tree(root, data):
    """ build a tree from ordered levelled data """
    for record in data:
        last = root
        for _ in range(record['level']):
            last = last.children[-1]
        last.children.append(Node(last, record['value']))


def print_tree(current, depth=0):
    """ print out the tree structure """
    for child in current.children:
        print('  ' * depth + '%r' % child)
        print_tree(child, depth + 1)


def reduce_tree(current):
    """ perform a post-order reduction on the tree """
    if not current:
        pass

    for child in current.children:
        reduce_tree(child)
        update_parent(child)


def update_parent(child):
    """ update the str for parent """
    input_str = child.value[2] or child.value[0]
    new_str = update_weights(input_str, child.value[1])
    pattern = re.escape("("+child.value[0]+")"+str(child.value[1]))
    old_str = child.parent.value[2] or child.parent.value[0]
    child.parent.value[2] = re.sub(pattern, new_str, old_str, 0)


def parse(string):
    # format the string to remove edge cases
    string = format_composition(string)
    # get nested bracket structure
    nested_levels = list(parenthetic_contents(string))
    if len(nested_levels) > 1:
        # reverse nested list
        nested_levels = nested_levels[::-1]
        # plant and grow the tree
        root = Node('root', ['None']*3)
        build_tree(root, nested_levels)
        # reduce the tree to get compositions
        reduce_tree(root)
        return splitout_weights(root.children[0].value[2])

    else:
        return splitout_weights(string)

