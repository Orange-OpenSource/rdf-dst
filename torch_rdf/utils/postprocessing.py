import re

def clean_node(node):

# _:booking/0699e989  BEFORE ALL OF THIS FIND PATTERNS LIKE THIS IN A GIVEN NODE, REPLACE FORWARD AND CODE WITH PAD TOKEN
    node = node.replace('_:', '')
    node = node.replace(',', '')  # removing commas to facilitate RDF creation and splitting
    underscoreRegex = re.compile(r"_")
    node = underscoreRegex.sub(' ', node).lower().strip()
    mask = ""  # Replace with empty mask to make it closer to NL
    randompatternRegex = re.compile(r'\/[a-zA-Z0-9]+')
    return randompatternRegex.sub(mask, node)

def clean_rdf(rdf):
    return '|'.join([clean_node(node) for node in rdf.split('|')])


def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    decoded_batch = [row.split(',') for row in decoded_batch]
    return [list(frozenset(clean_rdf(rdfs) for rdfs in row)) for row in decoded_batch]

