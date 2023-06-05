import re

def clean_node(node):

    node = node.replace('_:', '')
    node = node.replace(',', '')  # removing commas to facilitate RDF creation and splitting
    underscoreRegex = re.compile(r"_")
    node = underscoreRegex.sub(' ', node)
    return node.lower().strip()

def clean_rdf(rdf):
    return '|'.join([clean_node(node) for node in rdf.split('|')])


def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    decoded_batch = [row.split(',') for row in decoded_batch]
    return [frozenset(clean_rdf(rdfs) for rdfs in row) for row in decoded_batch]

