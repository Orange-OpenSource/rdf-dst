import re

def clean_node(node):

    node = node.replace('_:', '')
    node = node.replace(',', '')  # removing commas to facilitate RDF creation and splitting
    underscoreRegex = re.compile(r"_")
    node = underscoreRegex.sub(' ', node)
    return node.lower().strip()

def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    regexSplit = re.compile(r"(?<!\s),(?!\s)")
    decoded_batch = [regexSplit.split(row) for row in decoded_batch]
    decoded_batch = [[clean_node(word) for word in rdfs] for rdfs in decoded_batch]

    # they are frozensets so we can play around with them in evaluation
    clean_rdfs = [list(tuple([frozenset(rdfs[i:i+3]) for i in range(0, len(rdfs), 3)])) for rdfs in decoded_batch]

    return clean_rdfs
