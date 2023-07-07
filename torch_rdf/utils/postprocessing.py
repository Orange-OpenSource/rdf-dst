import re

def clean_node(node):

# _:booking/0699e989  BEFORE ALL OF THIS FIND PATTERNS LIKE THIS IN A GIVEN NODE, REPLACE FORWARD AND CODE WITH PAD TOKEN
    node = node.replace('_:', '')
    node = node.replace(',', '')  # removing commas to facilitate RDF creation and splitting
    node = node.replace(';', '')  # removing semicolons to facilitate RDF creation and splitting
    underscoreRegex = re.compile(r"_")
    node = underscoreRegex.sub(' ', node).lower().strip()
    mask = ""  # Replace with empty mask to make it closer to NL
    randompatternRegex = re.compile(r'\/[a-zA-Z0-9]+')
    return randompatternRegex.sub(mask, node)

def clean_rdf(rdf):
    rdf = rdf.split(';')
    # rdfs must be 3 elements. Removing patterns that break this rule
    if len(rdf) != 3:
        return None
    return ';'.join([clean_node(node) for node in rdf])


def clean_row(row):
    new_rows = []
    for rdfs in row:
        post_process_rdf = clean_rdf(rdfs)
        if post_process_rdf:
            new_rows.append(post_process_rdf)

    return list(frozenset(new_rows))

def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    decoded_batch = [row.split(',') for row in decoded_batch]
    decoded_batch = map(clean_row, decoded_batch)
    return list(decoded_batch)
        
    #return [list(frozenset(clean_rdf(rdfs) for rdfs in row)) for row in decoded_batch]

