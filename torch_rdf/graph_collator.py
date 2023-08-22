from dataclasses import dataclass
from utils.predata_collate import PreDataCollator
import re

@dataclass
class GraphCollator(PreDataCollator):

    def __init__(self, tokenizer, dataset, complex_repr=False):
        #data_collation = {"multiwoz": self.process_multiwoz}
        #self.data_collation = data_collation[dataset]
        self.complex_rp = complex_repr
        self.tokenizer = tokenizer
        self.dataset_type = dataset
        self.nodes_1 = set()  # 5
        self.edges = set()  # 33


    def __call__(self, batch):

        new_states = []
        graph_states = []
        for states in batch['states']:
            clean_states = self.clean_rdf_state(states)
            new_states.append(clean_states)
            # TODO RETURN NEW GRAPH STATES IF USING GRAPH LIBRARIES
            graph_state = self.create_vocab(clean_states)
            graph_states.append(graph_state)

        vocabulary = self.nodes_1 | self.edges
        non_rdf_vocab = set(self.tokenizer.get_vocab().keys())
        num_added_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": list(vocabulary)})

        # This is in Ribeiro, but fasttokenizer doesn't have this
        #self.tokenizer.unique_no_split_tokens.sort(key=lambda x: -len(x))

        for diag_id, dialogue, states in zip(batch['dialogue_id'], batch['turns'], new_states):
            continue


            #self.data_collation(batch)
        return {"states": new_states, "graph_states": graph_states}

    def create_vocab(self, state):
        graph = []
        for rdf in state:
            rdf_graph = {}
            for triple in rdf:
                node_1 = triple[0]
                edge = triple[1]
                self.nodes_1.add(node_1)
                self.edges.add(edge)

        return graph



    def clean_rdf_state(self, states):
        # REMOVE TRIPLES THAT WE DON'T NEED SUCH AS SYSTEM TRIPLES IN NODE 1 AND ONES WITH THE RANDOM PATTERN
        states = map(lambda state: list(filter(self.filter_triples, state['triples'])), states)
        states = [[self.clean_triple(triple) for triple in rdf if triple] for rdf in states]
        if self.dataset_type == "multiwoz":
            states = self.rearrange_sys_triples(states)

        return states


    def clean_triple(self, triple):
        #return [self.clean_node(el) for el in triple]
        return [self.clean_node(el) for el in triple]
 
    def clean_node(self, node):
    
    # _:booking/0699e989  BEFORE ALL OF THIS FIND PATTERNS LIKE THIS IN A GIVEN NODE, REPLACE FORWARD AND CODE WITH PAD TOKEN
        node = node.strip()
        node = node.replace(',', '')  # removing commas to facilitate RDF creation and splitting
        node = node.replace(';', '')  # removing semicolons to facilitate RDF creation and splitting
    
        mask = ""  # Replace with empty mask to make it closer to NL
        randompatternRegex = re.compile(r'\/[a-zA-Z0-9]+')
        return randompatternRegex.sub(mask, node)

