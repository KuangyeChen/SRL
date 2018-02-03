import numpy as np

from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *

test_percent = 0.1
corrupt_size = 100

def run_training():
    database = KnowledgeGraph()
    database.read_data_from_txt('data/kin')
    database.spilt_train_valid_test(0, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    train_set = database.get_train_set()
    result = factorize_rescal(train_set)
    test_batch, labels = make_corrupt_for_eval(database.get_test_set(), database,
                                               num_entities, corrupt_size)
    result.eval()