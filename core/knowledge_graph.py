import numpy as np
from os.path import join as path_join

__all__ = ['KnowledgeGraph']


class KnowledgeGraph(object):
    """
    A Class storing relational data.

    Stored information:
        All entities names,
        All relations names,
        All triples representing a existing relation.

    Triples are stored in form (e1, e2, r)
    e.g. `Disease_or_Syndrome Affects Plant`
         `Disease_or_Syndrome` is entity 2.
         `Plant` is entity 5.
         `Affects` is relation 1.
         This triple are stored as (2, 5, 1).
    """

    def __init__(self):
        """
        Create empty knowledge graph.
        """

        self.__entities = []
        self.__relations = []
        self.__triples_set = set()
        self.__triples_array = np.array([])
        self.__test_set_idx = []
        self.__train_set_idx = []
        self.__valid_set_idx = []

    def reset(self):
        """
        Clear the stored data.
        """

        self.__entities.clear()
        self.__relations.clear()
        self.__triples_set.clear()
        self.__triples_array = np.array([])
        self.__test_set_idx.clear()
        self.__train_set_idx.clear()
        self.__valid_set_idx.clear()

    def spilt_train_valid_test(self, valid_percent, test_percent):
        """
        Split the stored triples into train, validation and test set.

        :param valid_percent: Percentage of validation set.
        :param test_percent: Percentage of test set.
        """

        test_size = int(len(self.__triples_set) * test_percent)
        valid_size = int(len(self.__triples_set) * valid_percent)
        all_idx = np.random.permutation(len(self.__triples_set))
        self.__test_set_idx = all_idx[:test_size]
        self.__valid_set_idx = all_idx[test_size: test_size + valid_size]
        self.__train_set_idx = all_idx[test_size + valid_size:]

    def get_train_batch(self, batch_size):
        """
        Make a batch of triples for training.

        :param batch_size: Size of the batch.
        :return: A numpy array containing the selected triples.
        """

        all_idx = np.random.permutation(len(self.__train_set_idx))
        return self.__triples_array[self.__train_set_idx[all_idx[:batch_size]]]

    def get_train_set(self):
        """
        Return the train set of triples.

        :return: A numpy array containing the selected triples.
        """

        return self.__triples_array[self.__train_set_idx]

    def get_test_set(self):
        """
        Return the test set of triples.

        :return: A numpy array containing the selected triples.
        """

        return self.__triples_array[self.__test_set_idx]

    def get_valid_set(self):
        """
        Return the validation set of triples.

        :return: A numpy array containing the selected triples.
        """

        return self.__triples_array[self.__valid_set_idx]

    def get_all_triples(self):
        """
        Return all triples.

        :return: A numpy array containing all triples.
        """

        return np.array(list(self.__triples_set))

    def read_data_from_txt(self, path):
        """
        Read a database from txt files.
        A database should contain three files:
            data.txt: each row is a triple. e.g. `Disease_or_Syndrome Affects Plant`
            entities.txt: each row is a entity name. e.g. `Plant`
            relations.txt: each row is a relation name. e.g. `Affects`

        :param path: A string with the path of the directory contains the txt files.
        """

        with open(path_join(path, 'entities.txt'), 'r') as fin:
            self.__entities = fin.read().strip().split('\n')
        with open(path_join(path, 'relations.txt'), 'r') as fin:
            self.__relations = fin.read().strip().split('\n')

        entity_to_index = {self.__entities[i]: i for i in range(len(self.__entities))}
        relation_to_index = {self.__relations[i]: i for i in range(len(self.__relations))}
        with open(path_join(path, 'data.txt'), 'r') as fin:
            for line in fin:
                triple_lst = line.strip().split(' ')
                # Attention to the stored order (e1, e2, r)
                self.__triples_set.add((entity_to_index[triple_lst[0]],
                                        entity_to_index[triple_lst[2]],
                                        relation_to_index[triple_lst[1]]))

        self.__triples_array = np.array(list(self.__triples_set))

    def add_entities(self, entities_list):
        """
        Add entities to knowledge graph.
        Input must be a list.

        :param entities_list: List of new entities.
        """

        self.__entities = self.__entities + entities_list

    def add_relations(self, relations_list):
        """
        Add relations to knowledge graph.
        Input must be a list.
        :param relations_list: List of new relations.
        """

        self.__relations = self.__relations + relations_list

    def remove_add_triples(self, remove_list, add_list):
        """
        Remove and add some triples to knowledge graph.
        Input must be a list.
        Triples must be given in tuple. e.g. (e1, e2, r)

        :param remove_list: List of triples to remove.
        :param add_list: List of triples to add.
        """

        for triple in remove_list:
            triple in self.__triples_set and self.__triples_set.remove(triple)
        for triple in add_list:
            self.__triples_set.add(triple)
        self.__triples_array = np.array(list(self.__triples_set))

    def entity_index_to_name(self, i):
        """
        Return the name of ith entity.

        :param i: Index of the wanted entity.
        :return: Name of the wanted entity.
        """

        return self.__entities[i]

    def number_of_entities(self):
        """
        Return the number of entities.

        :return: Number of entities.
        """

        return len(self.__entities)

    def relation_index_to_name(self, i):
        """
        Return the name of ith relation.

        :param i: Index of the wanted relation.
        :return: Name of the wanted relation.
        """

        return self.__relations[i]

    def number_of_relations(self):
        """
        Return the number of relations.

        :return: Number of relations.
        """

        return len(self.__relations)

    def number_of_triples(self):
        """
        Return the number of triples.

        :return: Number of triples.
        """

        return len(self.__triples_set)

    def check_triple(self, entity1, entity2, relation):
        """
        Check if triple (e1, e2, r) exists in knowledge graph.

        :param entity1: e1.
        :param entity2: e2.
        :param relation: r.
        :return: True or False.
        """
        return (entity1, entity2, relation) in self.__triples_set

    def to_numpy_array(self):
        """
        Return the knowledge graph as a tensor T.
        T[e1, e2, r] = 1 if (e1, e2, r) exists in knowledge graph.
        T[e1, e2, r] = 0 otherwise.

        :return: tensor T (numpy.ndarray).
        """
        array = np.zeros([len(self.__entities),
                          len(self.__entities),
                          len(self.__relations)])

        for triple in self.__triples_set:
            array[triple[0], triple[1], triple[2]] = 1
        return array
