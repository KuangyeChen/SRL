import numpy as np
from typing import Tuple, Sequence, List, Set
from os.path import join as path_join

__all__ = ['KnowledgeGraph',
           'Triple']

Triple = Tuple[int, int, int]


class KnowledgeGraph(object):
    """
    A Class storing relational data.

    Stored information:
        All entities names,
        All relations names,
        All triples representing existing relations.

    Triples are stored in form (e1, e2, r)
    e.g. A Triple: `Disease_or_Syndrome Affects Plant`
         `Disease_or_Syndrome` is entity 2.
         `Plant` is entity 5.
         `Affects` is relation 1.
         This triple is stored as (2, 5, 1).
    """

    def __init__(self):
        """
        Create empty knowledge graph.
        """

        self.__entities = []                    # type: List[str]
        self.__relations = []                   # type: List[str]
        self.__triples_set = set()              # type: Set[Triple]
        self.__triples_array = np.array([])     # type: np.ndarray
        self.__corrupt_list = []
        self.__test_set_idx = []                # type: List[int]
        self.__train_set_idx = []               # type: List[int]
        self.__valid_set_idx = []               # type: List[int]

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

        :param float valid_percent: Percentage of validation set.
        :param float test_percent: Percentage of test set.
        """

        test_size = int(len(self.__triples_set) * test_percent)
        valid_size = int(len(self.__triples_set) * valid_percent)
        all_idx = np.random.permutation(len(self.__triples_set))
        self.__test_set_idx = all_idx[:test_size]
        self.__valid_set_idx = all_idx[test_size: test_size + valid_size]
        self.__train_set_idx = all_idx[test_size + valid_size:]

    def get_train_batch(self, batch_size, corrupt_size):
        """
        Make a batch of triples for training.

        :param int batch_size: Size of the batch.
        :return: A np.ndarray containing the selected triples.
        """

        all_idx = np.random.permutation(len(self.__train_set_idx))
        if batch_size > len(self.__train_set_idx):
            batch_size = len(self.__train_set_idx)
        data = []
        labels = []
        for i in range(batch_size):
            data.append(self.__triples_array[self.__train_set_idx[all_idx[i]]])
            data.extend(self.__corrupt_list[self.__train_set_idx[all_idx[i]]][:corrupt_size])
            labels.append(1)
            labels.extend([0]*(len(data)-len(labels)))
        return data, labels

    def get_train_set(self, corrupt_size):
        """
        Return the train set of triples.

        :return: A np.ndarray containing the selected triples.
        """
        data = []
        labels = []
        for i in self.__train_set_idx:
            data.append(self.__triples_array[i])
            data.extend(self.__corrupt_list[i][:corrupt_size])
            labels.append(1)
            labels.extend([0]*(len(data)-len(labels)))
            
        return data, labels

    def get_test_set(self, corrupt_size):
        """
        Return the test set of triples.

        :return: A np.ndarray containing the selected triples.
        """

        data = []
        labels = []
        for i in self.__test_set_idx:
            data.append(self.__triples_array[i])
            data.extend(self.__corrupt_list[i][:corrupt_size])
            labels.append(1)
            labels.extend([0]*(len(data)-len(labels)))
        return data, labels

    def get_valid_set(self, corrupt_size):
        """
        Return the validation set of triples.

        :return: A np.ndarray containing the selected triples.
        """

        data = []
        labels = []
        for i in self.__valid_set_idx:
            data.append(self.__triples_array[i])
            data.extend(self.__corrupt_list[i][:corrupt_size])
            labels.append(1)
            labels.extend([0]*(len(data)-len(labels)))
        return data, labels

    def get_all_triples(self):
        """
        Return all triples.

        :return: A np.ndarray containing all triples.
        """

        return np.array(list(self.__triples_set))

    def read_data_from_txt(self, path):
        """
        Read a database from txt files.
        A database should contain three files:
            data.txt: each row is a triple. e.g. `Disease_or_Syndrome Affects Plant`
            entities.txt: each row is a entity name. e.g. `Plant`
            relations.txt: each row is a relation name. e.g. `Affects`

        :param str path: The path of the directory contains the txt files.
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

        self.__triples_array = list(self.__triples_set)
        
    def make_corrupt(self, corrupt_size):
        self.__corrupt_list = [ [] for _ in range(len(self.__triples_array))]
        
        for i in range(len(self.__triples_array)):
            triple = self.__triples_array[i]

            all_idx = np.random.permutation(self.number_of_entities())

            idx = 0
            for corrupt_i in range(corrupt_size):
                while idx < len(all_idx) and self.check_triple((triple[0], all_idx[idx], triple[2])):
                    idx = idx + 1
                if idx == len(all_idx):
                    break

                self.__corrupt_list[i].append((triple[0], all_idx[idx], triple[2]))
                idx = idx + 1

    def add_entities(self, entities_list):
        """
        Add entities to knowledge graph.

        :param Sequence[str] entities_list: New entities to add.
        """
        for entity in entities_list:
            self.__entities.append(entity)

    def add_relations(self, relations_list):
        """
        Add relations to knowledge graph.

        :param Sequence[str] relations_list: New relations to add.
        """

        self.__relations = self.__relations + relations_list

    def remove_add_triples(self, remove_list, add_list):
        """
        Remove and add some triples to knowledge graph.
        Input must be a list.
        Triples must be given in tuple. e.g. (e1, e2, r)

        :param Sequence[Triple] remove_list: Triples to remove.
        :param Sequence[Triple] add_list: List of triples to add.
        """

        for triple in remove_list:
            triple in self.__triples_set and self.__triples_set.remove(triple)
        for triple in add_list:
            self.__triples_set.add(triple)
        self.__triples_array = np.array(list(self.__triples_set))

    def entity_index_to_name(self, idx):
        """
        Return the name of ith entity.

        :param int idx: Index of the wanted entity.
        :return: Name of the wanted entity.
        """

        return self.__entities[idx]

    def number_of_entities(self):
        """
        Return the number of entities.

        :return: Number of entities.
        """

        return len(self.__entities)

    def relation_index_to_name(self, idx):
        """
        Return the name of ith relation.

        :param int idx: Index of the wanted relation.
        :return: Name of the wanted relation.
        """

        return self.__relations[idx]

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

    def check_triple(self, triple):
        """
        Check if triple (e1, e2, r) exists in knowledge graph.

        :param Triple triple: e1.
        :return: True or False.
        """

        return triple in self.__triples_set

    def to_numpy_array(self):
        """
        Return the knowledge graph as a tensor T.
        T[e1, e2, r] = 1 if (e1, e2, r) exists in knowledge graph.
        T[e1, e2, r] = 0 otherwise.

        :return: np.ndarray representing the tensor.
        """

        array = np.zeros([len(self.__entities),
                          len(self.__entities),
                          len(self.__relations)])

        for triple in self.__triples_set:
            array[triple[0], triple[1], triple[2]] = 1
        return array
