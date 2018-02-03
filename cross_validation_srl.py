#!/usr/bin/env python

import logging
import argparse
import numpy as np
from core.knowledge_graph import KnowledgeGraph
from core.core import validation_in_tensor
from core.core import read_database_from_file

parser = argparse.ArgumentParser('Script for cross validation')
parser.add_argument('filename', metavar='F', help='Database file')
parser.add_argument('-f', '--fold', type=int, default=10,
                    help='How many folds in cross validation, default 10')
parser.add_argument('-a', '--algorithm', default='rescal',
                    choices=['rescal', 'tucker', 'cp'],
                    help='Which algorithm to use, default Rescal')
parser.add_argument('-r', '--rank', type=int, nargs='+', default=[100],
                    help='Rank used in algorithm, default 100. Tucker need 3 ranks')

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')

    logging.info('Database: %s' % args.filename)
    logging.info('%d fold cross validation' % args.fold)
    logging.info('Algorithm: %s, rank: %s' % (args.algorithm, str(args.rank)))
    logging.info('Loading database...')
    kg = KnowledgeGraph()
    kg.read_data_from_txt(args.filename)

    logging.info('Loading done.')
    logging.info('Rels: %d, Ents: %d' % (kg.number_of_relations(),kg.number_of_entities()))

    database = kg.to_numpy_array()
    # Do cross-validation
    FOLDS = args.fold
    IDX = list(range(database.size))
    np.random.shuffle(IDX)

    fsz = int(database.size / FOLDS)
    offset = 0
    AUC_PR_train = np.zeros(FOLDS)
    AP_train = np.zeros(FOLDS)
    AUC_ROC_train = np.zeros(FOLDS)
    AUC_PR_test = np.zeros(FOLDS)
    AP_test = np.zeros(FOLDS)
    AUC_ROC_test = np.zeros(FOLDS)
    for f in range(FOLDS):
        logging.info('----------------- Fold %d -----------------' % f)
        idx_test = IDX[offset:offset + fsz]
        idx_train = np.setdiff1d(IDX, idx_test)
        np.random.shuffle(idx_train)
        idx_train = idx_train[:fsz].tolist()

        logging.info('Train set.....')
        AUC_PR_train[f], AP_train[f], AUC_ROC_train[f] = validation_in_tensor(
            database, idx_train + idx_test, idx_train,
            args.algorithm, args.rank)
        logging.info('AUC_PR: %f , AP: %f , AUC_ROC: %f' % (
            AUC_PR_train[f], AP_train[f], AUC_ROC_train[f]))

        logging.info('Test set......')
        AUC_PR_test[f], AP_test[f], AUC_ROC_test[f] = validation_in_tensor(
            database, idx_test, idx_test,
            args.algorithm, args.rank)
        logging.info('AUC_PR: %f , AP: %f , AUC_ROC: %f\n' % (
            AUC_PR_test[f], AP_test[f], AUC_ROC_test[f]))

        offset += fsz

    logging.info('\nAUC-PR Train Mean / Std: %f / %f' % (AUC_PR_train.mean(), AUC_PR_train.std())
                 + '\nAP Train Mean / Std: %f / %f' % (AP_train.mean(), AP_train.std())
                 + '\nAUC-ROC Train Mean / Std: %f / %f\n' %(AUC_ROC_train.mean(), AUC_ROC_train.std()))
    logging.info('\nAUC-PR Test Mean / Std: %f / %f' % (AUC_PR_test.mean(), AUC_PR_test.std())
                 + '\nAP Test Mean / Std: %f / %f' % (AP_test.mean(), AP_test.std())
                 + '\nAUC-ROC Test Mean / Std: %f / %f\n' %(AUC_ROC_test.mean(), AUC_ROC_test.std()))
