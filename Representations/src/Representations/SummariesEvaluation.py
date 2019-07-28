import argparse
import json
import pickle
import numpy as np
import rouge


"""
Rouge evaluation script
"""


def prepare_results(metric, p, r, f):
    # return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(
    #   metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
    return 100.0 * p, 100.0 * r, 100.0 * f


def get_rouge_scores(doc_sents, goldens, maxlength, maxtype):
    res = {}
    for aggregator in ['Avg', 'Best', 'Individual']:

        res[aggregator] = {}
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l', 'rouge-w'],
            max_n=4,
            limit_length=True,
            length_limit=maxlength,
            # length_limit=100,
            length_limit_type=maxtype,
            # length_limit_type='words',
            apply_avg=apply_avg,
            apply_best=apply_best,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True)

        all_hypothesis = doc_sents
        all_references = goldens

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            res[aggregator][metric] = {}
            if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                res[aggregator][metric]["precision"] = []
                res[aggregator][metric]["recall"] = []
                res[aggregator][metric]["f1"] = []

                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        # print('\tHypothesis #{} & Reference #{}: '.format(
                        #hypothesis_id, reference_id))

                        res[aggregator][metric]["precision"].append(results_per_ref['p'])
                        res[aggregator][metric]["recall"].append(results_per_ref['r'])
                        res[aggregator][metric]["f1"].append(results_per_ref['f'])

                        #print('\t' + prepare_results(
                        #    metric, results_per_ref['p'][reference_id],
                        #    results_per_ref['r'][reference_id],
                        #    results_per_ref['f'][reference_id]))
                #print()
            else:
                res[aggregator][metric]["precision"] = results['p']
                res[aggregator][metric]["recall"] = results['r']
                res[aggregator][metric]["f1"] = results['f']
                # print(
                #     prepare_results(metric, results['p'], results['r'],
                #                     results['f']))
        #print()
    return res



if __name__ == "__main__":
    """
    Arguments:
    results_file: path to <num samples x DIM > ndarray of float predictions, in one-hot format (DIM=2), or a prediction vector (DIM=1)
                  containing label indexes or probability scores. In the latter case, a decision threshold of 0.5 is adopted.
                  If it's a list, the first element is retrieved (for compatibility reasons with other tools). 
    dataset_file: path to the json dataset serialization. The structure below should be present:
                  {"data": {"test":[{"text":"bla..", "document_index": 0}, .... ]}}
    golden_summaries_file: path to the golden summaries json dataset serialization. The structure below should be present:
                  {"golden": [{"summary": ["summary sent1", ...], "document_index": 0}, ...]}
    """
    parser = argparse.ArgumentParser(description='')
    # classification results file, to pick which sentences where classified as summary parts
    parser.add_argument('results_file', help="File containing predictions for an extractive summarization binary classification task.")
    # the dataset path corresponding to the results
    parser.add_argument('dataset_file', help="Path to the dataset json file of the test data.",)
    # the golden summaries path
    parser.add_argument('golden_summaries_file', help="Path to the golden summaries fo the test data.")
    args = parser.parse_args()

    with open(args.dataset_file) as f:
        dataset = json.load(f)
    with open(args.results_file, "rb") as f:
        results = pickle.load(f)
        if type(results) == list:
            results = results[0]
        else:
            results = np.squeeze(results)

    if len(results.shape) > 1:
        # get selected sentence indexes
        predictions = np.argmax(results, axis=1)
    else:
        predictions = results
    if set(predictions) != set([0,1]):
        # convert probability scores to label indexes
        predictions = np.round(predictions)
    selected_sents = np.where(predictions == 1)[0]
    # get sentences themselves
    selected_sents = [dataset['data']['test'][idx] for idx in selected_sents]
    # group by document index
    doc_sents = {}
    for dat in selected_sents:
        sent = dat['text']
        doc_index = dat['document_index']
        if doc_index not in doc_sents:
            doc_sents[doc_index] = []
        doc_sents[doc_index].append(sent)

    # merge sentences
    for doc_idx in doc_sents:
        sents = [s.strip() for s in doc_sents[doc_idx]]
        sents = [s if s.endswith(".") else s + "." for s in doc_sents[doc_idx]]
        doc_sents[doc_idx] = " ".join(doc_sents[doc_idx])

    with open(args.golden_summaries_file) as f:
        goldens = json.load(f)

    doc_goldens = {}
    for gold in goldens['golden']:
        summs = gold['summaries']
        doc_index = gold['document_index']
        if doc_index in doc_goldens:
            print("Encountered already existing doc index in goldens: ",
                  doc_index)
            exit(1)
        doc_goldens[doc_index] = " ".join(summs)
        if doc_index not in doc_sents:
            print(
                "[!] Document index {} has no assigned summary! -- setting empty string"
                    .format(doc_index))
            doc_sents[doc_index] = ""

    # read classification results

    # if it's a directory, average the results?
    # parse golden summaries
    doc_keys = list(doc_sents.keys())
    doc_sents = [doc_sents[k] for k in doc_keys]
    doc_goldens = [doc_goldens[k] for k in doc_keys]
    print("Evaluating {} input and {} golden summaries".format(
        len(doc_sents), len(doc_goldens)))
    print(doc_sents)
    print(doc_goldens)
    maxlength, maxtype = 100, "words"
    scores = get_rouge_scores(doc_sents, doc_goldens, maxlength, maxtype)
    print(scores)
    with open("results.pickle", "wb") as f:
        pickle.dump(scores, f)
