import argparse
import json
import scipy
import numpy as np
import sklearn
import nltk
from nltk import word_tokenize


def pairwise_meteor(candidate, reference):
    return nltk.translate.meteor_score.single_meteor_score(
        word_tokenize(reference), word_tokenize(candidate)
    )


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    scores = np.empty((len(src_data), len(tgt_data)))

    for i, src in enumerate(src_data):
        for j, tgt in enumerate(tgt_data):
            scores[i][j] = metric(src, tgt)

    return scores


def print_with_space(left, right, left_space=45):
    print_spaces = " " * (left_space - len(left))
    print(left + print_spaces + right)


class AVeriTeCEvaluator:

    verdicts = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
    pairwise_metric = None
    max_questions = 10
    metric = None
    averitec_reporting_levels = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    def __init__(self, metric="meteor"):
        self.metric = metric
        if metric == "meteor":
            self.pairwise_metric = pairwise_meteor

    def evaluate_averitec_veracity_by_type(self, srcs, tgts, threshold=0.25):
        types = {}
        for src, tgt in zip(srcs, tgts):
            score = self.compute_pairwise_evidence_score(src, tgt)

            if score <= threshold:
                score = 0

            for t in tgt["claim_types"]:
                if t not in types:
                    types[t] = []

                types[t].append(score)

        return {t: np.mean(v) for t, v in types.items()}

    def evaluate_averitec_score(self, srcs, tgts):
        scores = []
        for src, tgt in zip(srcs, tgts):
            score = self.compute_pairwise_evidence_score(src, tgt)

            this_example_scores = [0.0 for _ in self.averitec_reporting_levels]
            for i, level in enumerate(self.averitec_reporting_levels):
                if score > level:
                    this_example_scores[i] = src["pred_label"] == tgt["label"]

            scores.append(this_example_scores)

        return np.mean(np.array(scores), axis=0)

    def evaluate_veracity(self, src, tgt):
        src_labels = [x["pred_label"] for x in src]
        tgt_labels = [x["label"] for x in tgt]

        acc = np.mean([s == t for s, t in zip(src_labels, tgt_labels)])

        f1 = {
            self.verdicts[i]: x
            for i, x in enumerate(
                sklearn.metrics.f1_score(
                    tgt_labels, src_labels, labels=self.verdicts, average=None
                )
            )
        }
        f1["macro"] = sklearn.metrics.f1_score(
            tgt_labels, src_labels, labels=self.verdicts, average="macro"
        )
        f1["acc"] = acc
        return f1

    def evaluate_questions_only(self, srcs, tgts):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            if "evidence" not in src:
                # If there was no evidence, use the string evidence
                src_questions = self.extract_full_comparison_strings(
                    src, is_target=False
                )[: self.max_questions]
            else:
                src_questions = [
                    qa["question"] for qa in src["evidence"][: self.max_questions]
                ]
            tgt_questions = [qa["question"] for qa in tgt["questions"]]

            pairwise_scores = compute_all_pairwise_scores(
                src_questions, tgt_questions, self.pairwise_metric
            )

            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )

            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_questions))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils)

    def get_n_best_qau(self, srcs, tgts, n=3):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            assignment_utility = self.compute_pairwise_evidence_score(src, tgt)

            all_utils.append(assignment_utility)

        idxs = np.argsort(all_utils)[::-1][:n]

        examples = [
            (
                (
                    srcs[i]["questions"]
                    if "questions" in srcs[i]
                    else srcs[i]["string_evidence"]
                ),
                tgts[i]["questions"],
                all_utils[i],
            )
            for i in idxs
        ]

        return examples

    def compute_pairwise_evidence_score(self, src, tgt):
        """Different key is used for reference_data and prediction.
        For the prediction, the format is
        {"evidence": [
            {
                "question": "What does the increased federal medical assistance percentage mean for you?",
                "answer": "Appendix A: Applicability of the Increased Federal Medical Assistance Percentage ",
                "url": "https://www.medicaid.gov/federal-policy-guidance/downloads/smd21003.pdf"
            }],
        "pred_label": "Supported"}

        And for the data with fold label:
        {"questions": [
            {
                "question": "Where was the claim first published",
                "answers": [
                    {
                        "answer": "It was first published on Sccopertino",
                        "answer_type": "Abstractive",
                        "source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/",
                        "source_medium": "Web text",
                        "cached_source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"
                    }
                ]
            }]
        "label": "Refuted"}
        """

        src_strings = self.extract_full_comparison_strings(src, is_target=False)[
            : self.max_questions
        ]
        tgt_strings = self.extract_full_comparison_strings(tgt)
        pairwise_scores = compute_all_pairwise_scores(
            src_strings, tgt_strings, self.pairwise_metric
        )
        assignment = scipy.optimize.linear_sum_assignment(
            pairwise_scores, maximize=True
        )
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

        # Reweight to account for unmatched target questions
        reweight_term = 1 / float(len(tgt_strings))
        assignment_utility *= reweight_term
        return assignment_utility

    def evaluate_questions_and_answers(self, srcs, tgts):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            src_strings = self.extract_full_comparison_strings(src, is_target=False)[
                : self.max_questions
            ]
            tgt_strings = self.extract_full_comparison_strings(tgt)

            pairwise_scores = compute_all_pairwise_scores(
                src_strings, tgt_strings, self.pairwise_metric
            )

            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )

            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_strings))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils)

    def extract_full_comparison_strings(self, example, is_target=True):
        example_strings = []

        if is_target:
            if "questions" in example:
                for evidence in example["questions"]:
                    # If the answers is not a list, make them a list:
                    if not isinstance(evidence["answers"], list):
                        evidence["answers"] = [evidence["answers"]]

                    for answer in evidence["answers"]:
                        example_strings.append(
                            evidence["question"] + " " + answer["answer"]
                        )
                        if (
                            "answer_type" in answer
                            and answer["answer_type"] == "Boolean"
                        ):
                            example_strings[-1] += ". " + answer["boolean_explanation"]
                    if len(evidence["answers"]) == 0:
                        example_strings.append(
                            evidence["question"] + " No answer could be found."
                        )
        else:
            if "evidence" in example:
                for evidence in example["evidence"]:
                    example_strings.append(
                        evidence["question"] + " " + evidence["answer"]
                    )

        if "string_evidence" in example:
            for full_string_evidence in example["string_evidence"]:
                example_strings.append(full_string_evidence)
        return example_strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the veracity prediction.")
    parser.add_argument(
        "-i",
        "--prediction_file",
        default="data_store/dev_veracity_prediction.json",
        help="Json file with claim, evidence, and veracity prediction.",
    )
    parser.add_argument(
        "--label_file",
        default="data_store/averitec/dev.json",
        help="Json file with labels.",
    )
    args = parser.parse_args()

    with open(args.prediction_file) as f:
        predictions = json.load(f)

    with open(args.label_file) as f:
        all_references = json.load(f)

    # Match references to predictions by claim_id (which is the array index)
    # Build a mapping from claim_id to reference
    references = []
    for pred in predictions:
        claim_id = pred.get('claim_id')
        if claim_id is not None and 0 <= claim_id < len(all_references):
            references.append(all_references[claim_id])
        else:
            print(f"WARNING: claim_id {claim_id} out of range [0, {len(all_references)})")

    print(f"Evaluating {len(predictions)} predictions against {len(references)} references")
    if len(predictions) != len(references):
        print(f"WARNING: {len(predictions)} predictions but {len(references)} matching references")

    scorer = AVeriTeCEvaluator()
    q_score = scorer.evaluate_questions_only(predictions, references)
    print_with_space("Question-only score (HU-" + scorer.metric + "):", str(q_score))
    p_score = scorer.evaluate_questions_and_answers(predictions, references)
    print_with_space("Question-answer score (HU-" + scorer.metric + "):", str(p_score))
    print("====================")

    v_score = scorer.evaluate_veracity(predictions, references)
    print("Veracity F1 scores:")
    for k, v in v_score.items():
        print_with_space(" * " + k + ":", str(v))

    print("--------------------")
    print("AVeriTeC scores:")

    v_score = scorer.evaluate_averitec_score(predictions, references)

    for i, level in enumerate(scorer.averitec_reporting_levels):
        print_with_space(
            " * Veracity scores (" + scorer.metric + " @ " + str(level) + "):",
            str(v_score[i]),
        )
    print("--------------------")
    print("AVeriTeC scores by type @ 0.25:")
    type_scores = scorer.evaluate_averitec_veracity_by_type(
        predictions, references, threshold=0.25
    )
    for t, v in type_scores.items():
        print_with_space(" * Veracity scores (" + t + "):", str(v))
