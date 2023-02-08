import ast

import numpy as np
import pandas as pd

from expansion_embeddings import text_processing

pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=2)

N_INFERENCES = 10
base = 'baseline_12345'
new_version = 'intraspan_12345'


def process(all_expansions):
    inferences = all_expansions.split("After:")
    for i in range(len(inferences)):
        inferences[i] = text_processing(inferences[i])
    inferences.append("")
    inferences.append("")
    before_array = [inf.lstrip() + "." for inf in inferences[0].split(".") if len(inf.split()) > 3][
                   :int(N_INFERENCES / 2)]
    before_array = before_array + ["NONE"] * (int(N_INFERENCES / 2) - len(before_array))
    after_array = [inf.lstrip() + "." for inf in inferences[1].split(".") if len(inf.split()) > 3][
                  :int(N_INFERENCES / 2)]
    after_array = after_array + ["NONE"] * (int(N_INFERENCES / 2) - len(after_array))
    return before_array + after_array


if __name__ == '__main__':
    # Spans and span expansions
    spans = pd.read_csv(f"../coref/logs/{new_version}/span_examples_ns.csv")
    spans["exps"] = spans["exps"].astype(str)

    # baseline version
    errors1 = pd.read_csv(f'../coref/logs/{base}/errors.csv')
    print("# of errors in baseline:", errors1.shape[0])
    # print(errors1.columns)

    # new version
    errors2 = pd.read_csv(f'../coref/logs/{new_version}/errors.csv')
    print(f"# of errors in {new_version}:", errors2.shape[0])
    # print(errors2.columns)

    # what is common with baseline
    common = pd.merge(errors1, errors2, how='inner', on=['c1', 'c2', 'span1', 'span2'])
    print("# of errors common:", common.shape[0])

    # print(common.head())

    # indicators
    # errors1['indicator'] = errors1['c1'].str.cat(errors1['span1']).cat(errors1['c2']).str.cat(errors1['span2'])
    # errors2['indicator'] = errors2['c1'].str.cat(errors2['span1']).cat(errors2['c2']).str.cat(errors2['span2'])
    # common['indicator'] = common['c1'].str.cat(common['span1']).cat(common['c2']).str.cat(common['span2'])

    # ATTN files:
    attn = pd.read_csv(f'../coref/logs/{new_version}/attnention.csv')

    F1 = errors2.shape[0] - common.shape[0]
    F2 = errors1.shape[0] - common.shape[0]
    print(f"# errors more than baseline: {F1}")
    print(f"# of errors less than baseline: {F2}")

    # print(common['Unnamed: 0_y'] == errors2.index)
    # print(errors2.head())
    baseline_fails = errors1[~errors1['Unnamed: 0'].isin(common['Unnamed: 0_x'])]
    concat_fails = errors2[~errors2['Unnamed: 0'].isin(common['Unnamed: 0_y'])]

    print("Percentage errors rectified by new version:", baseline_fails.shape[0] / errors1.shape[0])
    p = 0
    n = 0
    for index, row in concat_fails.iterrows():
        if row["actual_labels"] == 1:
            p += 1
        else:
            n += 1
        if row["actual_labels"] == 1:
            print("######################### EXAMPLE ##################")
            print(f'{row["actual_labels"]}')
            print(f'{row["sent1"]} \n  {row["span1"]}\n')
            print(f'{row["sent2"]} \n {row["span2"]}\n')

            expansions1 = (
            spans[(spans["combined_id"] == row["c1"]) & (spans["spans"] == row["span1"])]["exps"].values[0])
            expansions2 = (
            spans[(spans["combined_id"] == row["c2"]) & (spans["spans"] == row["span2"])]["exps"].values[0])
            expansions1 = process(expansions1)
            expansions2 = process(expansions2)

            b1 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (
                        attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["b1"].values[0])
            a1 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (
                        attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["a1"].values[0])
            b2 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (
                        attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["b2"].values[0])
            a2 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (
                        attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["a2"].values[0])

            # if b1 and a1 and b2 and a2:
            before1 = list(zip(expansions1[:5], list(np.around(np.array(b1), 2))))
            before2 = list(zip(expansions2[:5], list(np.around(np.array(b2), 2))))
            after1 = list(zip(expansions1[5:], list(np.around(np.array(a1), 2))))
            after2 = list(zip(expansions2[5:], list(np.around(np.array(a2), 2))))
            before1 = sorted(before1, key=lambda l: l[1], reverse=True)
            before2 = sorted(before2, key=lambda l: l[1], reverse=True)
            after1 = sorted(after1, key=lambda l: l[1], reverse=True)
            after2 = sorted(after2, key=lambda l: l[1], reverse=True)
            print(f"\nBefore {row['span1']}:")
            for c in before1:
                print(f"{c[0]} ({c[1]})")
            print(f"\nAfter {row['span1']}:")
            for c in after1:
                print(f"{c[0]} ({c[1]})")
            print(f"\nBefore {row['span2']}:")
            for c in before2:
                print(f"{c[0]} ({c[1]})")
            print(f"\nBefore {row['span2']}:")
            for c in after2:
                print(f"{c[0]} ({c[1]})")
        # combined2 = sorted(combined2, key=lambda l:l[1], reverse=True)
        # combined1 = list(zip(expansions1, b1+a1))
        # combined2 = list(zip(expansions2, b2+a2))
        # combined1 = sorted(combined1, key=lambda l:l[1], reverse=True)
        # combined2 = sorted(combined2, key=lambda l:l[1], reverse=True)

        # print("First span inference:")
        # for c in combined1:
        #     print(f"{c[0]} {c[1]}")
        # # print(b1 + a1)
        # print("Second span inference:")
        # # print(expansions2, "\n")
        # for c in combined2:
        #     print(f"{c[0]}, {c[1]}")
        # print("#############################\n")

    print("# of positive error corrections", p)
    print("# of negative error corrections", n)
