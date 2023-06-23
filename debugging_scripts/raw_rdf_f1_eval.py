from collections import Counter
import re
# sources: quentin, webnlg, https://github.com/huggingface/datasets/blob/main/metrics/squad/evaluate.py
# https://github.com/snowblink14/smatch  --> quite similar to what we are doing with SMATCH


def getRefs(allreftriples):

    newreflist = []

    for entry in allreftriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return newreflist


def getCands(allcandtriples):

    newcandlist = []

    for entry in allcandtriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return newcandlist



def exact_triple_scores(newreflist, newcandlist):

    newcandlist = [{re.sub('[ _-]+', '', x).lower() for x in cand} for cand in newcandlist]
    newreflist = [{re.sub('[ _-]+', '', x).lower() for x in ref} for ref in newreflist]

    intersections = [
        c & r for c, r in zip(newcandlist, newreflist)
    ]


    precisions = [
        len(i) / len(c) if len(c) > 0 else 1
        for i,c in zip(intersections, newcandlist)
    ]

    recalls = [
        len(i) /len(r) if len(r) > 0 else 1
        for i,r in zip(intersections, newreflist)
    ]


    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    f1 = 2 * (p*r) / (p+r)

    return {'precision': p, 'recall': r, 'f1' : f1}


def scores(reflist, candlist):
    newreflist = getRefs(reflist)
    newcandlist = getCands(candlist)

    scores = exact_triple_scores(newreflist, newcandlist)

    return scores

def f1_score(ref, pred):
    ref = "|".join(ref).split('|')
    pred = "|".join(pred).split('|')
    common = Counter(ref) & Counter(pred)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(ref)
    f1 = (2 * precision * recall) / (precision + recall)

    return {'precision': precision, 'recall': recall, 'f1' : f1}

def counter_squad_method(reflist, candlist):
    # exact match is already JGA, so no need for implementing exact match here
    # this is looking for the nodes, much more flexible
    f1 = []
    precision = []
    recall = []
    for ref, pred in zip(reflist, candlist):
        res = f1_score(ref, pred)
        f1.append(res['f1'])
        precision.append(res['precision'])
        recall.append(res['recall'])

    f1 = sum(f1) / len(f1)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    return {'precision': precision, 'recall': recall, 'f1' : f1}




if __name__ == '__main__':
    candlist = [
        ['s|p|o', 'subject|property|object'],
        ['entity|relation|thing', 'this|and|that', 'what|the|hell'],
        ['a|b|c']
    ]

    reflist = [
        ['s|p|o', 'subject|predicate|object'],
        ['entity|relation|thing', 'this|and|that', 'what|the|hell'],
        ['a|b|c']
    ]

    for c, r in zip(candlist, reflist):
        print('Candidate:', c)
        print('Reference:', r)

    res = scores(reflist, candlist)

    print('Scores:', res)

    res = counter_squad_method(reflist, candlist)
    print('Scores:', res)
