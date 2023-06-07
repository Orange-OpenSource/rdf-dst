# Results

Storing CSVs with results from relevant results

| model          | JGA | JGC F1 | GLEU | METEOR |
|----------------|-----|--------|------|--------|
| T5-base. Exp 1 | 70& | 84%    |      |        |
| T5-base. Exp 2 | 38% | 72%    |      |        |
| T5-base. Exp 3 | 17% | 61%    |      |        |


Experiment 1 with T5 base has a slide window to remove older context, as the whole input does not fit into current T5. Experiments will be carried out with flanT5 and longT5 to gauge performance with the whole input
