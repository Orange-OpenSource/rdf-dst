# Results

Storing CSVs with results from relevant results

| Model          | JGA | JGC F1 | GLEU | METEOR | MEAN_RDFS |
|----------------|-----|--------|------|--------|-----------|
| T5-base. Exp 1 | 70% | 84%    | 80%  | 75%    | 96%       |
| T5-base. Exp 2 | 38% | 72%    | 65%  | 67%    | 90%       |
| T5-base. Exp 3 | 17% | 61%    | 59%  | 56%    | 74%       |


Experiment 1 with T5 base has a slide window to remove older context, as the whole input does not fit into current T5. Experiments will be carried out with flanT5 and longT5 to gauge performance with the whole input

The results so far show that to get ALL rdfs per batch, context + states is best. However, if we are more flexible with our evaluation, we can see that context does predict a fair number of RDFs and in average a good amount per batch (JGC F1 and MEAN_RDFS)