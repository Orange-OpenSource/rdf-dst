# Results

Storing CSVs with results from relevant results

MULTIWOZ RDF (2.3?)
| Model          | JGA | JG-F1 | GLEU | METEOR | JG-RECALL | JG-PRECISION | Batch size |
|----------------|-----|-------|------|--------|-----------|--------------|------------|
| Baseline Exp 1 | 74% | 49%   | 31%  | 57%    | 98%       | 33%          | 8          |
| Baseline Exp 2 | 45% | 47%   | 30%  | 55%    | 94%       | 32%          | 8          |
| Baseline Exp 3 | 37% | 41%   | 23%  | 50%    | 92%       | 26%          | 8          |
| T5-base. Exp 1 | 70% | 84%   | 80%  | 75%    | 96%       | 75%          | 2          |
| T5-base. Exp 2 | 38% | 72%   | 65%  | 67%    | 90%       | 60%          | 4          |
| T5-base. Exp 3 | 17% | 61%   | 59%  | 57%    | 74%       | 52%          | 4          |
           

Experiment 1 with T5 base has a slide window to remove older context, as the whole input does not fit into current T5. Experiments will be carried out with flanT5 and longT5 to gauge performance with the whole input

The results so far show that to get ALL rdfs per batch, context + states is best. However, if we are more flexible with our evaluation, we can see that context does predict a fair number of RDFs and in average a good amount per batch (JGC F1 and JGC RECALL)

The baseline does not have RDF data, but rather slot values. This may explain why the JGA is much higher with fewer information to generate. However, we see that our preprocessing steps for the RDF setup (i.e. extra tokens to uphold the direction of the graph) help with F1s and our span evaluation. 


# Notes:

Must test with DSTC and San Francisco. DSTC is closer to its original representation, whereas multiwoz rdf does not contain multitasks. It's only multidomain. Must improve dataset

we don't compute slot accuracy because some RDFs have been annotated as follows:

First run of T5-base was with a batch of 2, whereas baseline was with a batch of 8

```
system|canthelp|search/ff7735d7, system|canthelp|search/19a734d2
```

Adding slot accuracy and reviewing overall metrics is a possible extension of this project and an overall worthwhile discussion in the DST problem
