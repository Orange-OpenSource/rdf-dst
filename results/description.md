# Results

Storing CSVs with results from relevant results

--- UPDATED FLAN EXP2 AND LOCAL BASE 2, BUT REVIEW WITH PEFT? BEST MODEL WILL BE XL AND ADDED TOKENS IN BASE?

MULTIWOZ RDF (convlab 2.3)
| Model                   | JGA | JG-F1 | GLEU | METEOR | JG-RECALL | JG-PRECISION | Batch size |
|-------------------------|-----|-------|------|--------|-----------|--------------|------------|
| Baseline Exp 1          | 37% | 32%   | 17%  | 35%    | 65%       | 21%          | 16         |
| Baseline Exp 2          | 24% | 29%   | 18%  | 32%    | 53%       | 20%          | 16         |
| Baseline Exp 3          | 17% | 10%   | 10%  | 20%    | 27%       |  6%          | 32         |
| T5-base. Exp 1 review   | 70% | 84%   | 80%  | 75%    | 96%       | 75%          | 2          |
| T5-base. Exp 2 review   | 38% | 72%   | 65%  | 67%    | 90%       | 60%          | 4          |
| T5-base. Exp 3 review   | 17% | 61%   | 59%  | 57%    | 74%       | 52%          | 4          |
| flanT5-base. Exp 1      | 60% | 83% | 71%    | 73%    | 95%       | 74%          | 2          |
| flanT5-base. Exp 2 review       | 36% | 65%   | 43%    | 60%      | 92%       | 50%          |            |
| flanT5-base. Exp 3 none      | 10% | 59%   | %    | %      | 66%       | 53%          |            |
| longlocal-base. Exp 1   | 61% | 83%   | 73%  | 73%    | 96%       | 74%          | 2          |
| longlocal-base. Exp 2 review    | 25% | 71%   | 62%    | 66%      | 87%       | 60%          | 2          |
| longlocal-base. Exp 3  none  | 15% | 58%   | %    | %      | 69%       | 49%          |            |
| longtglobal-base. Exp 1 | 63% | 85%   | 76%    | 76%      | 95%       | 78%          |            |
| longtglobal-base. Exp 2 none | 48% | 76%   | %    | %      | 94%       | 63%          |            |
| T5-large. Exp 1   | 72% | 82%   | 69%  | 70%    | 97%       | 70%          | 2          |


PEFT TABLE 
| Model          | JGA | JG-F1 | GLEU | METEOR | JG-RECALL | JG-PRECISION | Batch size |
|----------------|-----|-------|------|--------|-----------|--------------|------------|
| Baseline Exp 1 | 49% | 36%   | 19%  | 40%    | 73%       | 24%          | 24         |
| Baseline Exp 2 | 62% | 41%   | 19%  | 39%    | 82%       | 27%          | 24         |
| Baseline Exp 3 |  4% |  7%   |  8%  | 15%    | 12%       |  5%          | 32         |
           

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

We carried experiments with the dynamic window for long with similar results as normal T5 vs the extended context. Extended context to fit the input is not worth it. Some states is better than none, but you don't need all of the states...
