# Results

CURRENT UP-TO-DATE TABLES ARE ON MY OVERLEAF  DOCUMENT. I WILL UPDATE THIS README ACCORDINGLY IN LATE AUGUST

https://www.overleaf.com/read/qjkcwzskzdpr

Storing CSVs with results from relevant results

--- UPDATED RESULTS WITH GREEDY SEARCH FOR BASELINE AND BEAM SEARCH FOR GRAPH. SCORES MAKE MUCH MORE SENSE NOW
-- TABLE BEING UPDATED, ACTUAL RESULTS IN THE OVERLEAF. WILL UPDATE THIS AFTER I'M DONE WITH THE REPORT

MULTIWOZ RDF (convlab 2.3)
| Model                 | JGA | JG-F1 | GLEU | METEOR | JG-RECALL | JG-PRECISION | Batch size |
|-----------------------|-----|-------|------|--------|-----------|--------------|------------|
| Baseline Exp 1        | 37% | 32%   | 17%  | 35%    | 65%       | 21%          | 16         |
| Baseline Exp 2        | 24% | 29%   | 18%  | 32%    | 53%       | 20%          | 16         |
| Baseline Exp 3        | 17% | 10%   | 10%  | 20%    | 27%       |  6%          | 32         |
| T5-base. Exp 1        | 76% | 72%   | 57%  | 64%    | 98%       | 57%          | 2          |
| T5-base. Exp 2        | 53% | 69%   | 54%  | 63%    | 95%       | 54%          | 4          |
| T5-base. Exp 3        | 16% | 59%   | 58%  | 60%    | 69%       | 51%          | 2          |
| flanT5-base. Exp 1    | 60% | 83%   | 71%  | 76%    | 95%       | 74%          | 2          |
| flanT5-base. Exp 2    | 48% | 70%   | 55%  | 64%    | 94%       | 56%          | 6          |
| longlocal-base. Exp 1 | 61% | 83%   | 73%  | 72%    | 97%       | 74%          | 4          |
| longlocal-base. Exp 2 | 42% | 80%   | 70%  | 71%    | 92%       | 71%          | 6          |
| tglobal-base. Exp 1   | 63% | 86%   | 77%  | 76%    | 95%       | 78%          | 4          |
| tglobal-base. Exp 2   | 49% | 76%   | 61%  | 66%    | 94%       | 63%          | 6          |
| T5-large. Exp 1       | 72% | 82%   | 69%  | 70%    | 97%       | 70%          | 2          |
| T5-large. Exp 2       | 49% | 73%   | 59%  | 66%    | 94%       | 59%          | 4          |

DYNAMIC WINDOW: These models use a max size of 1024 as the T5 setup with a dynamic window, instead of 2048 tokens. We obtain better results, i.e. longer input may degrade performance
| Model                   | JGA | JG-F1 | GLEU | METEOR | JG-RECALL | JG-PRECISION | Batch size |
|-------------------------|-----|-------|------|--------|-----------|--------------|------------|
| flanT5-base. Exp 1      | 75% | 76%   | 49%  | 64%    | 98%       | 61%          | 2          |
| longlocal-base. Exp 1   | 70% | 84%   | 75%  | 75%    | 97%       | 75%          | 2          |
| longtglobal-base. Exp 1 | 76% | 78%   | 64%  | 69%    | 97%       | 65%          | 2          |

PEFT TABLE: Results only available for baseline, interesting finding
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
