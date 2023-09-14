---
tags:
- dialogue
- rdf
- dst
task_categories:
- conversational
- text-generation
- text2text-generation
task_ids:
- dialogue-state-tracking
- natural-language-generation
maintainers:
- name: Morgan Veyret
  email: morgan.veyret@orange.com
license:
- GPL 3.0
- MIT
- 'Attribution 2.0 UK: England & Wales'
- CC BY-NC-SA 4.0
language:
- fr
access: public
dataset_info:
- config_name: default
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: multi_relations
      dtype: bool
    - name: triples
      list: string
  splits:
  - name: train
    num_bytes: 8058582
    num_examples: 2017
  download_size: 1304809
  dataset_size: 8058582
- config_name: all
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
  splits:
  - name: train
    num_bytes: 76437986
    num_examples: 13004
  - name: test
    num_bytes: 5207558
    num_examples: 969
  - name: validation
    num_bytes: 21670028
    num_examples: 3872
  download_size: 9298744
  dataset_size: 103315572
- config_name: dstc2
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 8761687
    num_examples: 1694
  - name: test
    num_bytes: 7161218
    num_examples: 1117
  - name: validation
    num_bytes: 2090951
    num_examples: 424
  download_size: 1187728
  dataset_size: 18013856
- config_name: sfxdial
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 6666994
    num_examples: 1613
  - name: test
    num_bytes: 341292
    num_examples: 81
  - name: validation
    num_bytes: 1372201
    num_examples: 323
  download_size: 612217
  dataset_size: 8380487
- config_name: multiwoz
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 48402208
    num_examples: 7673
  - name: test
    num_bytes: 2416211
    num_examples: 384
  - name: validation
    num_bytes: 9795769
    num_examples: 1535
  download_size: 7209947
  dataset_size: 60614188
- config_name: camrest-100
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
  splits:
  - name: train
    num_bytes: 321972
    num_examples: 80
  - name: test
    num_bytes: 65529
    num_examples: 16
  - name: validation
    num_bytes: 13651
    num_examples: 4
  download_size: 28632
  dataset_size: 401152
- config_name: camrest-1000
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
  splits:
  - name: train
    num_bytes: 3114529
    num_examples: 800
  - name: test
    num_bytes: 632021
    num_examples: 160
  - name: validation
    num_bytes: 147138
    num_examples: 40
  download_size: 271535
  dataset_size: 3893688
- config_name: multiwoz-100
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
  splits:
  - name: train
    num_bytes: 266130
    num_examples: 80
  - name: test
    num_bytes: 41433
    num_examples: 16
  - name: validation
    num_bytes: 7406
    num_examples: 4
  download_size: 26096
  dataset_size: 314969
- config_name: multiwoz-1000
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
  splits:
  - name: train
    num_bytes: 2714601
    num_examples: 800
  - name: test
    num_bytes: 581451
    num_examples: 160
  - name: validation
    num_bytes: 122921
    num_examples: 40
  download_size: 265415
  dataset_size: 3418973
- config_name: camrest-sim
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 3204032
    num_examples: 800
  - name: test
    num_bytes: 156002
    num_examples: 40
  - name: validation
    num_bytes: 640300
    num_examples: 160
  download_size: 298449
  dataset_size: 4000334
- config_name: multiwoz-sim
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 2759525
    num_examples: 800
  - name: test
    num_bytes: 138251
    num_examples: 40
  - name: validation
    num_bytes: 582298
    num_examples: 160
  download_size: 292631
  dataset_size: 3480074
- config_name: bundle-converted
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 63830889
    num_examples: 10980
  - name: test
    num_bytes: 9918721
    num_examples: 1582
  - name: validation
    num_bytes: 13258921
    num_examples: 2282
  download_size: 9009892
  dataset_size: 87008531
- config_name: bundle-simulated
  features:
  - name: dialogue_id
    dtype: string
  - name: turns
    list:
    - name: id
      dtype: int8
    - name: speaker
      dtype: string
    - name: text
      dtype: string
    - name: rdf-acts
      list: string
  - name: states
    list:
    - name: id
      dtype: int8
    - name: multi_relations
      dtype: bool
    - name: triples
      list:
        list: string
    - name: turn_ids
      list: int8
  splits:
  - name: train
    num_bytes: 5963557
    num_examples: 1600
  - name: test
    num_bytes: 294253
    num_examples: 80
  - name: validation
    num_bytes: 1222598
    num_examples: 320
  download_size: 591080
  dataset_size: 7480408
---

# Dataset Card for rdfdial

**Required packages:** `python-gitlab`

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://gitlab.tech.orange/NEPAL/task-oriented-dialogue/rdfdial
- **Repository:** https://gitlab.tech.orange/NEPAL/task-oriented-dialogue/rdfdial
- **Paper:** N/A
- **Leaderboard:** N/A
- **Point of Contact:** morgan.veyret@orange.com

### Dataset Summary

This dataset provides dialogues annotated in dialogue acts and dialogue
state in and RDF based formalism.

There is a conversion of `sfxdial`, `dstc2` and `multiwoz2.3` datasets
as well as two fully synthetic datasets created from simulated conversations:
`camrest-sim` and `multiwoz-sim`.

Original dataset before conversion are available here:
- DSTC2: https://github.com/matthen/dstc
- Multiwoz 2.3: https://github.com/thu-coai/ConvLab-2/tree/master/data/multiwoz2.3
- SfxDial: https://www.repository.cam.ac.uk/items/62011578-23d4-4355-8878-5a150fb72b43

### Supported Tasks and Leaderboards

This dataset was used for the following tasks:
- Natural Language Generation
- Dialogue State Tracking

### Languages

This dataset includes the following languages:
- English

## Dataset Structure

### Data Instances


For all datasets, each item has this schema:

```python
{
    "dialogue_id": "string",         # dialog identifier
    "turns": [{                      # list of dialog turns
        "id": "int8",                # dialog turn index in the conversation
        "speaker": "string",         # speaker identifier ('user' or 'system')
        "text": "string",            # speaker utterance
        "rdf-acts": ["string"],      # string representation of dialog acts
    }],
    "states": [{                     # dialog states for each turn
        "id": "int8",
        "multi_relations": "bool",   # are multiple instances of relations allowed ?
        "triples": [["string"]],     # triples representing the state
        "turn_ids": ["int8"],        # ids of turns contributing to this state
    }],
}
```

### Data Fields

For each dataset item, the following fields are provided:
- `dialogue_id`: unique dialogue identifier
- `turns`: list of speech turns, each turn contains the following fields:
  - `id`: turn index in the dialogue
  - `speaker`: identifier for the speaker (`user` or `system`)
  - `text`: turn utterance
  - `rdf-acts`: list of dialogue acts using string representation of rdf formalism
  each act has the form: `act(triple;...)` where `triple` is formatted as
  `(subject,predicate,object)`
- `states`: list of states for the dialogue, each entry contains the following fields:
  - `id`: state index in the dialogue
  - `multi_relations`: boolean indicating if multiple instances of the same predicate are
  allowed or not
  - `triples`: list of triples representing the graph state, each triple is a list of 3 string like
  `[subject,predicate,object]`
  - `turn_ids`: list of turn ids that contributed to this state


### Data Splits

For each dataset, splits were generated randomly in the following proportions:
- *train*: 80%
- *validation*: 16%
- *test*: 4%

## Dataset Creation

### Curation Rationale

This dataset has been created to work with graph base dialog state representation using
generative models (T5 family).

### Source Data

#### Initial Data Collection and Normalization

- *Converted datasets*:
  - DSTC2: https://github.com/matthen/dstc
  - Multiwoz 2.3: https://github.com/thu-coai/ConvLab-2
  - SfxDial: https://www.repository.cam.ac.uk/handle/1810/251304
- *Synthetic datasets*: rule-based simulations

#### Who are the source language producers?

- *Converted datasets*: see original datasets documentation
- *Synthetic datasets*: conversations were generated using an agenda-based user simulator and
a rule based agent working directly with dialogue acts. These conversations were then augmented
with natural language user/system utterances. Natural language generation was done using
a T5-base model fine-tuned on the converted datasets.

### Annotations

#### Annotation process

- *Converted datasets*: rule-based conversion of the user/system dialogue acts from slot-value
to RDF based format. The dialogue state is created automatically using another rule based
tracked working with triples. Some conversations could not be converted automatically and/or
contained wrong/confusing annotations and were removed from the dataset compared to the
original ones.
- *Synthetic datasets*: simulation work at the annotation level and the dataset was augmented
to include natural language information.

#### Who are the annotators?

All annotations were generated automatically.

For dialogue acts:
- converted data: rules were applied to convert slot-value based dialogue acts
to rdf-based ones
- synthetic data: rdf-based dialogue acts were directly generated by the dialogue simulation.

For dialogue states, a rule based system was using taking rdf-based dialogue acts as
its inputs.

### Personal and Sensitive Information

This dataset does not contains any personal or sensitive information.

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

Converted datasets follow their original licenses:
- DSTC2: [GPL 3.0](https://github.com/matthen/dstc/blob/master/LICENSE)
- Multiwoz 2.3: [Apache 2.0](https://github.com/thu-coai/ConvLab-2/blob/master/LICENSE)
- SfxDial: [Attribution 2.0 UK: England & Wales](https://creativecommons.org/licenses/by/2.0/uk/)

Simulated conversation are provided with the following licenses:
- camrest-sim: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
- multiwoz-sim: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

### Citation Information

[More Information Needed]

### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.
