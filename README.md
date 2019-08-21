# sentiment analysis

## Dataset Info

### Fake-News Detection

This dataset is available at this [Github repo](https://github.com/Tariq60/LIAR-PLUS).

<br><br>
This dataset has evidence sentences extracted automatically from the full-text verdict report written by journalists in Politifact. Objective of this dataset is to provide a benchmark for evidence retrieval and show empirically that including evidence information in any automatic fake news detection method (regardless of features or classifier) always results in superior performance to any method lacking such information.
<br><br>
Below is the description of the TSV file taken as is from the original [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), which was published in this [paper](https://www.aclweb.org/anthology/P17-2067). A new column is added at the end that has the extracted justification.
<br>
- Column 1: the ID of the statement ([ID].json).
- Column 2: the label.
- Column 3: the statement.
- Column 4: the subject(s).
- Column 5: the speaker.
- Column 6: the speaker's job title.
- Column 7: the state info.
- Column 8: the party affiliation.
- Columns 9-13: the total credit history count, including the current statement.
  - 9: barely true counts.
  - 10: false counts.
  - 11: half true counts.
  - 12: mostly true counts.
  - 13: pants on fire counts.
- Column 14: the context (venue / location of the speech or statement).
- **Column 15: the extracted justification**

The justification extraction method is done as follows:
- Get all sentences in the 'Our Ruling' section of the report if it exists or get the last five sentences.
- Remove any sentence that have the verdict and any verdict-related words. Verdict-related words are provided in the forbidden words file.
