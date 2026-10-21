# Named Entity Recognition (NER) from News Articles

A natural language processing (NLP) project for identifying and
categorizing named entities in English news articles using rule-based
methods and spaCy models trained or evaluated on the **CoNLL-2003**
dataset.

## Project Overview

This project explores Named Entity Recognition (NER), a task that
detects mentions of people, organizations, locations, and miscellaneous
named entities in text. It loads the CoNLL-2003 train, validation, and
test splits, analyzes the data with Pandas, builds rule-based baselines,
compares pretrained spaCy models, and trains a custom spaCy NER model
using the dataset's own entity labels.

## Objectives

-   Load and explore the CoNLL-2003 English dataset.
-   Convert BIO tagging sequences into entity spans.
-   Analyze sentence lengths, entity frequencies, and label
    distributions using Pandas.
-   Build a rule-based NER system using a gazetteer extracted from the
    training data.
-   Add heuristic rules for recognizing some organization and location
    names.
-   Evaluate pretrained spaCy models and compare their predictions.
-   Train a custom spaCy NER model with the CoNLL-2003 entity labels.
-   Evaluate entity-level Precision, Recall, and F1-score.
-   Visualize and categorize detected entities using spaCy's displaCy.
-   Inspect model disagreements and common missed entities.
-   Save the best custom model and create a ZIP archive for reuse.

## Dataset

The project uses **CoNLL-2003 (English)**, distributed in a Kaggle
dataset commonly identified as:

`alaakhaled/conll003-englishversion`

The notebook expects these files:

-   `train.txt` --- training sentences and entity tags.
-   `valid.txt` --- validation sentences and entity tags.
-   `test.txt` --- held-out test sentences and entity tags.

Each non-empty line contains token-level fields, with the token in the
first column and its BIO named-entity tag in the last column. Sentences
are separated by blank lines, and document boundaries may be marked with
`-DOCSTART-`.

The target entity categories are:

-   `PER` --- Person
-   `ORG` --- Organization
-   `LOC` --- Location
-   `MISC` --- Miscellaneous entity

The notebook first checks the preferred Kaggle input path and then
searches common Kaggle input directories for the required files. If
running locally, update the dataset path or place the files in a
location that the notebook can find.

## Workflow

1.  **Data loading** --- locate and read `train.txt`, `valid.txt`, and
    `test.txt`.
2.  **Data exploration** --- inspect sentence and token counts, entity
    distributions, and examples using Pandas.
3.  **BIO conversion** --- convert BIO tags into token-level entity
    spans.
4.  **Rule-based NER** --- build an `EntityRuler` gazetteer from
    entities found in the training split.
5.  **Heuristic rules** --- add simple patterns for selected
    organization and location name endings.
6.  **Pretrained model comparison** --- evaluate `en_core_web_sm` and
    `en_core_web_md` when available.
7.  **Label mapping** --- map compatible spaCy OntoNotes labels to the
    CoNLL-2003 labels for comparison.
8.  **Custom model training** --- train a spaCy NER pipeline on the
    training split using `PER`, `ORG`, `LOC`, and `MISC`.
9.  **Validation and checkpoint selection** --- calculate validation F1
    after each epoch, keep the best checkpoint, and stop early when
    validation performance no longer improves.
10. **Final evaluation** --- compare the rule-based systems, pretrained
    models, and custom model on the held-out test split.
11. **Error analysis and visualization** --- inspect entity-level
    results, missed entities, and highlighted predictions using
    displaCy.
12. **Model export** --- save the best custom model to
    `conll_custom_ner_best/` and package it as
    `conll_custom_ner_best.zip`.

## Technologies

-   Python
-   Jupyter Notebook
-   Pandas and NumPy
-   Matplotlib
-   spaCy
-   spaCy `EntityRuler`
-   spaCy `displaCy`
-   CoNLL-2003 BIO tagging format
-   Precision, Recall, and F1-score for entity-level evaluation

## Run the Project

1.  Download or clone the project repository.

2.  Download the CoNLL-2003 English dataset and make sure `train.txt`,
    `valid.txt`, and `test.txt` are available.

3.  Install the required libraries:

    ``` bash
    pip install spacy pandas numpy matplotlib jupyter
    ```

4.  Open the notebook:

    ``` bash
    jupyter notebook "named-entity-recognition-ner-from-news-articles.ipynb"
    ```

5.  Run the cells from top to bottom.

The notebook attempts to load or download the pretrained spaCy pipelines
`en_core_web_sm` and `en_core_web_md`. Internet access may be required
if these models are not already installed. In Kaggle, enable Internet in
notebook settings if downloads are needed.

To install the pretrained models manually, you can run:

``` bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

The custom model training can take several minutes or longer depending
on the environment. If runtime is limited, reduce `N_EPOCHS` in the
training configuration. The notebook selects the best checkpoint using
validation F1.

## Project Structure

A typical project layout may look like this:

``` text
NER-News-Articles/
├── named-entity-recognition-ner-from-news-articles.ipynb
├── README.md
└── data/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

The dataset files may instead be supplied through the supported Kaggle
input directory. After running the training and export cells, the
working directory will also contain:

``` text
conll_custom_ner_best/
└── [saved spaCy model files]

conll_custom_ner_best.zip
```

## Evaluation Notes and Limitations

-   The notebook reports measured Precision, Recall, and F1-score when
    it is run; no final score is assumed in advance.
-   Evaluation uses entity-level exact matching: a prediction is correct
    only when both the entity boundaries and its label match the gold
    annotation.
-   The rule-based gazetteer is based on entities observed in the
    training set and may miss unseen names or confuse entities that
    appear in different contexts.
-   The heuristic rules cover only selected naming patterns and are not
    a complete language understanding system.
-   The pretrained spaCy models use a different entity-label scheme from
    CoNLL-2003. Mapping their labels to CoNLL categories is approximate,
    so comparisons should be interpreted carefully.
-   The custom model is selected using the validation split. The test
    split should be reserved for final evaluation.
-   Results depend on the dataset version, software versions, available
    pretrained models, and training configuration.
-   This is an educational NLP project, not a fully reliable production
    entity-recognition service.

## Future Improvements

-   Tune training settings and evaluate multiple random seeds.
-   Add more systematic error analysis for each entity category.
-   Test on news from sources and domains outside CoNLL-2003.
-   Compare with transformer-based NER models.
-   Create a reusable prediction function or web interface for new
    articles.
-   Add automated tests and document reproducible environment versions.

## Author

**Raouf Ahmed**
