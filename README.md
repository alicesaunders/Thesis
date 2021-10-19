# Thesis

The files in this repo are all relevant for the submission of my thesis for MSc Data Science. 

adversarial_coref_spanbert_mine.jsonnet is the config file used to train the spanBERT coreference model with an adversary using the OntoNotes dataset. This config file is necessary for the use of AllenNLP. 

adversarial_mlm.jsonnet is the config file used to train BERT on the task of MLM with an adversary using the AllenNLP library. 

coref_spanbert_base_mine.jsonnet is the config file needed to train a spanBERT-base model on the task of coreference resolution with the AllenNLP library. 

SpanBERT_BiasMA_new.jsonnet is the config file used to train the spanBERT model on the task of coreference resolution using the BiasMitigatorApplicator from AllenNLP. 

coref_adv.py is the module used by AllenNLP to train the coreference model, with some edits of my own to attempt to make it work with an adversary. 

masked_language_model_ALS.py is the script based on the AllenNLP library that is used for training the masked language model. 

masked_language_modelling_ALS.py is the script for the dataset reader based on the AllenNLP library. 

ALLEN_NLP_TRAINING.ipynb is the script used to train AllenNLP models using the CLI. 

EXTRACTING_EMBEDDINGS_ADVERSARY.ipynb is the script used to extract embeddings that could be used as input to the adversary, in line with Approach 1.1 and 1.2. 

EXTRACTING_EMBEDDINGS_USING_WINOBIAS.ipynb is the script used to extract embeddings from the WinoBias dataset that were used to evaluate bias in BERT. 

ONTONOTES_BERT.ipynb is the script used to edit the original OntoNotes data that ensures it is in the correct format for MLM. 

PREDICTIONS_ALLENNLP.ipynb is the script used to make predicitions with the models trained using the AllenNLP framework. 

WINOBIAS_MLM.ipynb is the script used to train BERT on the task of MLM using the WinoBias data. 

WINOBIAS_MLM_ADVERSARY.ipynb is the script used to attempt to implemented adversarial debiasing with the MLM not using the AllenNLP framework. 

