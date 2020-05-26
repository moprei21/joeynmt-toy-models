# joeynmt-toy-models

This repo is just a collection of scripts showing how to install [JoeyNMT](https://github.com/joeynmt/joeynmt), preprocess
data, train and evaluate models.

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place and check out the correct branch:

    git clone https://github.com/moprei21/joeynmt-toy-models
    cd joeynmt-toy-models
    checkout ex5

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/download_install_packages.sh

Download and split data:

    ./scripts/download_data.sh

#Exercise 2

## sampling of training data
We first sample 100000 sentence pairs from each language to deliver a sampled training file  
Source language:

    shuf -n 100000 --random-source=train.de-nl.nl <(cat train.de-nl.de) > sample.train.de-nl.de
    wc sample.train.de-nl.de
Output:  
    100000 1491296 9788931 sample.train.de-nl.de  
    
Target language:  

    shuf -n 100000 --random-source=train.de-nl.nl <(cat train.de-nl.nl) > sample.train.de-nl.nl
    wc sample.train.de-nl.de
Output:  
    100000 1491296 9788931 sample.train.de-nl.de

## preprocessing of data with tokeniztion
Tokenized all necessary files with tokenizer.perl in tools/moses-scripts/scripts/tokenizer
  
sample.train.de-nl.de --> tokenized.sample.train.de-nl.de  
sample.train.de-nl.nl --> tokenized.sample.train.de-nl.nl  
test.de-nl.de --> tokenized.test.de-nl.de  
test.de-nl.nl --> tokenized.test.de-nl.nl    
dev.de-nl.de --> tokenized.dev.de-nl.de  
dev.de-nl.nl --> tokenized.dev.de-nl.nl  


## Training word level model
We copied the example config into the word_level.yaml and made the necessary adjustments for the model to work.  
These adjustments can be seen in configs/word_level.yaml  
After these adjustments the training begins: 

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 -m joeynmt train configs/word_level.yaml
    
## Evaluation of word level
Detokenization and bleu score evaluation:

    cat models/word_level/00002500.hyps.test | tools/moses-scripts/scripts/tokenizer/detokenizer.perl > data/detokenized.word_level.hyps.test
    cat data/detokenized.word_level.hyps.test | sacrebleu data/test.de-nl.nl
    
Outcome is to be found in the table at the end of the section  


## Training BPE-Models
For the BPE-Models to work we first need to learn the joint bpe and its vocab. If this is done then we are able to appy these new vocabs to build new training, test and dev files.  
Finally we have to build a single vocab file. If all these steps were executed correctly, then you are ready to train.  

mkdir bpe_vocab

Train different bpe models  
2000:

    subword-nmt learn-joint-bpe-and-vocab -i data/sample.tokenized.train.de-nl.de data/sample.tokenized.train.de-nl.nl --write-vocabulary data/bpe_vocab/vocab.2000.de data/bpe_vocab/vocab.2000.nl -s 2000 --total-symbols -o data/bpe_vocab/de-nl.2000.bpe
    subword-nmt apply-bpe -c data/bpe_vocab/de-nl.2000.bpe --vocabulary data/bpe_vocab/vocab.2000.de --vocabulary-threshold 10 <data/sample.tokenized.train.de-nl.de> data/bpe_vocab/sample.tokenized.bpe.2000-train.de-nl.de
    subword-nmt apply-bpe -c data/bpe_vocab/de-nl.2000.bpe --vocabulary data/bpe_vocab/vocab.2000.nl --vocabulary-threshold 10 <data/sample.tokenized.train.de-nl.nl> data/bpe_vocab/sample.tokenized.bpe.2000-train.de-nl.nl
    subword-nmt apply-bpe -c data/bpe_vocab/de-nl.2000.bpe --vocabulary data/bpe_vocab/vocab.2000.de --vocabulary-threshold 10 <data/tokenized.test.de-nl.de> data/bpe_vocab/tokenized.bpe.2000-test.de-nl.de
    subword-nmt apply-bpe -c data/bpe_vocab/de-nl.2000.bpe --vocabulary data/bpe_vocab/vocab.2000.nl --vocabulary-threshold 10 <data/tokenized.test.de-nl.nl> data/bpe_vocab/tokenized.bpe.2000-test.de-nl.nl
    subword-nmt apply-bpe -c data/bpe_vocab/de-nl.2000.bpe --vocabulary data/bpe_vocab/vocab.2000.nl --vocabulary-threshold 10 <data/tokenized.dev.de-nl.nl> data/bpe_vocab/tokenized.bpe.2000-dev.de-nl.nl
    subword-nmt apply-bpe -c data/bpe_vocab/de-nl.2000.bpe --vocabulary data/bpe_vocab/vocab.2000.de --vocabulary-threshold 10 <data/tokenized.dev.de-nl.de> data/bpe_vocab/tokenized.bpe.2000-dev.de-nl.de
    python3 tools/joeynmt/scripts/build_vocab.py data/bpe_vocab/sample.tokenized.bpe.2000-train.de-nl.de data/bpe_vocab/sample.tokenized.bpe.2000-train.de-nl.nl --output_path data/bpe_vocab/vocab.2000.de.nl
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 -m joeynmt train configs/bpe.2000.yaml


After training detokenization for calculation bleu 

    cat models/bpe.2000/00004000.hyps.test | tools/moses-scripts/scripts/tokenizer/detokenizer.perl > data/bpe_vocab/detokenized.2000.hyps.test
    cat data/bpe_vocab/detokenized.2000.hyps.test | sacrebleu data/test.de-nl.nl
    
This process can now be executed for different vocabulary sizes and lead to the following results:  


|    | use bpe | vocabulary size | BLEU |
|----|---------|-----------------|------|
| a) | no      | 2000            | 8.6  |
| b) | yes     | 2000            | 14.3 |
| c) | yes     | 3000            | 12.7 |
| d) | yes     | 1000            | 13.3 |

## Manual observations

###Word level:

    "Een paar jaar geleden , <unk> <unk> <unk> een <unk> ."
There are a lot of words with the label unknown in the hypothesis of the text. Otherwise the model did a very good job of translating as seen in this example:  
    
    "Dat kan een groot verschil maken ."
    
###BPE.2000:

Since this is the model with the highest Bleu Score its translations are quite solid. Here and there there are minor errors with sentence structure but otherwise it creates well readable sentences.
    
    Ze gaan terug gaan om ze te gaan . #three same verbs in the sentence
    We zijn in de zwellels verbazingwekkend , verbazingwekkend . #again repetition of words
    Heb een enkel in het publiek van iemand als je iemand in de publiek #again repetition of words but the gender is different :)
    
###BPE.3000:

This is the BPE Model with the lowest Bleu Score but still the translations are ok. Just like in BPE.2000 there are some errors but there are many sentences that are well readable.
    
    En prototypes bouwen , het prototypes .
    Ze gaan niet teruggaan . #same as sentence no.1 in BPE 2000 --> verbforms are not grammatically correct
    
###BPE.1000

The quality of this model is similar as the two models just described. There are errors but most of the translations are understandalbe and grammatically correct.

    Niet terug te gaan . #Same sentence as in the other models, but for me the most accurate representation. 
    Is het niet interessant dat hhe wetschappij een sterke effect hebben ?  #hhe is not an existing word
    
#Exercise 3


    


