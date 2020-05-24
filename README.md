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

    git clone https://github.com/bricksdont/joeynmt-toy-models
    cd joeynmt-toy-models
    checkout ex4

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/download_install_packages.sh

Download and split data:

    ./scripts/download_split_data.sh

Preprocess data:

    ./scripts/preprocess.sh

Then finally train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Evaluate a trained model with

    ./scripts/evaluate.sh

$ python3 tools/joeynmt/scripts/build_vocab.py data/sample.tokenized.train.de-nl.de data/sample.tokenized.train.de-nl.nl --output_path data/bpe_vocab/vocab.de.nl
$ subword-nmt learn-joint-bpe-and-vocab -i data/sample.tokenized.train.de-nl.de data/sample.tokenized.train.de-nl.nl --write-vocabulary data/bpe_vocab/de.vocab data/bpe_vocab/nl.vocab -s 2000 --total-symbols -o data/bpe_vocab/bpe.de-nl
Number of word-internal characters: 110
Number of word-final characters: 145
Reducing number of merge operations by 255
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/sample.tokenized.train.de-nl.> data/bpe_vocab/sample.tokenized.bpe.2000-train.de-nl.de
-bash: data/sample.tokenized.train.de-nl.: No such file or directory
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/sample.tokenized.train.de-nl.de> data/bpe_vocab/sample.tokenized.bpe.2000-train.de-nl.de
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/tokenized.test.de-nl.de> data/bpe_vocab/tokenized.test.bpe.2000-train.de-nl.de
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/tokenized.test.de-nl.de> data/bpe_vocab/tokenized.bpe.2000-test.de-nl.de
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/tokenized.test.de-nl.nl> data/bpe_vocab/tokenized.bpe.2000-test.de-nl.nl
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/tokenized.dev.de-nl.nl> data/bpe_vocab/tokenized.bpe.2000-dev.de-nl.nl
$ subword-nmt apply-bpe -c data/bpe_vocab/bpe.de-nl --vocabulary data/bpe_vocab/de.vocab --vocabulary-threshold 10 <data/tokenized.dev.de-nl.de> data/bpe_vocab/tokenized.bpe.2000-dev.de-nl.de
$ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 -m joeynmt train configs/bpe.2000.yaml