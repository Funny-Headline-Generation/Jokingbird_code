# Jokingbird: Funny headline generation for news

## About

This is the source code for the paper "Jokingbird: Funny Headline Generation for News" by Nikita Login, Alexander Baranov and Pavel Braslavski (To be published in LNCS 13217)

The research was done as a part of master's course at HSE University in 2020-2021

### Authors

Nikita Login, HSE University, Moscow, Russia

Alexander Baranov, HSE University, Moscow, Russia

Pavel Braslavski, Ural Federal University

Research supervisor: Pavel Braslavsky, Ural Federal University

Academic supervisor: Anastasia Bonch-Osmolovskaya, HSE University, Moscow, Russia

## Abstract

In this study, we address the problem of generating funny
headlines for news articles. Funny headlines are beneficial even for seri-
ous news stories – they attract and entertain the reader. Automatically
generated funny headlines can serve as prompts for news editors. More
generally, humor generation can be applied to other domains, e.g. con-
versational systems. Like previous approaches, our methods are based
on lexical substitutions. We consider two techniques for generating sub-
stitute words: one based on BERT and another based on collocation
strength and semantic distance. At the final stage, a humor classifier
chooses the funniest variant from the generated pool. An in-house eval-
uation of 200 generated headlines showed that the BERT-based model
produces the funniest and in most cases grammatically correct output.

## Data

Our input and training data was from publicly available datasets:

Times front page news - https://components.one/datasets/above-the-fold/

All the news - https://www.kaggle.com/snapcrack/all-the-news

Harvard news articles - https://doi.org/10.7910/DVN/GMFCTR

RedditJokes [1] - https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection

Humicroedit [2] - https://cs.rochester.edu/u/nhossain/funlines.html

FunLines [3] - https://cs.rochester.edu/u/nhossain/funlines.html


Output of our best model (BERTHumEdit)  on 1000 headlines from our input dataset is available here:

https://github.com/Funny-Headline-Generation/Jokingbird_code/blob/main/BertHumEditOutput.xlsx?raw=true

## References

1. Annamoradnejad, I., Zoghi, G.: ColBERT: Using BERT sentence embedding for humor detection. arXiv preprint arXiv:2004.12765 (2020)
2. Hossain, N., Krumm, J., Gamon, M.: “President Vows to Cut <Taxes>Hair”: Dataset and analysis of creative text editing for humorous headlines. In: Proceedings of the 2019 Conference of the North American Chapter of the Association forComputational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 133–142 (2019)
3. Hossain, N., Krumm, J., Sajed, T., Kautz, H.: Stimulating creativity with FunLines: A case study of humor generation in headlines. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations. pp. 256–262 (2020)

## Quickstart

1. Clone this repository.

2. Download the following folders and files and put them into repository folder:

Variables for humor classifier (put it in "new_colbert_predict/colbert-trained") - https://drive.google.com/drive/folders/157uwBlLrOwJgsgQD8EU36N94qZlzlTKE?usp=sharing

Collocation matrix (trained on joke corpus) - https://drive.google.com/drive/folders/1q0Z5-pLicPTTX_YlCHxkSkSVliPPI_Yt?usp=sharing

Collocation matrix (trained on news headlines) - https://drive.google.com/drive/folders/1q0Z5-pLicPTTX_YlCHxkSkSVliPPI_Yt?usp=sharing

Collocation matrix (trained on news body, slow) - https://drive.google.com/drive/folders/1q0Z5-pLicPTTX_YlCHxkSkSVliPPI_Yt?usp=sharing



Word2Vec model - https://drive.google.com/drive/folders/17vj8Ciu0bf_rtrfbuag3NQlAiKcgkwkJ?usp=sharing



Bert model (trained on Humicroedit/Funlines) - https://drive.google.com/file/d/1IngGcanB9pviw_-8Rd-GUsCfEDzVjCss/view?usp=sharing

Bert model (trained on joke corpus) - https://drive.google.com/file/d/1WYLu0XSC5MUrY2RxGI_fNt1N5pLMElNu/view?usp=sharing


3. Install dependencies:

<code>pip install -r requirements.txt</code>

4. Create a table (.CSV, .XLSX) with column named "headline" where you headlines will be

5. Run the script on your file:

<code>python main.py my_input_file.xlsx my_output_file.xlsx</code>

To see the possible list of options, type:

<code>python main.py --help</code>

Some of the key command line arguments

 -- word_replacer - Which algorithm to use for masked word replacement (BERT Replacer or DistReplacer)
 
 -- bert_model_path - Path to saved bert model for BERTReplacer
 
 -- colloc_matrix_path - Path to saved collocation matrix
 
 -- keep_case - Whether not to lowercase before identifying collocations (default: False)
 
 -- keep_all - Whether to consider all possible options that exceed collocation strength and semantic distance thresholds in GensimCollocateReplacer (default: False)
 
 -- score - Whether to score elements on output with humour classifier and keep only the most funny variant of a sentence (default: False)
 
 -- top_k int - Whether to sample top-k BERT predictions and pass them to next level instead of selecting only most probable one (default: 3)
 
 -- colloc_thresh - Threshold of collocation strength for n-grams to be considered collocations (default: 3.0)
 
 -- dist_thresh - Threshold of Word2Vec cosine distance (between 0 and 1) for DistReplacer (default: 0.4)
 
 -- colloc_metric - Which collocation strength metric to use (default: that of saved collocate matrix, PMI in files provided above,
                                                              supported: PMI, LL (Log-likelihood), Jaccard, Dice, TScore)

The workflow of the algorithm is illustrated below:

      headline
         ||
         ||
         ||
     Select and mask words to be replaced
       Collocation matrix
         ||
         ||
     Select words to be inserted as replacement for masked
      ||                                     ||
      ||                                     ||
    DistReplacer                        BERTReplacer
      Collocation matrix                    BERT model
      Word2Vec model                         ||
      ||                                     ||
      ||                                     ||
    Humor classifier                     Humor classifier
      ||                                     ||
      ||                                     ||
    Select most funny variant           Select most funny variant
      ||                                     ||
      ||                                     ||
    Output                                Output
      
