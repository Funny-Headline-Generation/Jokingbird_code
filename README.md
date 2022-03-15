# Jokingbird: Funny headline generation for news

Output on 1000 random headlines from our best model - https://github.com/Funny-Headline-Generation/Jokingbird_code/blob/main/BertHumEditOutput.xlsx?raw=true

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
      
