python -W ignore main.py --word_replacer BERTReplacer --colloc_matrix_path CM_nltk_5gram_lowercase --colloc_metric PMI3 --colloc_thresh 1.0 --keep_case --top_k 2 --score orionw_sentences.xlsx orionw_sentences_bert.xlsx
python -W ignore main.py --word_replacer BERTReplacer --colloc_matrix_path CM_nltk_5gram_lowercase --colloc_metric PMI3 --colloc_thresh 1.0 --keep_case --bert_model_path bert_jokes --top_k 5 --score orionw_sentences.xlsx  orionw_sentences_bert_jokes.xlsx
python -W ignore main.py --word_replacer GensimCollocateReplacer --colloc_matrix_path CM_nltk_5gram_lowercase --colloc_metric PMI3 --colloc_thresh 1.0 --dist_thresh 0.3 --keep_case --keep_all --score orionw_sentences.xlsx orionw_sentences_colloc.xlsx
@REM python -W ignore main.py --word_replacer GensimCollocateReplacer --colloc_matrix_path CM_nltk_5gram_lowercase --colloc_metric PMI3 --colloc_thresh 1.0 --dist_thresh 0.3 --colloc_matrix_path CM_nltk_jokes --keep_case --keep_all --score orionw_sentences.xlsx orionw_sentences_jokecolloc.xlsx