@REM python -W ignore main.py --word_selector BERTTokenSelector --word_replacer BERTReplacer --colloc_thresh 0.1 --top_k 5 --score hundred_lines_truecased.xlsx hundred_lines_bertselector.xlsx
@REM python -W ignore main.py --word_replacer BERTReplacer --top_k 5 --keep_case --score hundred_lines_truecased.xlsx hundred_lines_collocselector.xlsx
@REM python -W ignore main.py --word_replacer BERTReplacer --top_k 3 --score thousand_lines_truecased.xlsx thousand_lines_bertselector.
python -W ignore main.py --word_replacer BERTReplacer --colloc_thresh 0.3 --top_k 5 --score thousand_lines_truecased.xlsx thousand_lines_bertselector_top5.xlsx