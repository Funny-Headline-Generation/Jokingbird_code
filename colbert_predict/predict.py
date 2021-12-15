import pandas as pd
import numpy as np
import keras
import transformers
from transformers import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


training_sample_count = 160000
training_epochs = 3
test_count = 40000
running_folds = 1
MAX_SENTENCE_LENGTH = 20
MAX_SENTENCES = 5
MAX_LENGTH = 100


def return_id(str1, str2, truncation_strategy, length, tokenizer):

    inputs = tokenizer.encode_plus(str1, str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy)

    input_ids =  inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, column, tokenizer):
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])
    
    for _, row in df[[column]].iterrows():
        i = 0
        
        # sent
        sentences = sent_tokenize(row[column])
        for xx in range(MAX_SENTENCES):
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, None, 'longest_first', MAX_SENTENCE_LENGTH, tokenizer)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1
        
        # full row
        ids_q, masks_q, segments_q = return_id(row[column], None, 'longest_first', MAX_LENGTH, tokenizer)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)
        
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)
        
    #print(model_input[0].shape)
    return model_input


def predict_score(df, column, model_path, batch_size=1):
    df_train = df
    
    MODEL_TYPE = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
    model = keras.models.load_model(model_path)
    
    list_df = [df_train[i:i+batch_size] for i in range(0,df_train.shape[0],batch_size)]

    all_preds = list()
    for i in tqdm(range(len(list_df)), total=len(list_df)):
        inputs = compute_input_arrays(list_df[i], column, tokenizer)
        pred = model.predict(inputs)
        for j in range(len(pred)):
            all_preds += [pred[j][0]]
    
    return all_preds
    

if __name__ == '__main__':
    input_category = 'text'

    df_train = pd.DataFrame(column=['text'])

    df_train = df_train.append({'text':'just simple text'}, ignore_index=True)
    df_train = df_train.append({'text':'olololol this is joke so petty'}, ignore_index=True)

    batch_size = 1

    all_preds = predict_score(df_train, input_category, 'colbert')
    
    print(all_preds)