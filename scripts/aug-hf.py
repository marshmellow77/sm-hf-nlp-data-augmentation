import os
import io
import sys
import time
import torch
import shutil
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def back_translate(data, src2tar, tar2src):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src2tar_tokenizer = AutoTokenizer.from_pretrained(src2tar)
    src2tar_model = AutoModelForSeq2SeqLM.from_pretrained(src2tar).to(device)
    tar2src_tokenizer = AutoTokenizer.from_pretrained(tar2src)
    tar2src_model = AutoModelForSeq2SeqLM.from_pretrained(tar2src).to(device)
    
    translations = []
    dataloader = DataLoader(data, batch_size = 16, shuffle = False)
    
    for _, examples_chunk in enumerate(dataloader):
        examples_chunk = [text for text in examples_chunk]
        batch = src2tar_tokenizer(
            examples_chunk, 
            return_tensors="pt", 
            truncation=True, 
            padding="longest"
        ).to(device)
        src2tar_outputs = src2tar_model.generate(
            input_ids = batch.input_ids, 
            attention_mask=batch.attention_mask
        )
        tar2src_outputs = tar2src_model.generate(src2tar_outputs)
        dec = tar2src_tokenizer.batch_decode(
            tar2src_outputs, 
            skip_special_tokens = True,
        )
        translations += dec
    return translations

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str)
    parser.add_argument("--output-file-name", type=str)
    parser.add_argument("--language", type=str)
    args, _ = parser.parse_known_args()
    
    # Set up Logging
    logger = logging.getLogger(__name__)
    marker = '===== ===== ===== ====='
    
    logging.basicConfig(
        level = logging.getLevelName('INFO'),
        handlers = [logging.StreamHandler(sys.stdout)],
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    input_dir = '/opt/ml/processing/input'
    output_dir = '/opt/ml/processing/output'
    
    logger.info("Loading test input data")
    raw_train_dataset = load_dataset("csv", data_files = os.path.join(input_dir, args.file_name))["train"]
    
    df_dataset = raw_train_dataset.to_pandas()
    data = raw_train_dataset['text']
    logger.info('Done loading input data')
    
    models = []
    if args.language == 'de':
        models = ['Helsinki-NLP/opus-mt-de-en', 'Helsinki-NLP/opus-mt-en-de', 
                  'Helsinki-NLP/opus-mt-de-es', 'Helsinki-NLP/opus-mt-es-de', 
                  'Helsinki-NLP/opus-mt-de-fr', 'Helsinki-NLP/opus-mt-fr-de',
                  'Helsinki-NLP/opus-mt-de-it', 'Helsinki-NLP/opus-mt-it-de']
    elif args.language == 'en':
        models = ['Helsinki-NLP/opus-mt-en-de','Helsinki-NLP/opus-mt-de-en',
                  'Helsinki-NLP/opus-mt-en-es','Helsinki-NLP/opus-mt-es-en',
                  'Helsinki-NLP/opus-mt-en-fr','Helsinki-NLP/opus-mt-fr-en',
                  'Helsinki-NLP/opus-mt-en-ru','Helsinki-NLP/opus-mt-ru-en']
    
    start_time = time.time()
    
    for i in range(int(len(models)/2)):
        logger.info(len(data))
        src2tar, tar2src = models[i*2], models[i*2+1]
        
        translations = back_translate(data, src2tar, tar2src)
        data += translations
        current_size = df_dataset.shape[0]
        df_dataset = df_dataset.append(pd.DataFrame({'text':translations}))
        df_dataset['ID'][current_size:] = df_dataset['ID'][:current_size]
        df_dataset['label'][current_size:] = df_dataset['label'][:current_size]
        
    duration = time.time() - start_time
    
    # Deduplicate data
    df_dataset.drop_duplicates(inplace=True,subset=['text'])
    logging.info(f'\n\n\n\n {marker} Done with translation. Final dataset size is {df_dataset.shape[0]}, took {duration} seconds to process {marker}\n\n\n\n')
    
    # Save to disk in output_dir, which will uplodaded to S3 for us 
    df_dataset.to_csv(f'{output_dir}/{args.output_file_name}',index=False)

    