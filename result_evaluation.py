import json, argparse
import pyarrow.parquet as pq
from kgw_watermark import KGWHierarchicalSummarizer
import tqdm
from bert_score import score
from rouge_score import rouge_scorer
import os
import pandas as pd
import csv
from pathlib import Path
import ast
import time

# $env:HF_ENDPOINT="https://hf-mirror.com"
# Run the command above in PowerShell to use the mirror site for downloading BERT models.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

columns = ["z_score_wa", "z_score_un", "z_score_hi",
      "r1_wa_p", "r1_wa_r", "r1_wa_f", "r1_un_p", "r1_un_r", "r1_un_f",
      "r2_wa_p", "r2_wa_r", "r2_wa_f", "r2_un_p", "r2_un_r", "r2_un_f",
      "rl_wa_p", "rl_wa_r", "rl_wa_f", "rl_un_p", "rl_un_r", "rl_un_f",
      "b_wa_p", "b_wa_r", "b_wa_f", "b_un_p", "b_un_r", "b_un_f"
      ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, default='cuda:0')
    parser.add_argument("-llms", type=str, default="'t5-small', 't5-base', 'flan-t5-base', 'flan-t5-small',\
                                          'bart-large-cnn', 'bart-large-xsum'")
    parser.add_argument("-data_type", type=str, default='cnn')
    # parser.add_argument("-source_path", type=str, default='dataset/cnn_test.parquet')
    # parser.add_argument("-result_path", type=str, default='outputs/cnn')
    # parser.add_argument("-analysis_path", type=str, default='analysis/cnn')
    parser.add_argument("-run_type", type=str, default='single')
    
    paras = parser.parse_args()
    device = paras.device
    llms = ast.literal_eval(paras.llms)
    data_type = paras.data_type
    source_path = str((Path(__file__).parent / f'dataset/{data_type}_test.parquet').resolve())
    result_path = str((Path(__file__).parent / f'outputs/{data_type}').resolve())
    analysis_path = str((Path(__file__).parent / f'analysis/{data_type}').resolve())
    run_type = paras.run_type

    # result_path = 'outputs/cnn/'
    # source_path = 'dataset/cnn_test.parquet'
    # analysis_path = 'analysis/cnn'
    
    print(f" device: {device}\n llms: {llms}\n source_path: {source_path}\n result_path: {result_path}\n analysis_path: {analysis_path}\n run_type: {run_type}\n")
    
    os.makedirs(analysis_path, exist_ok=True)

    for llm in llms: 
        print(f"========={llm}=====")

        # Load original dataset
        source_data = pq.ParquetDataset(source_path).read().to_pandas()

        out_f = open(f'{analysis_path}/{llm}-{run_type}.csv', 'w', newline='', encoding='utf-8')
        csv.writer(out_f).writerow(columns)

        # Load generation results and drop rows where generated or reference text is empty
        res_data = pd.read_json(f'{result_path}/{llm}-{run_type}.jsonl', lines=True)
        for i in range(len(res_data)):
            if len(res_data.unwatermarked_text[i].strip()) == 0 or \
                len(source_data.highlights[i].strip()) == 0:
                res_data = res_data.drop(index=i)
                source_data = source_data.drop(index=i)
                
        res_data = res_data.reset_index(drop=True)
        source_data = source_data.reset_index(drop=True)

        # Evaluate generation quality using BERTScore
        print("Computing bert score for unwatermarked texts ...")
        bert_score_un = score(list(res_data.unwatermarked_text), list(source_data.highlights[:len(res_data)]),
                             lang="en", verbose=False,
                             use_fast_tokenizer=True, device='cuda')
        
        print("Computing bert score for watermarked texts ...")
        bert_score_wa = score(list(res_data.watermarked_text), list(source_data.highlights[:len(res_data)]),
                             lang="en", verbose=False,
                             use_fast_tokenizer=True, device='cuda')

        kgw = KGWHierarchicalSummarizer('models/' + llm, overlap_sents=2, lead_k_ratio=0)
        
        for i in tqdm.tqdm(range(len(res_data))):
            if res_data.id[i] == source_data.id[i]:
                res = []

                # Evaluate watermark detection scores
                score_watermarked = kgw.detect_watermark(res_data.watermarked_text[i])['score']
                score_unwatermarked = kgw.detect_watermark(res_data.unwatermarked_text[i])['score']
                score_highlight = kgw.detect_watermark(source_data.highlights[i])['score']
                res.append(score_watermarked)
                res.append(score_unwatermarked)
                res.append(score_highlight)
                # print('%.6f, %.6f, %.6f' % (score_watermarked, score_unwatermarked, score_highlight))

                # Evaluate text generation quality using ROUGE
                score_method = ['rouge1', 'rouge2', 'rougeL']
                for m in score_method:
                    scorer = rouge_scorer.RougeScorer([m], use_stemmer=True)
                    for x in ['watermarked_text', 'unwatermarked_text']:
                        scores = scorer.score(res_data[x][i], source_data.highlights[i])
                        for k in scores[m]:
                            res.append(k)
                        # print(m, score)
                
                # Append BERTScore for watermarked text
                for k in range(3):
                    res.append(bert_score_wa[k][i].item())
                # Append BERTScore for unwatermarked text
                for k in range(3):
                    res.append(bert_score_un[k][i].item())
                
                csv.writer(out_f).writerow(res)

            # if i > 10:
            #     break
        out_f.close()


# Example:
#   $env:HF_ENDPOINT="https://hf-mirror.com"; $env:CUDA_VISIBLE_DEVICES='0'
#   & python result_analysis.py -llms "'t5-base', 'flan-t5-small'" -data_type 'cnn'
