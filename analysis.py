import pandas as pd
import os


if __name__ == "__main__":
    analysis_path = 'analysis/'
    output_path = 'analysis/results.xlsx'
    
    pd.DataFrame().to_excel(output_path, index=False)
    excelwriter = pd.ExcelWriter(output_path, engine='openpyxl',mode='a', if_sheet_exists='replace')

    for run_type in ['batch', 'single']:
        if run_type == 'batch':
            llms = ['t5-small', 't5-base', 'flan-t5-base', 'flan-t5-small',\
                    'bart-large-cnn', 'bart-large-xsum']
            data_type = ['cnn', 'xsum']
        else:
            llms = ['t5-base', 'flan-t5-base']
            data_type = ['cnn', 'cnn2000']
        
        for data in data_type:
            res = pd.DataFrame()

            for llm in llms: 
                print(f"========={llm}=====")

                # loading raw data
                if run_type == 'single' and data == 'cnn2000':
                    eval_data = pd.read_csv(f'{analysis_path}cnn/{llm}-batch.csv')
                    eval_data = eval_data[:2000]
                else:
                    eval_data = pd.read_csv(f'{analysis_path}{data}/{llm}-{run_type}.csv')
                
                # Calculate mean and standard deviation
                res_tmp = pd.concat([eval_data.mean(), eval_data.std()], axis=1)
                res_tmp = round(eval_data.mean(), 3).astype(str)+'('+round(eval_data.std(), 3).astype(str)+')'
                
                for x in ['wa', 'un']:
                    indx = [x in ind for ind in res_tmp.index]
                    data_tmp = res_tmp[indx]
                    index_new = [indx.replace('_'+ x, '') for indx in data_tmp.index]
                    data_tmp.index = index_new
                    data_tmp.name = f'{llm}-{x}'               
                    res = pd.concat([res, data_tmp], axis=1)
            res.to_excel(excelwriter, sheet_name=data+'-'+run_type)
            
    excelwriter.close()