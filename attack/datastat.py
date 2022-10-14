import os, re
import pandas as pd
def find(query,text):
    return re.search(r"total:\s*(.*)\s*",text).group(1)[5:-1]

def parse(logfile):
    with open(logfile) as f:
        lines = f.readlines()
        text = ''.join(lines)
    values = []
    for line in lines: 
        match = re.search(r"\[94m(.*)\[0m",line)
        if match is not None:
            values += [match.group(1)]
    # get the last few rows of the text file
    labels = text.split('\n\n')[-1].replace('(','').replace(')','')
    # sort the labels
    labels = labels.split('\n')
    labels.sort()
    labels = ' '.join(labels)
    # remove white space characters
    labels = ' '.join(labels.split())
    # seperate lable and their frequencies
    label = labels.split()[0::2]
    count = labels.split()[1::2]
    return values[1:]+[label,count]

def get_stats(datasets):
    data = []
    for name in datasets:
        # print(f"====== {name} ======")
        logfile = f'/home/h/dev/pathfind/attack/help/dataset/{name}.txt'
        command = f'echo ""> {logfile}; textattack peek-dataset --dataset-from-huggingface {name} >> {logfile} 2>&1'
        # os.system(command)
        data += [[name]+parse(logfile)]
    df = pd.DataFrame(data,columns=['name','sample size','total words','mean','std','min','max','lowercased','label','count'])
    return df

if __name__ == '__main__':
    datasets = ['ag_news','rotten_tomatoes','imdb','yelp_polarity','snli','multi_nli']
    df = get_stats(datasets)
    df.to_excel('/home/h/dev/pathfind/attack/help/dataset/summary.xlsx')
    print(df)