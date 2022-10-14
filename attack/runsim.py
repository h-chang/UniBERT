import os, random
import datastat, pretrain, finetune
import pandas as pd

if __name__ == "__main__":
    # clear the tsp que
    os.system("TS_SOCKET=/tmp/gpu0 tsp -C")
    os.system("TS_SOCKET=/tmp/gpu1 tsp -C")
    os.system("TS_SOCKET=/tmp/gpu0 tsp -S 1") 
    os.system("TS_SOCKET=/tmp/gpu1 tsp -S 1")
    # shared settings
    sizes = ['base']
    vers = ['orig','svm_cay','roberta'] # 'orig','svm_cay','roberta'
    testing = False
    gpu = 0
    #######################
    ##    Pretraining    ##
    #######################
    # pretrain settings
    pretrain_repeat = 1
    if 0:
        for size in sizes:
            for i in range(pretrain_repeat):
                seed = i
                for ver in vers:
                    ver_pretrain = f"{ver}_{i}"
                    command = pretrain.pretrain(size, ver_pretrain, seed, testing, i)
                    os.environ["CUDA_VISIBLE_DEVICES"]=f'{gpu}'
                    os.system(f"TS_SOCKET=/tmp/gpu{gpu} tsp {command}")
                    # gpu = (gpu+1)%2

    #############################
    ##    Finetune & Attack    ##
    #############################
    # finetune settings
    finetune_repeat = 4
    datasets = ['ag_news','snli','yelp_polarity',] #'imdb','rotten_tomatoes','multi_nli'
    recipes = ['textfooler','textbugger','pwws'] # ['alzantot','bae','bert-attack','checklist','clare','deepwordbug','faster-alzantot','iga','input-reduction','kuleshov','pruthi','pso','pwws','textbugger','textfooler']
    if 1:
        df = datastat.get_stats(datasets)
        for size in sizes:
            for i in range(finetune_repeat):
                if i == 0:
                    continue
                seed = i
                for ver in vers:
                    ver_pretrain = f"{ver}_0"
                    ver_finetune = f"{ver}_{i}"
                    tsp = True
                    gpu = finetune.fintune_and_attack(size,df,ver_pretrain,ver_finetune,seed,gpu,testing,recipes,tsp)

                    

