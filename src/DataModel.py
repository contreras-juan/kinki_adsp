import os
import pandas as pd
from glob import glob

class DataModel:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def create_merge_folders(self):
        root_train = "/".join(self.train_path.split("/")[:-2])
        self.path_train_merged = root_train + "/merged"
        
        root_test = "/".join(self.test_path.split("/")[:-2])
        self.path_test_merged = root_test + "/merged"
        
        os.makedirs(self.path_train_merged, exist_ok=True)
        os.makedirs(self.path_test_merged, exist_ok=True)

    def get_data_files(self, format_type):
        self.train_files = glob(self.train_path + f"*.{format_type}")
        self.test_files = glob(self.test_path + f"*.{format_type}")
    
    def merge_train_test(self, on="ID", how="outer"):
        self.df_train = pd.DataFrame([], columns=[on])
        self.df_test  = pd.DataFrame([], columns=[on])
        
        for train_file in self.train_files:
            temp_df = pd.read_csv(train_file)
            self.df_train = pd.merge(self.df_train, temp_df, on=on, how=how)

        for test_file in self.test_files:
            temp_df = pd.read_csv(test_file)
            self.df_test = pd.merge(self.df_test, temp_df, on=on, how=how)

    def save_merged_data(self, name_train, name_test):
        self.df_train.to_csv(self.path_train_merged + f"/{name_train}", sep=";", index=None)
        self.df_test.to_csv(self.path_test_merged + f"/{name_test}", sep=";", index=None)

    def merge_data(self):

        ## PARAMS ##
        format_type="csv"
        on="ID"
        how="outer"
        name_train="train_merged.csv"
        name_test="test_merged.csv"

        ### PIPELINE ###
        self.create_merge_folders()
        self.get_data_files(format_type=format_type)
        self.merge_train_test(on=on, how=how)
        self.save_merged_data(name_train=name_train, name_test=name_test)