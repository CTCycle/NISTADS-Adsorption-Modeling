import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, configuration, dataframe: pd.DataFrame):
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'     

        # Set the sizes for the train and validation datasets        
        self.validation_size = configuration["dataset"]["VALIDATION_SIZE"]
        self.seed = configuration["dataset"]["SPLIT_SEED"]
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe        
        
        total_samples = len(dataframe)
        self.train_size = int(total_samples * self.train_size)
        self.val_size = int(total_samples * self.validation_size)
            
    #--------------------------------------------------------------------------
    def split_train_and_validation(self):       
        self.dataframe = shuffle(
            self.dataframe, random_state=self.seed).reset_index(drop=True) 
        train_data = self.dataframe.iloc[:self.train_size]
        validation_data = self.dataframe.iloc[self.train_size:self.train_size + self.val_size]
        
        return train_data, validation_data
    
    
    
   
