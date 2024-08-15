import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger
      

# [CONVERSION OF PRESSURE]
###############################################################################
class PressureConversion:

    def __init__(self):        
        self.P_COL = 'pressure'
        self.P_UNIT_COL = 'pressureUnits'
        self.P_TARGET_COL = 'pressure_in_Pascal'                                        
       
    #--------------------------------------------------------------------------
    def bar_to_Pascal(self, p_vals):            
                    
        return [int(p_val * 100000) for p_val in p_vals]  
    
    #--------------------------------------------------------------------------
    def convert_based_on_units(self, row):

        if row[self.P_UNIT_COL] == 'bar':
            return self.bar_to_Pascal(row[self.P_COL])

    #--------------------------------------------------------------------------
    def convert_data(self, dataframe : pd.DataFrame):

        dataframe[self.P_TARGET_COL] = dataframe.apply(lambda x : self.convert_based_on_units(x), axis=1) 
        dataframe.drop(columns=[self.P_COL, self.P_UNIT_COL], inplace=True)    

        return dataframe  
    

# [CONVERSION OF UPTAKE]
###############################################################################
class UptakeConversion:  

    def __init__(self):        
        self.Q_COL = 'adsorbed_amount'
        self.Q_UNIT_COL = 'adsorptionUnits'
        self.Q_TARGET_COL = 'uptake_in_mol_g'  
        self.mol_W = 'molecular_weight'
        self.VALID_UNITS = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 'cm3(STP)/g',
                            'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g']
        
        logger.debug
        
    #--------------------------------------------------------------------------
    def convert_mmol_g_or_mol_kg(self, q_vals):
        return [q_val / 1000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_mmol_kg(self, q_vals):
        return [q_val / 1000000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_mg_g(self, q_vals, mol_weight):
        return [q_val / 1000 / float(mol_weight) for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_g_g(self, q_vals, mol_weight):
        return [q_val / float(mol_weight) for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_wt_percent(self, q_vals, mol_weight):
        return [(q_val / 100) / float(mol_weight) for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_g_adsorbate_per_100g_adsorbent(self, q_vals, mol_weight):
        return [(q_val / 100) / float(mol_weight) for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_ml_stp_g_or_cm3_stp_g(self, q_vals):
        return [q_val / 22.414 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_based_on_units(self, row):
             
        if row[self.Q_UNIT_COL] in ('mmol/g', 'mol/kg'):
            return self.convert_mmol_g_or_mol_kg(row[self.Q_COL])
        elif row[self.Q_UNIT_COL] == 'mmol/kg':
            return self.convert_mmol_kg(row[self.Q_COL])
        elif row[self.Q_UNIT_COL] == 'mg/g':
            return self.convert_mg_g(row[self.Q_COL], row[self.mol_W])
        elif row[self.Q_UNIT_COL] == 'g/g':
            return self.convert_g_g(row[self.Q_COL], row[self.mol_W])
        elif row[self.Q_UNIT_COL] == 'wt%':
            return self.convert_wt_percent(row[self.Q_COL], row[self.mol_W])
        elif row[self.Q_UNIT_COL] in ('g Adsorbate / 100g Adsorbent', 'g/100g'):
            return self.convert_g_adsorbate_per_100g_adsorbent(row[self.Q_COL], row[self.mol_W])
        elif row[self.Q_UNIT_COL] in ('ml(STP)/g', 'cm3(STP)/g'):
            return self.convert_ml_stp_g_or_cm3_stp_g(row[self.Q_COL])
        else:
            return row[self.Q_COL]
        
    #--------------------------------------------------------------------------
    def convert_data(self, dataframe : pd.DataFrame):

        dataframe[self.Q_TARGET_COL] = dataframe.apply(lambda x : self.convert_based_on_units(x), axis=1)
        dataframe.drop(columns=[self.Q_COL, self.Q_UNIT_COL], inplace=True)       

        return dataframe 
        
    
    

   
  


    

        
    
    

    
 

    