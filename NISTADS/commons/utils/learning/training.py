import torch
import keras

from NISTADS.commons.utils.learning.callbacks import callbacks_handler
from NISTADS.commons.utils.data.serializer import ModelSerializer
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self, configuration, metadata):        
        keras.utils.set_random_seed(configuration["SEED"])        
        self.selected_device = CONFIG["device"]["DEVICE"]
        self.device_id = CONFIG["device"]["DEVICE_ID"]
        self.mixed_precision = configuration["device"]["MIXED_PRECISION"] 
        self.serializer = ModelSerializer()
        self.configuration = configuration
        self.metadata = metadata          

    # set device
    #--------------------------------------------------------------------------
    def set_device(self):
        if self.selected_device == 'GPU':
            if not torch.cuda.is_available():
                logger.info('No GPU found. Falling back to CPU')
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(f'cuda:{self.device_id}')
                torch.cuda.set_device(self.device)  
                logger.info('GPU is set as active device')            
                if self.mixed_precision:
                    keras.mixed_precision.set_global_policy("mixed_float16")
                    logger.info('Mixed precision policy is active during training')                   
        else:
            self.device = torch.device('cpu')
            logger.info('CPU is set as active device')         

    #--------------------------------------------------------------------------
    def train_model(self, model : keras.Model, train_data, validation_data, 
                    checkpoint_path, from_checkpoint=False):       

        # perform different initialization duties based on state of session:
        # training from scratch vs resumed training
        # calculate number of epochs taking into account possible training resumption
        if not from_checkpoint:            
            epochs = self.configuration["training"]["EPOCHS"] 
            from_epoch = 0
            history = None
        else:
            _, self.metadata, history = self.serializer.load_session_configuration(checkpoint_path)                     
            epochs = history['total_epochs'] + CONFIG["training"]["ADDITIONAL_EPOCHS"] 
            from_epoch = history['total_epochs']           
       
        # add all callbacks to the callback list
        RTH_callback, callbacks_list = callbacks_handler(
            self.configuration, checkpoint_path, history)       
        
        # run model fit using keras API method.  
        training = model.fit(train_data, epochs=epochs, validation_data=validation_data, 
                             callbacks=callbacks_list, initial_epoch=from_epoch)
        
        # save model parameters in json files
        history = {'history' : RTH_callback.history, 
                   'val_history' : RTH_callback.val_history,
                   'total_epochs' : epochs}
        
        # save pretrained model as serialized keras model 
        # save metadata and training history in json files, including preprocessing metadata    
        self.serializer.save_pretrained_model(model, checkpoint_path)       
        self.serializer.save_session_configuration(
            checkpoint_path, history, self.configuration, self.metadata)

        


      