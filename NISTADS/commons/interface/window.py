from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from functools import partial
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)

from NISTADS.commons.utils.data.database import AdsorptionDatabase
from NISTADS.commons.configuration import Configuration
from NISTADS.commons.interface.events import GraphicsHandler, DatasetEvents, ValidationEvents, ModelEvents
from NISTADS.commons.interface.workers import Worker
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


###############################################################################
class MainWindow:
    
    def __init__(self, ui_file_path: str): 
        super().__init__()           
        loader = QUiLoader()
        ui_file = QFile(ui_file_path)
        ui_file.open(QIODevice.ReadOnly)
        self.main_win = loader.load(ui_file)
        ui_file.close()           

        # Checkpoint & metrics state
        self.selected_checkpoint = None
        self.selected_metrics = {'dataset': [], 'model': []}       
          
        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        # set thread pool for the workers        
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None
        self.worker_running = False  

        # initialize database
        self.database = AdsorptionDatabase(self.configuration)
        self.database.initialize_database()  
        self.database.update_database()                

        # --- Create persistent handlers ---
        self.graphic_handler = GraphicsHandler()
        self.dataset_handler = DatasetEvents(self.database, self.configuration)
        self.validation_handler = ValidationEvents(self.database, self.configuration)
        self.model_handler = ModelEvents(self.database, self.configuration)        

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([ 
            (QPushButton,'stopThread','stop_thread'),  
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QProgressBar,'progressBar','progress_bar'),         
            # 1. dataset tab page
            (QCheckBox,'adsIsothermCluster','experiments_clustering'),            
            (QPushButton,'evaluateDataset','evaluate_dataset'),

            (QSpinBox,'seed','general_seed'),
            (QDoubleSpinBox,'sampleSize','sample_size'), 
            (QSpinBox,'maxPoints','max_measurements'), 
            (QSpinBox,'minPoints','min_measurements'),
            (QSpinBox,'smileSeqSize','SMILE_sequence_size'),
            (QSpinBox,'maxPressure','max_pressure'),
            (QSpinBox,'maxUptake','max_uptake'), 
            (QPushButton,'buildMLDataset','build_ML_dataset'),                    

            (QDoubleSpinBox,'guestFraction','guest_fraction'),
            (QDoubleSpinBox,'hostFraction','host_fraction'),
            (QDoubleSpinBox,'expFraction','experiments_fraction'),
            (QSpinBox,'parallelTasks','parallel_tasks'),  
            (QPushButton,'collectAdsData','collect_adsorption_data'),  
            (QPushButton,'retrieveChemProperties','retrieve_properties'),        
                      
            # 2. training tab page                
            (QCheckBox,'setShuffle','use_shuffle'),
            (QDoubleSpinBox,'trainSampleSize','train_sample_size'),            
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QSpinBox,'shuffleSize','shuffle_size'),
            (QRadioButton,'setCPU','use_CPU'),
            (QRadioButton,'setGPU','use_GPU'),
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'numWorkers','num_workers'),
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','get_real_time_history'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QSpinBox,'trainSeed','train_seed'),
            (QSpinBox,'splitSeed','split_seed'),
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),            
            (QSpinBox,'saveCPFrequency','save_cp_frequency'),
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QDoubleSpinBox,'initialLearningRate','initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','target_LR'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'), 
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'),         
            (QSpinBox,'initialNeurons','initial_neurons'),
            (QDoubleSpinBox,'dropoutRate','dropout_rate'),                    
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),
            (QComboBox,'checkpointsList','checkpoints_list'),            
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            # 3. model evaluation tab page
            (QPushButton,'evaluateModel','model_evaluation'),
            (QCheckBox,'runEvaluationGPU','use_GPU_evaluation'), 
            (QPushButton,'checkpointSummary','checkpoints_summary'),
            (QCheckBox,'evalReport','get_evaluation_report'), 
            (QCheckBox,'adsIsothermsComparison','image_reconstruction'),      
            (QSpinBox,'numImages','num_evaluation_images'),           
            # 4. inference tab page  
            (QCheckBox,'runInferenceGPU','use_GPU_inference'),      
            (QPushButton,'encodeImages','encode_images'),          
            # 5. Viewer tab
            (QPushButton,'loadImages','load_source_images'),
            (QPushButton,'previousImg','previous_image'),
            (QPushButton,'nextImg','next_image'),
            (QPushButton,'clearImg','clear_images'),
            (QRadioButton,'viewDataPlots','data_plots_view'),
            (QRadioButton,'viewEvalPlots','model_plots_view')                       
            ])
        
        self._connect_signals([  
            ('checkpoints_list','currentTextChanged',self.select_checkpoint), 
            ('refresh_checkpoints','clicked',self.load_checkpoints),
            ('stop_thread','clicked',self.stop_running_worker),          
            # 1. dataset tab page            
            ('experiments_clustering','toggled',self._update_metrics),
            ('evaluate_dataset','clicked',self.run_dataset_evaluation_pipeline),  
            ('collect_adsorption_data','clicked',self.collect_data_from_NIST),  
            ('retrieve_properties','clicked',self.retrieve_properties_from_PUBCHEM),           
            ('build_ML_dataset','clicked',self.build_ML_dataset),
            # 2. training tab page               
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # 3. model evaluation tab page            
            ('get_evaluation_report','toggled',self._update_metrics), 
            ('prediction_quality','toggled',self._update_metrics),
            ('model_evaluation','clicked', self.run_model_evaluation_pipeline),
            ('checkpoints_summary','clicked',self.get_checkpoints_summary),                  
           
            # 4. inference tab page  
            ('encode_images','clicked',self.encode_images_with_checkpoint),            
            # 5. viewer tab page 
            ('data_plots_view', 'toggled', self._update_graphics_view),
            ('model_plots_view', 'toggled', self._update_graphics_view),           
            ('previous_image', 'clicked', self.show_previous_figure),
            ('next_image', 'clicked', self.show_next_figure),
            ('clear_images', 'clicked', self.clear_figures),            
        ]) 
        
        self._auto_connect_settings() 
        self.use_GPU.toggled.connect(self._update_device)
        self.use_CPU.toggled.connect(self._update_device)
        
        # Initial population of dynamic UI elements
        self.load_checkpoints()
        self._set_graphics() 


    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()           

    # [HELPERS]
    ###########################################################################
    def connect_update_setting(self, widget, signal_name, config_key, getter=None):
        if getter is None:
            if isinstance(widget, (QCheckBox, QRadioButton)):
                getter = widget.isChecked
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                getter = widget.value
            elif isinstance(widget, QComboBox):
                getter = widget.currentText
           
        signal = getattr(widget, signal_name)
        signal.connect(partial(self._update_single_setting, config_key, getter))

    #--------------------------------------------------------------------------
    def _update_single_setting(self, config_key, getter, *args):
        value = getter()
        self.config_manager.update_value(config_key, value)

    #--------------------------------------------------------------------------
    def _auto_connect_settings(self):
        connections = [
            # 1. dataset tab page
            ('general_seed', 'valueChanged', 'general_seed'),
            ('sample_size', 'valueChanged', 'sample_size'),
            ('min_measurements', 'valueChanged', 'min_measurements'),
            ('max_measurements', 'valueChanged', 'max_measurements'),
            ('SMILE_sequence_size', 'valueChanged', 'SMILE_sequence_size'),
            ('guest_fraction', 'valueChanged', 'guest_fraction'),
            ('host_fraction', 'valueChanged', 'host_fraction'),
            ('experiments_fraction', 'valueChanged', 'experiments_fraction'),

            # 2. training tab page          
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('num_workers', 'valueChanged', 'num_workers'),
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
            ('use_JIT_compiler', 'toggled', 'use_jit_compiler'),
            ('jit_backend', 'currentTextChanged', 'jit_backend'),
            ('use_tensorboard', 'toggled', 'run_tensorboard'),
            ('get_real_time_history', 'toggled', 'real_time_history'),
            ('save_checkpoints', 'toggled', 'save_checkpoints'),
            ('LR_scheduler', 'toggled', 'use_lr_scheduler'),
            ('split_seed', 'valueChanged', 'split_seed'),
            ('train_seed', 'valueChanged', 'train_seed'),
            ('shuffle_size', 'valueChanged', 'shuffle_size'),
            ('epochs', 'valueChanged', 'epochs'),
            ('additional_epochs', 'valueChanged', 'additional_epochs'),
            ('initial_neurons', 'valueChanged', 'initial_neurons'),      
            ('dropout_rate', 'valueChanged', 'dropout_rate'),            
            ('batch_size', 'valueChanged', 'batch_size'),
            ('device_ID', 'valueChanged', 'device_id'),
            # 3. model evaluation tab page
            ('num_evaluation_images', 'valueChanged', 'num_evaluation_images'),
            # 4. inference tab page           
            ('validation_size', 'valueChanged', 'validation_size')]

        self.data_metrics = [
            ('experiments_clustering', self.experiments_clustering)]
        self.model_metrics = [
            ('evaluation_report', self.get_evaluation_report),
            ('adsorption_isotherms_prediction', self.prediction_quality)]                

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

    #--------------------------------------------------------------------------
    def _update_device(self):
        device = 'GPU' if self.use_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', device)  

    #--------------------------------------------------------------------------
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0)

    #--------------------------------------------------------------------------
    def get_current_pixmaps_and_key(self):
        for radio, (pixmap_key, idx_key) in self.pixmap_source_map.items():
            if radio.isChecked():
                return self.pixmaps[pixmap_key], idx_key
        return [], None 

    #--------------------------------------------------------------------------
    def _set_graphics(self):
        self.graphics = {}        
        view = self.main_win.findChild(QGraphicsView, 'canvas')
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        view.setRenderHint(QPainter.Antialiasing, True)
        view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        view.setRenderHint(QPainter.TextAntialiasing, True)
        self.graphics = {'view': view,
                         'scene': scene,
                         'pixmap_item': pixmap_item}
        
        # Image data                
        self.pixmaps = {          
            'dataset_eval_images': [],  
            'model_eval_images': []}        

        # Canvas state        
        self.current_fig = {'dataset_eval_images' : 0, 'model_eval_images' : 0}

        self.pixmap_source_map = {
            self.data_plots_view: ("dataset_eval_images", "dataset_eval_images"),
            self.model_plots_view: ("model_eval_images", "model_eval_images")}           

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _start_worker(self, worker : Worker, on_finished, on_error, on_interrupted,
                      update_progress=True):
        if update_progress:       
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)        
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)
        self.worker_running = True

    #--------------------------------------------------------------------------
    def _send_message(self, message): 
        self.main_win.statusBar().showMessage(message)    

    # [SETUP]
    ###########################################################################
    def _setup_configuration(self, widget_defs):
        for cls, name, attr in widget_defs:
            w = self.main_win.findChild(cls, name)
            setattr(self, attr, w)
            self.widgets[attr] = w

    #--------------------------------------------------------------------------
    def _connect_signals(self, connections):
        for attr, signal, slot in connections:
            widget = self.widgets[attr]
            getattr(widget, signal).connect(slot)   
   
    # [SLOT]
    ###########################################################################
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    #--------------------------------------------------------------------------
    Slot()
    def stop_running_worker(self):
        if self.worker is not None:
            self.worker.stop()       
        self._send_message("Interrupt requested. Waiting for threads to stop...")

    #--------------------------------------------------------------------------
    @Slot()
    def load_checkpoints(self):       
        checkpoints = self.model_handler.get_available_checkpoints()
        self.checkpoints_list.clear()
        if checkpoints:
            self.checkpoints_list.addItems(checkpoints)
            self.selected_checkpoint = checkpoints[0]
            self.checkpoints_list.setCurrentText(checkpoints[0])
        else:
            self.selected_checkpoint = None
            logger.warning("No checkpoints available")

    #--------------------------------------------------------------------------
    @Slot(str)
    def select_checkpoint(self, name: str):
        self.selected_checkpoint = name if name else None 

    #--------------------------------------------------------------------------
    @Slot()
    def _update_metrics(self):             
        self.selected_metrics['dataset'] = [
            name for name, box in self.data_metrics if box.isChecked()]
        self.selected_metrics['model'] = [
            name for name, box in self.model_metrics if box.isChecked()]
        
    #--------------------------------------------------------------------------
    # [GRAPHICS]
    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self):  
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            self.graphics['pixmap_item'].setPixmap(QPixmap())
            self.graphics['scene'].setSceneRect(0, 0, 0, 0)
            return

        idx = self.current_fig.get(idx_key, 0)
        idx = min(idx, len(pixmaps) - 1)
        raw = pixmaps[idx]
        
        qpixmap = QPixmap(raw) if isinstance(raw, str) else raw
        view = self.graphics['view']
        pixmap_item = self.graphics['pixmap_item']
        scene = self.graphics['scene']
        view_size = view.viewport().size()
        scaled = qpixmap.scaled(
            view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())     

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_previous_figure(self):             
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] > 0:
            self.current_fig[idx_key] -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self):
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] < len(pixmaps) - 1:
            self.current_fig[idx_key] += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self):
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        self.pixmaps[idx_key].clear()
        self.current_fig[idx_key] = 0
        self._update_graphics_view()
        self.graphics['pixmap_item'].setPixmap(QPixmap())
        self.graphics['scene'].setSceneRect(0, 0, 0, 0)
        self.graphics['view'].viewport().update()

    #--------------------------------------------------------------------------
    # [DATASET TAB]
    #--------------------------------------------------------------------------        
    @Slot()
    def collect_data_from_NIST(self):        
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.dataset_handler = DatasetEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.dataset_handler.run_data_collection_pipeline)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_data_success,
            on_error=self.on_data_error,
            on_interrupted=self.on_task_interrupted)  
        
    #--------------------------------------------------------------------------        
    @Slot()
    def retrieve_properties_from_PUBCHEM(self): 
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.dataset_handler = DatasetEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.dataset_handler.run_chemical_properties_pipeline)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_data_success,
            on_error=self.on_data_error,
            on_interrupted=self.on_task_interrupted)  
        
    #--------------------------------------------------------------------------        
    @Slot()
    def run_dataset_evaluation_pipeline(self):  
        if not self.data_metrics:
            return 
        
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics['dataset'])   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_evaluation_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def build_ML_dataset(self):          
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.dataset_handler = DatasetEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.dataset_handler.run_dataset_builder)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_processing_finished,
            on_error=self.on_data_error,
            on_interrupted=self.on_task_interrupted)       

    #--------------------------------------------------------------------------
    # [TRAINING TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def train_from_scratch(self):
        if self.worker_running:            
            return 
                  
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder model from scratch...")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.model_handler.run_training_pipeline)                            
       
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self): 
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}")         
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.model_handler.resume_training_pipeline,
            self.selected_checkpoint)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)

    #--------------------------------------------------------------------------
    # [MODEL EVALUATION TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def run_model_evaluation_pipeline(self):  
        if self.worker_running:            
            return 

        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)    
        device = 'GPU' if self.use_GPU_evaluation.isChecked() else 'CPU'   
        # send message to status bar
        self._send_message(f"Evaluating {self.select_checkpoint} performances... ")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_model_evaluation_pipeline,
            self.selected_metrics['model'], self.selected_checkpoint, device)                
        
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)     

    #-------------------------------------------------------------------------- 
    @Slot()
    def get_checkpoints_summary(self):       
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)           
        # send message to status bar
        self._send_message("Generating checkpoints summary...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.validation_handler.get_checkpoints_summary) 

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    # [INFERENCE TAB]
    #--------------------------------------------------------------------------   
    @Slot()    
    def encode_images_with_checkpoint(self):  
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)  
        device = 'GPU' if self.use_GPU_inference.isChecked() else 'CPU'
        # send message to status bar
        self._send_message(f"Encoding images with {self.selected_checkpoint}") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.model_handler.run_inference_pipeline,
            self.selected_checkpoint,
            device)

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_inference_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)


    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################     
    def on_data_success(self, session):          
        self.dataset_handler.handle_success(
            self.main_win, 'Data has been collected from NIST-A database')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_dataset_processing_finished(self, session):         
        self.dataset_handler.handle_success(self.main_win, 'Dataset has been built successfully')
        self.worker_running = False

    #--------------------------------------------------------------------------      
    def on_dataset_evaluation_finished(self, plots):   
        key = 'dataset_eval_images'      
        if plots:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p) 
                 for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(self.main_win, 'Figures have been generated')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self.model_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots):  
        key = 'model_eval_images'         
        if plots is not None:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p)
                for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(
            self.main_win, f'Model {self.selected_checkpoint} has been evaluated')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):          
        self.model_handler.handle_success(
            self.main_win, 'Inference call has been terminated')
        self.worker_running = False


    ###########################################################################   
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################     
    @Slot() 
    def on_data_error(self, err_tb):
        self.dataset_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False  
   
    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_evaluation_error(self, err_tb):
        self.validation_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False   

    #--------------------------------------------------------------------------
    @Slot() 
    def on_model_error(self, err_tb):
        self.model_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False  

    #--------------------------------------------------------------------------
    def on_task_interrupted(self):         
        self.progress_bar.setValue(0)
        self._send_message('Current task has been interrupted by user') 
        logger.warning('Current task has been interrupted by user')
        self.worker_running = False        
        
          
         


        

    
       

    
