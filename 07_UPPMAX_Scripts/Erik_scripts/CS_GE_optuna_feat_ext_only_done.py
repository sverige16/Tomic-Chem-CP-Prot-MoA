import optuna
import datetime
import time

# Torch
import torch
import torch.nn as nn
import neptune.new as neptune
import sys

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL,  
    choose_device,
    different_loss_functions, 
    apply_class_weights_GE, 
    adapt_training_loop, 
    smiles_to_array,
    GE_driving_code,
    extracting_pretrained_single_models,
    set_parameter_requires_grad,
    returning_smile_string,
    dubbelcheck_dataset_length,
  
)

from Helper_Models import (
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
        
                           Chemical_Structure_Model,
                           Modified_GE_Model)
import torch
import torch.nn as nn

class CS_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, tprofiles_df, split_sets, dict_moa, dict_cell_line, checking_mechanism):
        self.compound_df = compound_df
        self.tprofiles_df = tprofiles_df
        self.split_sets = split_sets
        self.dict_moa = dict_moa
        self.dict_cell_line = dict_cell_line
        self.check = checking_mechanism
    def __len__(self):
        check_criteria = self.check
        assert len(self.tprofiles_df) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.tprofiles_df)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        t_profile = self.tprofiles_df.iloc[idx, :-1]          # extract image from csv using index
        t_sig_id = self.tprofiles_df.iloc[idx, -1]
        CID = self.split_sets["Compound ID"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        cell_line_key = self.split_sets["cell_iname"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        CID = CID.iloc[0]
        cell_line_key = cell_line_key.iloc[0]
        t_cell_line = torch.tensor(self.dict_cell_line[cell_line_key])
        smile_string = returning_smile_string(self.compound_df,CID)
        compound_array = smiles_to_array(smile_string)
        if compound_array.shape[0] != 2048:
            raise ValueError("Compound array is not the correct size")
        assert not torch.isnan(compound_array).any(), "NaN value found in compound array"
        label_tensor = torch.from_numpy(self.dict_moa[moa_key])                  # convert label to number
        t_profile_features = torch.tensor(t_profile) 
        return compound_array.float(), t_profile_features, t_cell_line.float(), label_tensor.float() # returns 
        
class CS_GE_Model(nn.Module):
    def __init__(self, modelCS, modelGE, init_nodes):
        super(CS_GE_Model, self).__init__()
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.modelGE = modelGE
        self.linear_layer_pre = nn.Linear(int(103 + 20), init_nodes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm = nn.BatchNorm1d(init_nodes)

    def forward(self, x1in, x2in, x3in):
        if x1in.shape[1] != 2048:
            raise ValueError("The input shape for the compound is not correct")
        x1 = self.modelCS(x1in)
        x2 = self.modelGE(x2in, x3in)
        x  = torch.cat((x1, x2), dim = 1)
        x = self.leaky_relu(self.linear_layer_pre(x))
        return x


# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
incl_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
model_name = 'CS_GE'
print("Begin Training")

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 0.0000195589 
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2,
         'shuffle' : True} 
          


file_name = "erik10_hq_8_12"
fold_num = 0
train_np, valid_np, test_np, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training, L1000_validation, L1000_test = GE_driving_code(file_name, fold_num)

using_cuda = True 
device = choose_device(using_cuda)         

# -----------------------------------------Prepping Ensemble Model ---------------------#

# Create datasets with relevant data and labels
training_dataset_CSGE = CS_GE_Dataset(training_set_cmpds, train_np, L1000_training, dict_moa, dict_cell_lines, ["train" , "GE", fold_num])
valid_dataset_CSGE = CS_GE_Dataset(validation_set_cmpds, valid_np, L1000_validation, dict_moa, dict_cell_lines, ["valid" , "GE", fold_num])
test_dataset_CSGE = CS_GE_Dataset(test_set_cmpds, test_np, L1000_test, dict_moa, dict_cell_lines, ["test" , "GE", fold_num])

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSGE, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSGE, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSGE, **params)


def optuna_combinations_fe_ft(trial, 
                            num_classes, 
                            training_generator,
                            validation_generator, 
                           
                            model_name, 
                            device, 
                            driver, 
                            fe_ft,
                            dict_moa,
                            df_train_labels = None,
                            train_np = None,
                            L1000_training = None):
    
    
    class Extended_Model(nn.Module):
            def __init__(self, trial, pretrained_model, num_classes, n_layers):
                super(Extended_Model, self).__init__()
                self.base_model = pretrained_model
                in_features = pretrained_model.linear_layer_pre.out_features
                self.num_classes = num_classes
                if n_layers > 0:
                    layers = []
                    for i in range(n_layers):
                        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 250)
                        layers.append(nn.Linear(in_features, out_features))
                        layers.append(nn.LeakyReLU())
                        layers.append(nn.BatchNorm1d(out_features))
                        p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.8)
                        layers.append(nn.Dropout(p))
                        in_features = out_features
                    layers.append(nn.Linear(in_features, num_classes))
                else:
                    layers = [nn.Linear(in_features, num_classes)]
                # Additional layers for feature extraction
                self.additional_layers = nn.Sequential(*layers)
                print(self.additional_layers)

            def forward(self, x, y, z):
                x = self.base_model(x, y,z)
                x = self.additional_layers(x)
                return x
        
    print("Loading Pretrained Models...")
    cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
    cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)
    modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
    modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
    modelCS = Chemical_Structure_Model(num_features = 2048, num_targets = 10)
    modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
    modified_GE_model = Modified_GE_Model(modelGE)
    init_nodes = trial.suggest_int('init_nodes', 4, 250)
    pretrained_model = CS_GE_Model(modelCS, modified_GE_model, init_nodes)
    n_layers = trial.suggest_int('n_layers', 0, 3)
    model = Extended_Model(trial, pretrained_model, num_classes, n_layers = n_layers) 


    set_parameter_requires_grad(model, feature_extracting = True, added_layers = 1)

    # decide hyperparameters for trial
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 2, 10), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 2, 10))


    yn_class_weights = trial.suggest_categorical('yn_class_weights', [True, False])
    if yn_class_weights:     # if we want to apply class weights
        if driver == 'CP': # we want CP first if possible, since that is our driver
            class_weights = apply_class_weights_CL(df_train_labels, dict_moa, device)

        elif driver == 'GE':
            class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
        else:
            ValueError('Driver must be CP or GE')
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    fe_ft = 'fe'
    if fe_ft == 'fe':
        patience = 0
        max_epochs = 15
        val_str = 'f1'

#------------------------------   Calling functions --------------------------- #
        train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                    optimizer = optimizer,
                    model = model,
                    loss_fn = loss_fn,
                    loss_fn_train = loss_fn_train,
                    loss_fn_str = loss_fn_str,
                    train_loader=training_generator, 
                    valid_loader=validation_generator,
                    my_lr_scheduler = scheduler,
                    model_name=model_name,
                    device = device,
                    val_str = val_str,
                    early_patience = patience)
        return max(val_f1_score_per_epoch)

fe_ft = "fe"

storage = 'sqlite:///' + model_name + '_' + fe_ft +'.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: optuna_combinations_fe_ft(trial,
                            num_classes = num_classes, 
                            training_generator = training_generator,
                            validation_generator = validation_generator,
                        
                            model_name = model_name, 
                            device = device, 
                            driver = "GE", 
                            fe_ft = "fe",
                            dict_moa = dict_moa,
                            df_train_labels = None,
                            train_np = train_np,
                            L1000_training = L1000_training),
                            n_trials = 75)


print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)


