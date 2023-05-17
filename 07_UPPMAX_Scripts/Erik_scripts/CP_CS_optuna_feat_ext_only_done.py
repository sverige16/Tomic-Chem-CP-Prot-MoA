
import h5py
import datetime
import time

# Torch
import torch
import torch.nn as nn
import neptune.new as neptune
import sys
import optuna 
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')

from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    different_loss_functions, 
    apply_class_weights_GE, 
    adapt_training_loop, 
    set_parameter_requires_grad,
    smiles_to_array,
    extracting_pretrained_single_models,
    CP_driving_code,
    returning_smile_string,
    dubbelcheck_dataset_length,
)
from Helper_Models import (image_network, Chemical_Structure_Model)


class CS_CP_Model(nn.Module):
    def __init__(self, modelCP, modelCS, init_nodes):
        super(CS_CP_Model, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.linear_layer_pre = nn.Linear(int(1280 + 103), init_nodes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm = nn.BatchNorm1d(init_nodes)

    def forward(self, x1in, x2in):
        x1 = self.modelCP(x1in)
        x2 = self.modelCS(x2in)
        x  = torch.cat((torch.squeeze(x1), x2), dim = 1)
        x = self.leaky_relu(self.linear_layer_pre(x))
        return x

class CS_CP_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 meta_data,  
                 checking_mechanism, 
                 dict_moa, 
                compound_df,
                images_np
                 ):
        self.meta_data = meta_data
        self.dict_moa = dict_moa
        self.check = checking_mechanism
        self.compound_df = compound_df
        self.hdf5_file = images_np

    def __len__(self):
        check_criteria = self.check
        assert len(self.meta_data) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.meta_data)
    
    def __getitem__(self,idx):
        '''Retrieving the compound'''
        meta_data_idx = self.meta_data.index[idx]  
        label = self.meta_data["moa"][meta_data_idx]
        cmpdID = self.meta_data["compound"][meta_data_idx]
        
        with h5py.File(self.hdf5_file, 'r') as f:
            image = torch.from_numpy(f['CP_data'][meta_data_idx])

        smile_string = returning_smile_string(self.compound_df, cmpdID)
        compound_array = smiles_to_array(smile_string) 
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        return torch.squeeze(image.float()), compound_array.float(), label_tensor.float() # returns 


#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'CP_CS'
#---------------------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------#


file_name = 'erik10_hq_8_12'
fold_int = 0
training_meta_data, validation_meta_data, test_df_meta_data, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, images = CP_driving_code(file_name, fold_int)

images = '/home/jovyan/scratch-shared/erikp/CP_file.h5'
# --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
# Create datasets with relevant data and labels
training_dataset_CSCP = CS_CP_Dataset(meta_data =training_meta_data,   
                 checking_mechanism = ["train" , "CP", fold_int], 
                 dict_moa = dict_moa, 
                compound_df = training_set_cmpds,
                images_np = images)
valid_dataset_CSCP = CS_CP_Dataset(
                 meta_data = validation_meta_data, 
                 checking_mechanism = ["valid" , "CP", fold_int],
                 dict_moa = dict_moa, 
                compound_df = validation_set_cmpds,
                images_np = images)
test_dataset_CSCP = CS_CP_Dataset(
                 meta_data = test_df_meta_data,  
                 checking_mechanism = ["test" , "CP", fold_int],
                 dict_moa = dict_moa, 
                compound_df = test_set_cmpds,
                images_np = images)

# parameters for the dataloader
batch_size = 7
params = {'batch_size' : batch_size,
         'num_workers' : 2,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )


# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSCP, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSCP, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSCP, **params)

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
    '''
    Using Optuna to find the best hyperparameters for the model
    '''

        
    class Extended_Model(nn.Module):
        '''
        Extending the model with one or more fully connected layer
        '''
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
                    p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
                    layers.append(nn.Dropout(p))
                    in_features = out_features
                layers.append(nn.Linear(in_features, num_classes))
            else:
                layers = [nn.Linear(in_features, num_classes)]
            # Additional layers for feature extraction
            self.additional_layers = nn.Sequential(*layers)
            print(self.additional_layers)
        def forward(self, x, y):
            x = self.base_model(x, y)
            x = self.additional_layers(x)
            return x
        
    # load individual models
    print("Loading Pretrained Models...")
    modelCS = Chemical_Structure_Model(num_features = 2048, num_targets=   num_classes)
    modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
    modelCP = image_network()
    modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])

    # create a model combining both models
    init_nodes = trial.suggest_int('init_nodes', 4, 250)
    pretrained_model = CS_CP_Model(modelCP, modelCS, init_nodes)
    n_layers = trial.suggest_int('n_layers', 0, 3)
    model = Extended_Model(trial, pretrained_model, num_classes, n_layers) 
    set_parameter_requires_grad(model, feature_extracting = True, added_layers = 0)

    # hyperparameters for trial
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 2, 9), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 2, 9))

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
        max_epochs = 10
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
                            num_classes = 10, 
                            training_generator = training_generator,
                            validation_generator = validation_generator,
                        
                            model_name = model_name, 
                            device = device, 
                            driver = "CP", 
                            fe_ft = "fe",
                            dict_moa = dict_moa,
                            df_train_labels = training_meta_data),
                            n_trials = 75)


print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/data_for_models/" + model_name + '_' + fe_ft + '_' + now +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()


