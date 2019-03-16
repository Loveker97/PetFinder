import pandas as pd
import numpy as np
import seaborn as sns
import zipfile
import os.path
from os import path
import matplotlib.pyplot as plt
import difflib

# Load the data 
def load_data(file_path):
	'''
	Loads the data
	Input: 
		File path 
	Output: 
		Breed Labels, 
		Color Labels,
		State Labels,
		Training data
		'''
	breed_labels = pd.read_csv(file_path + "breed_labels.csv")
	color_labels = pd.read_csv(file_path + "color_labels.csv")
	state_labels = pd.read_csv(file_path + "state_labels.csv")

	# Check if file already exists
	if not path.exists(file_path + 'train.csv'):
	    train_zip = zipfile.ZipFile(file_path + 'train.zip')
	    train_zip.extractall(file_path)
	    train_zip.close()

	train = pd.read_csv(file_path + "train.csv")

	return train, breed_labels, color_labels, state_labels

# Initial look at the data
def initial_analysis(data):
    '''
    Presents an overview of the data
    
    Inputs: 
        data: of type Pandas.DataFrame object
    Outputs:
        First 5 data points
        Statistical info of all the features
        Histograms
    '''
    assert isinstance(data, pd.DataFrame)
    print(data.head())
    print(data.describe())
    
    # Histogram plots of the all features 
    data.hist(bins=50, figsize=(20,15))
    plt.show()

def state_dictionary(state_labels):
    '''
    Input: 
        state_labels: State labels
    Output: 
        Dictionary defining stateID and StateName correspondence
    '''
    return dict(zip(state_labels["StateID"], state_labels["StateName"]))

def state_vs_HDI(state_labels, state_dict, train_copy):
    '''
    Input: 
        state_labels: State labels,
        train_copy: Data
    Output: 
        x,y, bar_color: For plotting
    '''
    # Correlating top states with HDI 
    state_labels['HDI'] = [0.785, 0.769, 0.741, 0.822, 
                           0.742, 0.794, 0.789, 0.766, 
                           0.778, 0.767, 0.803, 0.674, 
                           .707, 0.819, 0.762]

    # Idenify Top States
    top_states = train_copy['State'].value_counts()
    x = []
    y = []
    bar_color = []
    for state in top_states.index:
        x.append(state_dict[state])
        y.append(top_states[state])
        bar_color.append(state_labels.loc[state_labels['StateID'] == state]['HDI'].values[0])
        print(state_dict[state], state_labels.loc[state_labels['StateID'] == state]['HDI'].values[0])

    return x,y, bar_color 

def recommender_system(state_dict, breed_dict, train_copy):
    '''
    A simple recommender for pet adoption!!!
    Input Arguments:
        state_dict: Dictionary defining stateID and StateName correspondence
        breed_dict: Dictionary defining breedID and BreedName correspondence
        train_copy: Data


    User Inputs: 
        Name of the state
        Type of Pet {dog OR cat}
        Breed of the Pet
        
    Output:
        Top 5 Pet Profiles that fit the input information-
        -based on higher chances of adoptability
    '''
    
    state = input("Enter State: ")
    pet_type = input("Type of Pet: ")
    pet_breed = input("Breed: ")
    
    inv_state_dict = {v: k for k, v in state_dict.items()}
    
    state = difflib.get_close_matches(state, inv_state_dict.keys(), n=1, cutoff=0)[0]

    if state not in inv_state_dict:
        print("Enter a valid State, Thank you!")
        return 
    else:
        stateID = inv_state_dict[state]
        
    if pet_type.lower() =='dog':
        pet_type = 1
    elif pet_type.lower() =='cat':
        pet_type = 2
    else:
        print("Enter a valid Pet Type, we only have cats and dogs, Thank you!")
    
    pet_breed = difflib.get_close_matches(pet_breed, breed_dict.keys(), n=1, cutoff=0)[0]
    
    if pet_breed not in breed_dict:
        print("Enter a valid breed, Thank you")
        return
    else:
        pet_breed = breed_dict[pet_breed]
    
    pet_data = train_copy[(train_copy['State'] == stateID) & (train_copy['Breed1'] == pet_breed) & (train_copy['Type'] == pet_type)]
    pet_data = pet_data.sort_values(by=['AdoptionSpeed'])
    
    pet_data = pet_data.drop(['State', 'Color1', 'Color2', 'Color3', 'Breed1', 'Breed2', 
                              'RescuerID', 'PhotoAmt', 'VideoAmt', 'AdoptionSpeed',
                             'MaturitySize', 'Health'], axis =1)
    
    pet_data['Vaccinated'] = pet_data['Vaccinated'].map({1:'Yes', 2:'No', 3:'Not Sure'})
    pet_data['Dewormed'] = pet_data['Dewormed'].map({1:'Yes', 2:'No', 3:'Not Sure'})
    pet_data['Sterilized'] = pet_data['Sterilized'].map({1:'Yes', 2:'No', 3:'Not Sure'})

    
    return pet_data.head()

