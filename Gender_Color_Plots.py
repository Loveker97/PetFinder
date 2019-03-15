
# coding: utf-8

# ## Average Adoption Speed for Each Gender

# In[5]:


def adopt_gender_state(state):
    
    """
    Plots bar graphs of average adoption speed of dogs and cats divided by gender in a specified state
    
    :param state: str specifies which state to display data about
    
    The adoption speed scale is 0-4 where 0 means immediate adoption and 4 means no adoption after
    91 days at the shelter.
    
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools

    assert isinstance(state, str)
    
    train = pd.read_csv('./data/train.csv')
    state_labels = pd.read_csv('./data/state_labels.csv')
    
    # Dictionary for state labels
    state_dict = dict(zip(state_labels["StateName"], state_labels["StateID"]))
    
    assert state in state_dict, "state not in state dictionary"
    
    # Split data to only contain given state data
    this_state = train.loc[train['State'] == state_dict[state],
                                ['State','Type', 'Gender', 'AdoptionSpeed']]
    

    # Divide by dog (Type = 1) and cat (Type = 2)
    dog_df = this_state.loc[this_state['Type'] == 1, :]
    cat_df = this_state.loc[this_state['Type'] == 2, :]
    
    labels = ['Male', 'Female']
    dog_avg = []
    cat_avg = []
    
    # Find average adoption speed for each gender
    # Male (Gender = 1) and Female (Gender = 2)
    dog_avg.append(dog_df.loc[dog_df['Gender'] == 1, ['AdoptionSpeed']].mean()[0])
    dog_avg.append(dog_df.loc[dog_df['Gender'] == 2, ['AdoptionSpeed']].mean()[0])
    
    cat_avg.append(cat_df.loc[cat_df['Gender'] == 1, ['AdoptionSpeed']].mean()[0])
    cat_avg.append(cat_df.loc[cat_df['Gender'] == 2, ['AdoptionSpeed']].mean()[0])
    
    # Plot bar graph
    plt.figure()
    index = np.arange(len(labels))
    plt.bar(index, dog_avg)
    plt.xlabel('Gender', fontsize = 5)
    plt.xticks(index, labels, fontsize = 5)
    plt.ylabel('Adoption Speed', fontsize = 5)
    plt.title('Dog Average Adoption Speed for Each Gender in ' + state)
    
    plt.figure()
    index = np.arange(len(labels))
    plt.bar(index, cat_avg)
    plt.xlabel('Gender', fontsize = 5)
    plt.xticks(index, labels, fontsize = 5)
    plt.ylabel('Adoption Speed', fontsize = 5)
    plt.title('Cat Average Adoption Speed for Each Gender in ' + state)
    
adopt_gender_state('Kuala Lumpur')


# In[4]:


def adopt_gender():
    
    """
    Plots bar graphs of average adoption rate of dogs and cats divided by gender 
    in total dataset
    
    The adoption speed scale is 0-4 where 0 means immediate adoption and 4 means no adoption after
    91 days at the shelter.
    
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools

    train = pd.read_csv('./data/train.csv')

    # Divide by dog (Type = 1) and cat (Type = 2)
    dog_df = train.loc[train['Type'] == 1, :]
    cat_df = train.loc[train['Type'] == 2, :]
    
    labels = ['Male', 'Female']
    dog_avg = []
    cat_avg = []
    
    # Find average adoption speed for each gender
    # Male (Gender = 1) and Female (Gender = 2)
    dog_avg.append(dog_df.loc[dog_df['Gender'] == 1, ['AdoptionSpeed']].mean()[0])
    dog_avg.append(dog_df.loc[dog_df['Gender'] == 2, ['AdoptionSpeed']].mean()[0])
    
    cat_avg.append(cat_df.loc[cat_df['Gender'] == 1, ['AdoptionSpeed']].mean()[0])
    cat_avg.append(cat_df.loc[cat_df['Gender'] == 2, ['AdoptionSpeed']].mean()[0])
    
    # Plot bar graph
    plt.figure()
    index = np.arange(len(labels))
    plt.bar(index, dog_avg)
    plt.xlabel('Gender', fontsize = 5)
    plt.xticks(index, labels, fontsize = 5)
    plt.ylabel('Adoption Speed', fontsize = 5)
    plt.title('Dog Average Adoption Speed for Each Gender')
    
    plt.figure()
    index = np.arange(len(labels))
    plt.bar(index, cat_avg)
    plt.xlabel('Gender', fontsize = 5)
    plt.xticks(index, labels, fontsize = 5)
    plt.ylabel('Adoption Speed', fontsize = 5)
    plt.title('Cat Average Adoption Speed for Each Gender')
    
adopt_gender()


# ## Color Group and Average Adoption Speed

# In[8]:


def adopt_speed_color_group():
    
    """
    Plots bar graphs of average adoption speed based on the pet's coat color group.
    The color groups are divided between:
        - Dark: brown and black
        - Light: golden, cream, gray, white, yellow
        
    The adoption speed scale is 0-4 where 0 means immediate adoption and 4 means no adoption after
    91 days at the shelter.
    
    """
    
    # Plot color group and average adoption rate
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    train = pd.read_csv('./data/train.csv')
    train.head
    dark = train[train['Color1'] < 3]
    light = train[train['Color1'] > 2]


    dark_adopt_avg = sum(dark.loc[:, 'AdoptionSpeed']) / len(dark) 
    light_adopt_avg = sum(light.loc[:, 'AdoptionSpeed']) / len(light)

    label = ['Dark', 'Light']
    avg = [dark_adopt_avg, light_adopt_avg]
    index = np.arange(2)

    plt.bar(index, avg)
    plt.xlabel('Color', fontsize = 5)
    plt.ylabel('Average Adoption Speed')
    plt.xticks(index, label, fontsize = 5)
    plt.title('Average Adoption Speed Based on Color Group')
    plt.show()
    
    
adopt_speed_color_group()


# ## Color and Average Adoption Speed

# In[9]:


def adopt_speed_color():
    
    """
    Plots bar graphs of average adoption speed related to each pet coat color
    
    The adoption speed scale is 0-4 where 0 means immediate adoption and 4 means no adoption after
    91 days at the shelter.
    """
    
    # Plot color and average adoption rate
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools

    train = pd.read_csv('./data/train.csv')
    color = pd.read_csv('./data/color_labels.csv')


    # color_dict = dict(zip(color["ColorID"], color["ColorName"]))
    color_list = list(zip(color["ColorID"], color["ColorName"]))
    sub_color = []
    avg = []

    for i in range(len(color_list)) :
        color_list[i] = [color_list[i][0], color_list[i][1]]



    label = []
    for i in range(len(color_list)) :
        sub_color.append(train[train['Color1'] == color_list[i][0]])
        avg.append(sum(sub_color[i].loc[:, 'AdoptionSpeed']) / len(sub_color[i]))
        color_list[i].append(avg[i])
        label.append(color_list[i][1])

    index = np.arange(len(avg))

    plt.figure()
    plt.bar(index, avg)
    plt.xlabel('Color', fontsize = 5)
    plt.ylabel('Average Adoption Speed')
    plt.xticks(index, label, fontsize = 5)
    plt.title('Average Adoption Speed Based on Color')
    plt.show()
    
adopt_speed_color()

