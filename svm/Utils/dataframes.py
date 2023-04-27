import pandas as pd
import json

def Dict_to_df(jacc_dict):
    
    # converting dict to pd.DataFrame 
    df2 = pd.DataFrame.from_dict(jacc_dict, orient='index')
    # only pulling out tone because thats our label
    tones = pd.DataFrame(df2.tone)

    # convert vectory array to list so that each vector has its own column
    jaccard_df = pd.DataFrame(df2.vec.tolist())
    return (jaccard_df, tones)

####################################
# Function that extracts the jaccard variables and tones
# into two different dataframes
#
# Param: Str:= filepath to json file 
#
#Return: Two DF's = (Jaccard Variables, Tones)
####################################
def json_to_dataframe(file_name :str):
    # Opening and loading jaccard distance tri-tone file
    file = open(file_name)
    #imported as a dict
    data = json.load(file)
    file.close()
    # converting dict to pd.DataFrame 
    df2 = pd.DataFrame.from_dict(data, orient='index')
    # only pulling out tone because thats our label
    tones = pd.DataFrame(df2.tone)
    ##########################
    #print(df2) # to see issue
    ##########################

    # convert vectory array to list so that each vector has its own column
    jaccard_df = pd.DataFrame(df2.vec.tolist())

    # # merging tones and new dataframe together
    # jaccard_df = pd.DataFrame(np.hstack([dataframe, tones])) # use np.hstack instead of pd.concat becuase of error issues
    # #rename last column to Tone
    # jaccard_df.rename(columns={jaccard_df.columns[332]: "Tone"}, inplace=True)
    # jaccard_df 
    
    # Our feature
    return (jaccard_df, tones)

##############################
# Categorically encodes the tones 
# to negative: 0, neural: 1, and positive: 2
# 
# Param: DF:= Tones Label 
#
# Return: tones encoded
#############################
def Category_encode(tones): 
    t = pd.Categorical(tones['tone'])
    tones['tone'] = t.codes
    return tones