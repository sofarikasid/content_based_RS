#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import ast
#from recommender_system import ContentBasedRS, Recommender

# Load the data
data = pd.read_csv('item_descr_train_doc2vec.csv', index_col="article_id")
user_profile = pd.read_csv('user_profile_train_and_relevant.csv').drop('Unnamed: 0', axis=1)

import numpy as np
import ast

class ContentBasedRS:
    
    def __init__(self, data, k):
        self.data = data
        self.k = k
        
    def compute_recommendation(self, user_previous_purchase):
        "This method computes a content-based recommendation for a given user"
        Data_ = np.array(self.data)
        user_cenriod = self.data.loc[user_previous_purchase].mean()
        D_norm = np.array([np.linalg.norm(Data_[i]) for i in range(len(Data_))])
        x_norm = np.linalg.norm(user_cenriod)
        sims = np.dot(Data_,user_cenriod)/(D_norm * x_norm)
        dists = 1 - sims
        idx = np.argsort(dists)
        return idx[:self.k], sims[idx][:self.k]
    

class Recommender:
    
    def __init__(self, user_profile, data, k):
        self.user_profile = user_profile
        self.data = data
        self.k = k
        self.recommendations = []
        self.cosine_sims = []
        
    def run_recommendation(self):
        "This method runs the recommendation on all users in the user_profile"
        for user in range(len(self.user_profile)):
            rs = ContentBasedRS(self.data, self.k)
            similar_songs, cosine_sim = rs.compute_recommendation(self.user_profile[user])
            self.recommendations.append(similar_songs)
            self.cosine_sims.append(cosine_sim)

    def calculate_recall_precision(self, relevance_user_purchases):
        "This method calculates recall and precision for the recommendations"
        recalls = []
        precisions = []
        for i in range(len(self.recommendations)):
            recs = self.recommendations[i]
            rels = relevance_user_purchases[i]
            common_items = set(recs).intersection(set(rels))
            recall = len(common_items) / len(rels)
            precision = len(common_items) / len(recs) if len(recs) > 0 else 0
            recalls.append(recall)
            precisions.append(precision)
        return np.mean(recalls), np.mean(precisions)


import pandas as pd

data = pd.read_csv('item_descr_train_doc2vec.csv', index_col="article_id")
user_profile = pd.read_csv('user_profile_train_and_relevant.csv').drop('Unnamed: 0', axis=1)

k = 10
batch_size = 5

batch_input = user_profile[:batch_size]
train_data = np.array(batch_input['user_train_items'])
relevance_data = np.array(batch_input['relevant_items'])

relevance_user_purchases = []
user_purchases = []

for items_purchased in range(len(train_data)):
    lst = ast.literal_eval(train_data[items_purchased])
    user_purchases.append(lst)
    
for relevant in range(len(relevance_data)):
    lst = ast.literal_eval(relevance_data[relevant])
    relevance_user_purchases.append(lst)

rec = Recommender(user_purchases, data, k)
rec.run_recommendation()

recall, precision = rec.calculate_recall_precision(relevance_user_purchases)

print("Recall: ", recall)
print("Precision: ", precision)


st.title('Recommender System')
st.write(data.head())


# Convert the user profile to a list of purchases
user_purchases = []
for items_purchased in range(len(train_data)):
    lst = ast.literal_eval(user_profile['user_train_items'][items_purchased])
    user_purchases.append(lst)

rec = Recommender(user_purchases, data, k)
rec.run_recommendation()

# Define the Streamlit app
def app():
    st.title('Recommender System')
    
    # Add a dropdown menu to select a user
    user_id = st.selectbox('Select a user:', range(len(user_purchases)))
    
    # Display the top recommendations for the selected user
    st.subheader(f'Top {k} recommendations for user {user_id}:')
    recommendation=rec.recommendations[user_id]
    items= pd.read_csv("articles.csv",index_col="article_id")
    recomendatio_item=items.iloc[recommendation,[1,20]]
    recomendatio_item["cosine_sim"]=rec.cosine_sims[user_id]
    recomendatio_item
    #for idx in recommendations:
        #st.write(data.index[idx])

app()

    






