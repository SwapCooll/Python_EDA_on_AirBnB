#!/usr/bin/env python
# coding: utf-8

# ## Project Name - Airbnb Booking Analysis 

# **Project Type - EDA
#   Contribution - Individual
#   Name - Swapnil Bodhe**

# About Airbnb
# Airbnb is an online marketplace that connects people who want to rent out their property with people who are looking for accommodations in specific locales. Airbnb offers people an easy, relatively stress-free way to earn some income from their property.

# ##### GitHub Link - https://github.com/SwapCooll/Python_EDA_on_AirBnB
#  

# #### 1.Know Your Data

# In[ ]:


#import Libraries 
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


from google.colab import drive

drive.mount('/content/drive/')


# In[130]:


# load and Read dataset
df = pd.read_csv(r'C:\Users\swapn\Downloads\Airbnb NYC\AB_NYC_2019.csv')


# In[129]:


#Dataset first view
df.head()


# In[85]:


new_df.tail()


# In[33]:


#Count Rows and Columns of Dataset
df.shape


# In[34]:


#info about Dataset
df.info()


# In[ ]:


#Duplicate row values check in dataset


# In[35]:


#Duplicate row values check in dataset
Duplicate_row =df[df.duplicated()]
Duplicate_row


# In[49]:


#Missing Values/ null values
df.isnull().sum()


# In[45]:


#Visualizing the missing  values
import missingno as msno
msno.bar(df,figsize=(18,5))


# **What did you know about your dataset?By shallow looking this dataset We come to know about primary information like**
# *This Airbnb dataset contains 16 Columns and 48895 Rows.
# *In this there is no Duplicate values.
# *some of attiributes contains Null values and Missing values.

# 2. Understanding Variables

# In[47]:


#dataset columns
Dataset_Columns= df.columns
Dataset_Columns


# **Variables Description**
The features in the dataset can be described as follows:

id - This is the identity number of the property listed by a particular host.
name - It stands for the name of the property listed by the host.
host_id - It is the identity number of the hosts who have registered on Airbnb website.
host_name - These are the names of the hosts who have listed their properties.
neighbourhood_group - These are the names of the neighbourhood groups present in the NYC.
neighbourhood - These are the names of the neighbourhood present in the neighbourhood groups in NYC.
latitude - These represent the coordinates of latitude of the property listed.
longitude - These represent the coordinates of longitude of the property listed.
room type - This represent the various types of room listed by host.
price - This is the rent of the property listed in USD.
minimum nights - This represent the minimum number of nights customer rented the property.
Number_of_reviews - This represent the number of customers reviewed the property.
last_review - This represent the date when the property was last reviewed.
reviews_per_month - It is the count of reviews per month which the property received.
calculated_host_listings_count - It is the number of listings done by a particular host.
Availability_365 - This represent the number of days the property is available among 365 days.
# In[ ]:





# ### 3. Data Wrangling

# In[48]:


#filling missing values
df['name'].fillna('Absent', inplace= True)
df["host_name"].fillna('Absent', inplace =True)


# In[52]:


#Dropping unnecesarry columns
new_df = df.drop(['latitude','longitude','last_review','reviews_per_month'], axis =1)
new_df.head()


# In[71]:


#After the data cleaning 
print(f'The number of missing values after cleaning the data are:')
new_df.isnull().sum()


# In[ ]:





# #### 4. Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables and find Insights

# 1. What can we learn about different hosts and areas?

# In[61]:


#who has hightest listing?
Hosting = new_df.groupby(['host_name'])['calculated_host_listings_count'].max().reset_index()
Highest_listing = Hosting.sort_values(by='calculated_host_listings_count', ascending=False).head()
Highest_listing


# In[70]:


#visulisation of highest listings
plt.rcParams['figure.figsize'] = (12,6)
host_name = Highest_listing['host_name']
host_lisitng = Highest_listing['calculated_host_listings_count']
plt.bar(host_name,host_lisitng)
plt.title('Hosts with most listings in NYC',{'fontsize':15})
plt.xlabel('Host Names',{'fontsize':15})
plt.ylabel('Number of host listings',{'fontsize':13})
plt.show()


# **Findings**
# As we can see from above results,Host name Sonder(NYC) has the highest listing of 327 .

# In[75]:


#Which area has highest listing?

Host_area = new_df.groupby(['neighbourhood_group'])['id'].count().reset_index().rename(columns = {'id':'count'}).sort_values(by='count', ascending = False)
Host_area.head()


# In[78]:


# visulisation of highest listings
plt.rcParams['figure.figsize'] = (12,5)
neighbourhood_group = Host_area['neighbourhood_group']
count = Host_area['count']
plt.bar(neighbourhood_group,count)
plt.title('Number of listings upon neighbourhood group',{'fontsize':15})
plt.xlabel('Neighbourhood Group',{'fontsize':15})
plt.ylabel('Number of listings',{'fontsize':13})
plt.show()


# Findings
# As we can see from above results,Highest amount of listings has been done from Manhattan area.

#  2. What can we learn from predictions? (ex: locations, prices, reviews, etc)

# In[83]:


# Correlation matrix

plt.figure(figsize=(8,5))
sns.heatmap(new_df.corr(),annot=True)


# **Findings**
# By correaltion Matrix we come to know that there no much correlaton in any attributes.

# In[88]:


#Which area has Highest ratings.

Areas_reviews =new_df.groupby(['neighbourhood_group'])['number_of_reviews'].max().reset_index().sort_values(by= 'number_of_reviews', ascending= False)
Areas_reviews


# In[94]:


# Understand by visulisation
plt.rcParams['figure.figsize'] = (12,8)
neighbourhood_group = Areas_reviews['neighbourhood_group']
number_of_reviews = Areas_reviews['number_of_reviews']
plt.bar(neighbourhood_group,number_of_reviews)
plt.title('Number of reviews per Neighbourhood Group ',{'fontsize':15})
plt.xlabel('Neighbourhood Group',{'fontsize':13})
plt.ylabel('number_of_reviews',{'fontsize':13})
plt.show()


# **Findings**
# By this chart we can clearly see that Queens and Manhattan are mostly reviewd Area.

# In[95]:


# Price distribution in each Neighbourhood group
sns.boxplot(y='price',x ='neighbourhood_group',data=new_df)


# In[96]:


# Checking Price Data
df.agg({'price':['mean','median','max','min','count']})


# **Findings**
#  From looking this boxplot and data we are concluding that there is outliers in this data of price because there is minimum price is zero and maximum price is 10000 which and mean and median is not close so data is not normaly distributed.
# 
#  To overcome this we can remove upper and lower 10 percent of data.

# In[97]:


# Lower and Upper quatile of Price Data
Lower = new_df['price'].quantile(0.10)
Upper = new_df['price'].quantile(0.90)
print(Lower,Upper)


# In[98]:


# Upper and lower 10 percent is removing
new_df = new_df.drop(new_df[new_df['price']<Lower].index)
new_df = new_df.drop(new_df[new_df['price']>Upper].index)


# In[100]:


new_df.agg({'price':['mean','median','max','min','count']})


# In[101]:


# Price distribution in each Neighbourhood group
sns.boxplot(y='price',x ='neighbourhood_group',data=new_df)


# **Findings**
# Fom this box plot we clearly see that most of the Manhattan price range is Higher then afer Brooklyn.
# Queens,Staten isalnd and Bronx has affordable pricing range.

# #### 3.Which hosts are the busiest and why?

# In[103]:


# Busiest Host Findings by number of reviews

Busiest_hosts = new_df.groupby(['host_name','host_id','room_type','neighbourhood_group'])['number_of_reviews'].max().reset_index()
Top_10_Busiest_hosts = Busiest_hosts.sort_values(by='number_of_reviews',ascending=False).head(10)
Top_10_Busiest_hosts


# In[113]:


sns.barplot(x='host_name', y='number_of_reviews',data=Top_10_Busiest_hosts).set_title('Top 10 Busiest Host')


# In[114]:


# Histogram on most number of room type listing
sns.histplot(new_df['room_type'])


# **Findings**
# Top 5 Busiest Host are
# 1 Ji
# 2 Carol
# 3 Asa
# 4 Wanda
# 5 Linda
# Because they have listed their properties in most required catagory of room type which is Privet Room or Entire Home.

# Is there any noticeable difference of traffic among different areas and what could be the reason for it?

# In[115]:


Traffic_areas = new_df.groupby(['neighbourhood_group','neighbourhood','room_type'])['minimum_nights'].count().reset_index()
Top_busiest_area = Traffic_areas.sort_values(by='minimum_nights', ascending=False).head(10)
Top_busiest_area


# In[118]:


N = new_df.groupby(['neighbourhood'])['id'].count().nlargest(10)
N


# In[125]:


plt.figure(figsize=(15,8))
x = list(N.index)
y = list(N.values)
y.reverse()
x.reverse()
plt.title("Top 10 Neighbourhoods with the Most Listings", {'fontsize':15})
plt.ylabel("Neighbourhood", {'fontsize':18})
plt.xlabel("Total Listings", {'fontsize':18})

plt.barh(x, y)
plt.show()


# **Findings**
# Mosly traffic is generated in **Williamsburg , Bedford-Stuyvesant** like area which mainly under brooklyne neighbourhood area also **Harlem , East village** like area are under Manhattan neighbourhood.
# 
# Beacuse they have more listing of Privet room or Entire Home

# **All Findings**
# Host name Sonder(NYC) has the highest listing of 327
# 
# Highest amount of listings has been done from Manhattan area.
# 
# Most of the Manhattan price range is Higher then afer Brooklyn. Queens,Staten isalnd and Bronx has affordable pricing range
# 
# Busiest Hosts have listed their properties in most required catagory of room type which is Privet Room or Entire Home.
# 
# Mosly traffic is generated in Williamsburg , Bedford-Stuyvesant like area which mainly under brooklyne neighbourhood area also Harlem , East village like area are under Manhattan neighbourhood.Beacuse they have more listing of Privet room or Entire Home

# **Conclusion**
# From above analysis we come to know certain points like
# 
# Most of the listing has been done in Manhattan area and people chose to pay high for their convenince of Personal room type and Privet Houses.
# After Manhattn, Brooklyn got the second number in terms high price listings.
# Queens,Staten isalnd and Bronx areas are under affordable pricing range.In which most of people chose to go.
# Most of the people are willing to pay more if the listing is in good area and have all required amenities with personal space.

# **THANK YOU**
