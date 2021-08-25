# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
from string import punctuation
import plotly.express as px
import itertools

@st.cache
def Load_Data(path):
    data = pd.read_csv(path)
    data['date'] = data['invoicedate'].apply(lambda x: date.fromisoformat(x))
    data['invoicedate'] = data['invoicedate'].astype('datetime64[D]')
    for col in ['invoiceno', 'stockcode']:
        data[col] = data[col].astype('category')
    
    return data


# def Filter_Data(df):
#     timerange = st.slider('CHOSE TIME RANGE:', value=(df['date'].min(), df['date'].max()))
#     new_df = df[(df['date']>=timerange[0]) & (df['date']<=timerange[1])].reindex()
#     return new_df


def Time_filter(data):
    col1, col2 = st.beta_columns(2)
    with col1:
        timerange = st.slider('CHOSE TIME RANGE:', value=(data['date'].min(), data['date'].max()))
        df = data[(data['date']>=timerange[0]) & (data['date']<=timerange[1])]

        timeround = {'date': 'datetime64[D]', 'month':'datetime64[M]', 'year':'datetime64[Y]'}
        t = st.radio('Round up by', ['date','month','year'],0)
        df['invoicedate'] = df['invoicedate'].astype(timeround[t])

    with col2:
        fig, ax = plt.subplots()
        df[df['quantity']>0].groupby('invoicedate')[['stockcode', 'invoiceno']].nunique().plot.line(ax=ax)
        ax.set(xlabel='Time', ylabel='Stockcode/Invoiceno Count', title='NUMBER OF ORDERS, PRODUCTS AND TOTAL REVENUE THROUGH TIME')
        ax.legend(loc='upper left')
        ax2 = ax.twinx()
    
        df.groupby('invoicedate')['revenue'].sum().plot.line(ax=ax2, color='green', label='revenue')
        ax2.set(ylabel='Total Revenue')
        ax2.legend(loc='upper right')

        st.pyplot(fig)

    return df, t


def Product_Value_Calculation(df, n_cluster, t):
    product = pd.read_csv(product_path, index_col='stockcode')
    product['keyword'] = product['keyword'].apply(lambda x: set(x.translate({ord(i): None for i in punctuation}).split()))

    RFP = df[df['quantity']>0].groupby('stockcode').agg({'invoicedate': lambda x: today - x.max(), 'invoiceno': 'nunique', 'unitprice': 'mean'}).rename(columns={'invoicedate': 'recency', 'invoiceno': 'frequency'})
    M = df.groupby('stockcode')['revenue'].sum().rename('monetary')

    product_f = product.join([RFP, M]).dropna()

    timeunit = {'date': 'D', 'month':'M', 'year':'Y'}
    product_f['recency'] = product_f['recency'].apply(lambda x: x/np.timedelta64(1,timeunit[t]))
    product_f['group'] = 0

    if n_cluster>=2:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        X = StandardScaler().fit_transform(product_f[['recency', 'frequency','monetary','unitprice']])
        product_f['group'] = KMeans(n_cluster, random_state=0).fit_predict(X)

    product_f.sort_values('group', inplace=True)
    product_f['group'] = product_f['group'].apply(lambda x: str(x))


    fig = px.scatter_3d(product_f, x='recency', y='frequency', z='monetary', size='unitprice', color='group', size_max=25, hover_name='description', color_discrete_sequence=px.colors.qualitative.Dark24, title='PRODUCT RECENCY - FREQUENCY - MONETARY - UNITPRICE (size) DISTRIBUTION')
    st.plotly_chart(fig)
    st.write('Analyze {} products in {} database products'.format(len(product_f), len(product)))

    return product_f


def Total_Keyword(df):
    top_keyword = st.number_input('Number of top keywords for showing:', step=1, min_value=0, value=10)
    word_count = pd.Series(list(itertools.chain.from_iterable(df['keyword']))).value_counts()

    col1, col2 = st.beta_columns(2)
    with col2:
        st.table(word_count.describe().rename('Keyword count distribution'))

    with col1:
        if len(word_count)<=top_keyword:
            st.table(word_count.rename('Top keyword count'))
        else:
            st.table(word_count.head(round(top_keyword)).rename('Top keyword count'))


def Group_Keyword(df, threshold):
    word_count = pd.Series(list(itertools.chain.from_iterable(df['keyword']))).value_counts()
    report = {}
    for group in df.drop_duplicates('group')['group']:
        report[group] = Keyword_Compare(df[df['group']==group]['keyword'], reference=word_count[word_count> threshold], reference_size=len(df)) 
    report = pd.DataFrame.from_dict(report, orient='index')
    st.table(report)


def Keyword_Compare (sample, reference, reference_size, alpha=0.05):
    count0 = pd.Series(list(itertools.chain.from_iterable(sample))).value_counts()
    words = list(itertools.chain.from_iterable(sample))
    count = pd.Series({w: words.count(w) for w in reference.index}).sort_values()
    ztest = pd.DataFrame.from_dict({w: Proportion_Z_Test ([reference.loc[w], count.loc[w]], [reference_size, len(sample)]) for w in reference.index}, orient='index')
    report = {} 
    report['groupsize'] = len(sample)
    report['top keyword not in reference'] = [w for w in count0.index[:10] if w not in reference.index]
    report['higher proportion than reference'] = ztest[(ztest['pvalue']<=alpha) & (ztest['rdiff']>0)].sort_values('rdiff', ascending=False).index.tolist()
    report['lower proportion than reference'] = ztest[(ztest['pvalue']<=alpha) & (ztest['rdiff']<0)].sort_values('rdiff').index.tolist() 
    return pd.Series(report)


def Proportion_Z_Test (success, trial):
    import math
    import scipy.stats as st
    p1 = success[0]/trial[0] 
    p2 = success[1]/trial[1]
    p_combined = (success[0] + success[1]) / (trial[0] + trial[1])
    z_value = (p1-p2) / math.sqrt(p_combined * (1 - p_combined) * (1/trial[0] + 1/trial[1]))
    distr = st.norm(0, 1)
    p_value = (1 - distr.cdf(abs(z_value))) * 2
    rel_diff = (p2-p1)/p1
    return {'pvalue':p_value, 'zvalue': z_value, 'rdiff': rel_diff, 'p1':p1, 'p2':p2}



# %%
ecom_path = ('ecom_df.csv')
product_path = ('ecom_product.csv')

st.title('PRODUCT RANGE ANALYSIS')

data_load_state = st.text('Loading data...')
data = Load_Data(ecom_path)
data_load_state.text('Dataset: Yandex Practicum "ecommerce_dataset_us.csv"')

today = data['invoicedate'].max()
#ecom_df = Filter_Data(data)

ecom_df, t = Time_filter(data)

n_cluster = st.number_input('PRODUCTS CAN BE GROUPED USING KMEANS CLUSTERING. CHOOSE NUMBER OF GROUP:', step=1, min_value=1 ,value=1)
product = Product_Value_Calculation(ecom_df, n_cluster, t)

st.subheader('PRODUCT GROUP AVERAGE METRICS:')
st.table(product.groupby('group', sort=False).mean()[['recency', 'frequency','monetary','unitprice']])

st.subheader('PRODUCT GROUP KEYWORD:')

if n_cluster<2:
    report = Total_Keyword(product)
else:
    threshold = st.number_input('Keywords of total dataset with apperance count above certain threshold are used as reference for comparision. Adjust threshold:', step=1, min_value=0, value=50)
    report = Group_Keyword(product, threshold)

