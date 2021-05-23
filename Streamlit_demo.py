import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import os

st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache
def load_data():
    url='https://en.wikipedia.org/wiki/List_of_BSE_SENSEX_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df
 
def load_data2(): 
	# assign data of lists.  
	data = {'Sector': ['Chemical', 'Chemical', 'Chemical', 'Chemical', 'Banking'], 'Name': ['Aarti Industries', 'Aarti Surfactant', 'Advanced Enzyme', 'Alkyl Amines', 'Axis']}  
  
	# Create DataFrame  
	df = pd.DataFrame(data)  
	return df


df = load_data2()
sector = df.groupby('Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[ (df['Sector'].isin(selected_sector)) ]

# Sidebar - Company selection
company_names = sorted( df['Name'])
selected_company_names = st.sidebar.multiselect('Name', company_names, company_names)

df_selected_company_names = df.loc[((df['Sector'].isin(selected_sector)) & (df['Name'].isin(selected_company_names)))]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_company_names)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

#st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
        tickers = list(df_selected_sector[:10]),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()

def display_balance_sheet(name):
	targetPathBase = "C:\MACps\Lakshman\Govind\Chemicals"
	targetPath = os.path.join(targetPathBase, name, name+'.xlsx')
	df = pd.read_excel(targetPath, sheet_name = 'Balance_sheet')
	return st.dataframe(df)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    #for i in list(df_selected_sector.Symbol)[:num_company]:
        #price_plot(i)

num_company = st.sidebar.slider('Number of Companies', 1, 5)

if st.button('Show Balance sheets of selected Stocks'):
    for each_selected_company in list(df_selected_company_names.Name): #[:num_company]:
	    st.header('Stock Balance Sheets {0}'.format(each_selected_company))
	    display_balance_sheet(each_selected_company)
		
	#	price_plot(i)
