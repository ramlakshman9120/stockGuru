import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import os
import re as re
import glob
import ChromeDriverPython as cd

st.title('STOCK-GURU GOVIND')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** Screener, MoneyControl).
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
	data = {'Sector': ['Pharmaceuticals','FMCG'], 'Path': ['C:\MACps\Lakshman\Govind\Pharmaceuticals\Pharmaceuticals.xlsx', 'C:\MACps\Lakshman\Govind\FMCG\FMCG.xlsx']}  
  
	# Create DataFrame  
	df = pd.DataFrame(data)  
	return df

df = load_data2()

#Convert all the column values to integer in dataframe
def convertEachColumnToInteger(input_df):
    #for each_column in input_df.columns:
    #print(df[each_column])
    #df[each_column] = pd.to_numeric(df[each_column])#astype(int)
    for each_column in input_df.columns:
        try:
            input_df[each_column] = input_df[each_column].apply(pd.to_numeric)
            print("Converted the column to integer: {}".format(each_column))#.format(each_column))
        except:
            print("Cannot convert column :{} to integer".format(each_column))#format(each_column))
            continue
    #input_df['Mar Cap Rs.Cr.'] = input_df['Mar Cap Rs.Cr.'].astype(int)
    #input_df['CMP Rs.'] = input_df['CMP Rs.'].astype(int)
    return input_df


sector = df.groupby('Sector')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input Excel file with StockName and StockCode", type=["xlsx"])

def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return " ".join(num)

#df['mcStrength']=df['mcStrength'].apply(lambda x: find_number(x))

# get data file names
path = os.getcwd()
#filenames = glob.glob("/*.xlsx")

filePaths = glob.glob(path + "/*.xlsx")

list_of_file_names = list()
for each_filePath in filePaths:
    list_of_file_names.append(os.path.basename(each_filePath.split('.')[0]))
    print(list_of_file_names)

# Sidebar - Sector selection
#sorted_sector_unique = sorted( df['Sector'].unique() )
sorted_sector_unique = sorted(list_of_file_names)
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique[0])
print("The selected {} from list {}".format(selected_sector,sorted_sector_unique))

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file, sheet_name="Peer Comparision")
else:
    #df[df['Sector'] == selected_sector]
    #print(os.path.join(str(df['Path'])[0]))
    #df[df['Sector'].str.match(selected_sector)]
    #df[df['Sector'] in selected_sector]
    #df2 = df[df['Sector'].str.contains(selected_sector[0])]
    #path_of_selected_sector = df2['Path'].tolist()
    #input_df = pd.read_excel(os.path.join(os.getcwd(),str(selected_sector),".xlsx"), sheet_name="Peer Comparision")
    input_df = pd.read_excel(os.path.join(os.getcwd(),selected_sector[0]+'.xlsx'), sheet_name="Peer Comparision")
    #st.write(input_df)

input_df.replace(',','', regex=True, inplace=True)
#Fill nan with zero
input_df=input_df.fillna(0)
input_df['mcStrength']=input_df['mcStrength'].apply(lambda x: find_number(x))
#input_df['mcWeekness']=input_df['mcWeekness'].apply(lambda x: find_number(x))
input_df['mcPassPrec']=input_df['mcPassPrec'].apply(lambda x: find_number(x))
input_df = convertEachColumnToInteger(input_df)
print("The type of input_df['mcStrength'] is :{}".format(type(input_df['mcStrength'].tolist())))
print("The type of input_df['mcPassPrec'] is :{}".format(type(input_df['mcPassPrec'].tolist())))
print("The type of input_df['mcPiotski'] is :{}".format(type(input_df['mcPiotski'].tolist())))
#input_df['mcPiotski']=input_df['mcPiotski'].apply(lambda x: find_number(x))
#Remove commas in the table and save changes to dataframe
#Convert all the values to integers
#input_df = input_df.apply(pd.to_numeric)
# convert just columns "a" and "b"
#input_df['Mar Cap Rs.Cr.'] = input_df['Mar Cap Rs.Cr.'].astype(int)
#input_df['CMP Rs.'] = input_df['CMP Rs.'].astype(int)
#input_df[['mcStrength', 'mcStrength', 'mcPiotski']] = input_df[['mcStrength', 'mcStrength', 'mcPiotski']].apply(pd.to_numeric)


#st.header('Display Companies based on range of companies')
#num_company = st.sidebar.slider('Number of Companies', 0, len(input_df))
#st.dataframe(input_df.head(num_company))


#MICROCAP = st.sidebar.checkbox('MICROCAP')
#SMALLCAP = st.sidebar.checkbox('SMALLCAP')
#MIDCAP = st.sidebar.checkbox('MIDCAP')
#LARGECAP = st.sidebar.checkbox('LARGECAP')


marketCap_selected = st.sidebar.slider('Filter Mar Cap Rs.Cr. in range:', min(input_df['Mar Cap Rs.Cr.'].tolist()), max(input_df["Mar Cap Rs.Cr."].tolist()), value = [min(input_df['Mar Cap Rs.Cr.'].tolist()), max(input_df["Mar Cap Rs.Cr."].tolist())])

MarCap_step_value = (max(input_df["Mar Cap Rs.Cr."].tolist()) - min(input_df['Mar Cap Rs.Cr.'].tolist()))/10
defMinMarCapValue = (max(input_df["Mar Cap Rs.Cr."].tolist()) + min(input_df['Mar Cap Rs.Cr.'].tolist()))/2
defMaxMarCapValue = (max(input_df["Mar Cap Rs.Cr."].tolist()) + min(input_df['Mar Cap Rs.Cr.'].tolist()))*(3/4)
marketCap_selected_min = st.sidebar.number_input ('Select Minimum marketCap', min_value=min(input_df['Mar Cap Rs.Cr.'].tolist()), max_value=max(input_df["Mar Cap Rs.Cr."].tolist()), value=defMinMarCapValue, step=MarCap_step_value)

marketCap_selected_max = st.sidebar.number_input ('Select Maximum marketCap', min_value=min(input_df['Mar Cap Rs.Cr.'].tolist()), max_value=max(input_df["Mar Cap Rs.Cr."].tolist()), value=defMaxMarCapValue, step=MarCap_step_value)


peg_selected = st.sidebar.slider('Filter PEG in range:', min(input_df['PEG'].tolist()), max(input_df["PEG"].tolist()), value = [min(input_df['PEG'].tolist()), max(input_df["PEG"].tolist())])


peg_step_value = (max(input_df["PEG"].tolist()) - min(input_df['PEG'].tolist()))/10
defMinPEGValue = (max(input_df["PEG"].tolist()) + min(input_df['PEG'].tolist()))/4
defMaxPEGValue = (max(input_df["PEG"].tolist()) + min(input_df['PEG'].tolist()))*(3/4)
peg_selected_min = st.sidebar.number_input ('Select Minimum PEG', min_value=min(input_df['PEG'].tolist()), max_value=max(input_df["PEG"].tolist()), value=defMinPEGValue, step=peg_step_value)

peg_selected_max = st.sidebar.number_input ('Select Maximum PEG', min_value=min(input_df['PEG'].tolist()), max_value=max(input_df["PEG"].tolist()), value=defMaxPEGValue, step=peg_step_value)




#else:
#    column_names = ["Name"]
#    # create an Empty DataFrame object
#    input_df = pd.DataFrame(columns = column_names)
#    st.write('Awaiting Excel file upload from User...')


# Filtering data
df_selected_sector = df[ (df['Sector'].isin(selected_sector)) ]

# Sidebar - Company selection
#company_names = sorted( df['Name'])
#selected_company_names = st.sidebar.multiselect('Name', company_names, company_names)


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
  df.T.plot()
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
    st.dataframe(df)
    fig = plt.figure(figsize = (20, 10))
    plt.plot(df.iloc[0][1:], df.iloc[3][1:], color ='maroon')
    plt.title('Captital vs Liabilities')
    plt.xlabel('Share Capital +', fontweight='bold')
    plt.ylabel('Other Liabilities +', fontweight='bold')
    #return st.pyplot()
    st.line_chart(df.iloc[1][1:].sort_valuese())
    return st.line_chart(df.iloc[0][1:].sort_values())

#if st.button('Show Plots'):
 #   st.header('Stock Closing Price')
  #  for i in list(df_selected_sector.Symbol)[:num_company]:
    #price_plot(i)
   # ypoints = np.array([3, 8, 1, 10])
   # plt.plot(ypoints, linewidth = '20.5')
   # plt.show()   

input_df['mcStrength'] = input_df['mcStrength'].astype(int)
#input_df['mcWeekness'] = input_df['mcWeekness'].astype(int)
#input_df['mcStrength'] = input_df['mcStrength'].apply(pd.to_numeric)
#input_df['mcStrength'] = input_df['mcStrength'].astype('Int64')

mcStrength_selected = st.sidebar.slider('Filter mcStrength in range:', min(input_df['mcStrength'].tolist()), max(input_df["mcStrength"].tolist()), value = [min(input_df['mcStrength'].tolist()), max(input_df["mcStrength"].tolist())])
#first_n_companies = input_df.head(num_company);
#input_df2 = input_df[input_df['mcStrength']>=mcStrength_selected]
#st.dataframe(input_df[input_df['mcStrength']>mcStrength_selected])

#input_df['mcPiotski'] = input_df['mcPiotski'].astype(int)
mcPiotski_selected = st.sidebar.slider('Filter mcPiotski in range:', min(input_df['mcPiotski'].tolist()), max(input_df["mcPiotski"].tolist()), value = [min(input_df['mcPiotski'].tolist()), max(input_df["mcPiotski"].tolist())])
#first_n_companies = input_df.head(num_company);
#input_df2 = input_df[input_df['mcPiotski']>=mcPiotski_selected]
#st.dataframe(input_df[input_df['mcPiotski']>mcPiotski_selected])

#input_df['mcPassPrec'] = input_df['mcPassPrec'].apply(pd.to_numeric)
#input_df['mcPassPrec'] = input_df['mcPassPrec'].astype('Int64')
#input_df['mcPassPrec'] = input_df['mcPassPrec'].astype(int)
mcPassPrec_selected = st.sidebar.slider('Filter mcPassPrec in range:', min(input_df['mcPassPrec'].tolist()), max(input_df["mcPassPrec"].tolist()), value = [min(input_df['mcPassPrec'].tolist()), max(input_df["mcPassPrec"].tolist())])
#first_n_companies = input_df.head(num_company);
#input_df2 = input_df[input_df['mcPassPrec']>=mcPassPrec_selected]
#st.dataframe(input_df[input_df['mcPassPrec']>mcPassPrec_selected])


if st.sidebar.button('Filter with given parameters'):
    st.header('Filtered mcStrength, mcPiotski,  mcPassPrec:')
    input_df2 = input_df[input_df['Mar Cap Rs.Cr.']>=marketCap_selected_min]
    input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']<=marketCap_selected_max]
    input_df2 = input_df2[input_df2['PEG']>=peg_selected_max]
    input_df2 = input_df2[input_df2['PEG']<=peg_selected_min]
    input_df2 = input_df2[input_df['mcStrength']>=mcStrength_selected[0]]
    input_df2 = input_df2[input_df2['mcStrength']<=mcStrength_selected[1]]
    input_df2 = input_df2[input_df2['mcPassPrec']>=mcPassPrec_selected[0]]
    input_df2 = input_df2[input_df2['mcPassPrec']<=mcPassPrec_selected[1]]
    input_df2 = input_df2[input_df2['mcPiotski']>=mcPiotski_selected[0]]
    input_df2 = input_df2[input_df2['mcPiotski']<=mcPiotski_selected[1]]
    # shift column 'Name' to first position
    # insert column using insert(position,column_name,
    # first_column) function
#    if LARGECAP is True:
#        input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']>20000]
#    if MIDCAP is True:
#        input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']>5000]
#        input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']<20000]
#    if SMALLCAP is True:
#        input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']>2000]
#        input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']<5000]
#    if MICROCAP is True:
#        input_df2 = input_df2[input_df2['Mar Cap Rs.Cr.']<2000]

    first_column = input_df2.pop('mcStrength')
    input_df2.insert(4, 'mcStrength', first_column)
    first_column = input_df2.pop('mcPassPrec')
    input_df2.insert(5, 'mcPassPrec', first_column)
    first_column = input_df2.pop('mcPiotski')
    input_df2.insert(6, 'mcPiotski', first_column)
    # In this case default index is exist 
    input_df2.reset_index(inplace = True)
    st.header('{}/{} companies met the search criteria:'.format(len(input_df2),len(input_df)))
    st.dataframe(input_df2)
    #for each_selected_company in list(df_selected_company_names.Name): #[:num_company]:
    #    st.header('Stock Balance Sheets {0}'.format(each_selected_company))
    #    write_excel_data()

company_names = sorted( input_df['Name'])
selected_company_names = st.sidebar.multiselect('Select Name of company', company_names, company_names[0])
	
df_selected_company_names = input_df.loc[((input_df['Name'].isin(selected_company_names)))]


if st.sidebar.button('Display Companies in Selected Sector'):
    st.header('Display paramters of Selected Sectors')
    st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
    st.dataframe(df_selected_company_names)

#if st.button('Show Balance sheets of selected Stocks') and False:
#    for each_selected_company in list(df_selected_company_names.Name): #[:num_company]:
#	    st.header('Stock Balance Sheets {0}'.format(each_selected_company))
#	    display_balance_sheet(each_selected_company)


if st.button('Scrap data:'):
    cd.runChromeDriver()


