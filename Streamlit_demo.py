import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import numberParserFromString as uf

st.title('STOCK-GURU GOVIND')

st.markdown("""
This APP reads the stock data from the browsed XLSX file
And displays the output based inputs provided from the custom filters
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** Screener, MoneyControl).
""")

st.sidebar.header('Hey..Welcome back!!')
 
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Please Browse stock XLSX file", type=["xlsx"])

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file, sheet_name="Peer Comparision")
    
    input_df = uf.removeCharactersOtherThanDigits(input_df, ['mcStrength','mcPassPrec'])
    input_df = uf.dataFrameFormatting(input_df)

    #Convert all the values to integers
    #input_df['Mar Cap Rs.Cr.'] = input_df['Mar Cap Rs.Cr.'].astype(int)
    #input_df['CMP Rs.'] = input_df['CMP Rs.'].astype(int)
    input_df = uf.convertEachColumnToInteger(input_df)
    input_df[['mcStrength', 'mcStrength', 'mcPiotski']] = input_df[['mcStrength', 'mcStrength', 'mcPiotski']].apply(pd.to_numeric)
    
    
    st.header('Display Companies based on range of companies')
    num_company = st.sidebar.slider('Number of Companies', 0, len(input_df))
    #first_n_companies = input_df.head(num_company);
    st.dataframe(input_df.head(num_company))
