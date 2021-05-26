import re as re

#This function parses digits from given string and returns the parsed number as string
def numberParserFromString(input_text):
    num = re.findall(r'[0-9]+',input_text)
    return " ".join(num)


def removeCharactersOtherThanDigits(input_df, listOfcolumnsHeaders):
    for eachColumnOfDataFrame in listOfcolumnsHeaders:
        input_df[eachColumnOfDataFrame]=input_df[eachColumnOfDataFrame].apply(lambda eachValueOfPandasColumn: numberParserFromString(eachValueOfPandasColumn))
    return input_df

#Remove commas from the given dataframe	
def removeCommasFromDataFrame(input_df):
    input_df.replace(',','', regex=True, inplace=True)

#Fill nan with zero
def fillZeroForNanValuesInDataFrame(input_df):
    input_df=input_df.fillna(0)
    return input_df

#BasicDataFormatting Like removing commas and filling Nan values to zero in dataframe
def dataFrameFormatting(input_df):
    removeCommasFromDataFrame(input_df)
    input_df=fillZeroForNanValuesInDataFrame(input_df)
    return input_df

#Convert all the column values to integer in dataframe
def convertEachColumnToInteger(input_df):
    #for each_column in input_df.columns:
    try:
        #print(df[each_column])
        #df[each_column] = pd.to_numeric(df[each_column])#astype(int)
        df[[input_df.columns]] = df[[input_df.columns]].apply(pd.to_numeric)
        print("Converted the column to integer")#.format(each_column))
    except:
        print("Cannot convert..")#format(each_column))
        #continue
    #input_df['Mar Cap Rs.Cr.'] = input_df['Mar Cap Rs.Cr.'].astype(int)
    #input_df['CMP Rs.'] = input_df['CMP Rs.'].astype(int)
    return input_df


