import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
import json

# reformat from 1d list to 2d list of lists
# https://stackoverflow.com/questions/14681609/create-a-2d-list-out-of-1d-list
def to_matrix_list(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

# get Robinhood data
def getRobhinhooldData():
    with open('RobinhoodMaunalPull.txt') as f:
        lines = [x.rstrip('\n') for x in f] # read data into list and remove \n from the end of each value
    list_of_lists = to_matrix_list(lines, 7) # reformat from 1d list to 2d list of lists
    npdata = np.array(list_of_lists) # easier to slice as numpy
    df = pd.DataFrame(npdata[1:,:], columns = npdata[0,:]) # convert to pandas, first row is the header
    return df

# Description:  This function will get 5 years of closing prices for a single ticker.
# Resources:
# - https://medium.com/@kasperjuunge/yfinance-10-ways-to-get-stock-data-with-python-6677f49e8282
# - https://stackoverflow.com/questions/20461165/how-to-convert-index-of-a-pandas-dataframe-into-a-column
def get_stock_data_given_tckrs(tckrs, more_than_one=False):
    if more_than_one==False: # number of tickers passed = 1
        recent_data = yf.download(tckrs, period="5y") # grab 5 years of data
        recent_data = recent_data.reset_index() # convernt index to column
        recent_data['Date'] = recent_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d')) # convert timestamp to string
        recent_data = recent_data[['Date','Close']] # keep date and close
    else: # number of tickers passed > 1
        recent_data = [] # supporting >1 ticker at once is possible, but not yet built out
    return recent_data.values.tolist()

# Description:  Open JSON file containing stock data
# https://www.geeksforgeeks.org/read-json-file-using-python/
def load_stock_data():
    try:
        # Opening JSON file
        with open('stock_data.json') as f:
            # returns JSON object as a dictionary
            data = json.load(f)
    except:
        data = {}
    return data

# Description:  Pull stock data and save to txt file.
def pull_stock_data():
    stock_results = load_stock_data() # get latest stock file results, if none exist then initialize dictionary for new results
    df_all_stcks = pd.read_csv('full_stock_tickers_list_061817_6k.csv') # get list of all stock tickers
    for index, row in df_all_stcks.iterrows(): # loop through all stock tickers
        # if ticker is in latest stock results, check if the data is > 7 days old in which case pull new data and overwrite
        if row['Symbol'] in list(stock_results.keys()):
            latest_pull_date = stock_results[row['Symbol']]['PullDate']
            d = datetime.strptime(latest_pull_date, "%Y-%m-%d") # convert pull date to datetime
            now = datetime.now() # get current datetime
            delta = d - now # get datetime delta
            # print(type(stock_results), row['Symbol'], latest_pull_date, d, now, delta, delta.days) ### DEBUGGING
            # break ### DEBUGGING
            if delta.days > 7: # if the data is > 7 days old, then pull new data and overwrite
                stock_results[row['Symbol']] = {
                    'Data': get_stock_data_given_tckrs(row['Symbol']),
                    'PullDate': date.today().strftime('%Y-%m-%d')
                }
        else: # if ticker is NOT in latest stock results, then pull ticker data and add to dictionary

            # print("HI!!!", row['Symbol'], type(stock_results), list(stock_results.keys())) ### DEBUGGING
            stock_results[row['Symbol']] = {
                'Data': get_stock_data_given_tckrs(row['Symbol']),
                'PullDate': date.today().strftime('%Y-%m-%d')
            }
        # if index==12: ### DEBUGGING
        #     break ### DEBUGGING
    # write results out
    with open('stock_data.json', 'w') as results_file:
        results_file.write(json.dumps(stock_results))
    # function complete


if __name__ == '__main__': 
    # print(getRobhinhooldData().head())
    # print(get_stock_data_given_tckrs("AAPL"))
    pull_stock_data()