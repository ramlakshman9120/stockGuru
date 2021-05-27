from selenium import webdriver
from selenium.webdriver.common.keys import Keys
#from selenium.common.exception import TimeoutException
from selenium.webdriver.common.by import By
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
import os
import time
import streamlit as st

def scrapeMC(name,driver):

    print("Company-", name)
    try:
        for t in str(name):
            # Enters the company name in the search box
#             print(t)
            time.sleep(0.2)
            driver.find_element_by_xpath('//*[@id="search_str"]').send_keys(t)
            time.sleep(0.4)#randint(0.1,0.3))
            
    except:
        time.sleep(1)
        print("Failed with error, retrying...",name)
        time.sleep(1)
        driver.get('https://www.moneycontrol.com/')
        scrapeMC(name,driver)
    else:# Success case
        time.sleep(0.1)

        try:
            time.sleep(0.3)#randint(0.1,0.3))
            # Select the first option from the search box
            driver.find_element_by_xpath('//*[@id="autosuggestlist"]/ul/li[1]/a').click()
        except:
            time.sleep(1)
            print("Failed with error1, retrying...",name)
            time.sleep(1)
            scrapeMC(name,driver)
        else:#Success case
            time.sleep(0.5)

            td=driver.find_element_by_xpath('//*[@id="mcessential_div"]/div/div[2]/div')
            
            print(td.text)
            return td.text

            time.sleep(5)
            driver.close()

            st.title('APP1')


def runChromeDriver(): 
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    chromedriver = dir_path + "/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chrome_options=options, executable_path=chromedriver)
    
    driver.get('https://www.moneycontrol.com/')
    name="532805"
    scrapeMC(name,driver)



# Selenium Exception handling

#driver = webdriver.Chrome(executable_path=r"C:\Users\srirama.g\Desktop\chromedriver.exe")

#if _name=="main_":

