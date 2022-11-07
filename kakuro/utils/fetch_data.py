import argparse
import base64
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By


class Constants:
    """
    Some constants are being initialised
    """
    GENERATE_BUTTON_XPATH = '/html/body/div/div[2]/div[1]/input[8]'
    CANVAS_XPATH = '/html/body/div/div[2]/div[4]/canvas'
    SLEEP_TIME = 3
    DIFFICULTY_XPATHS = {
        'easy': '/html/body/div/div[2]/div[1]/input[3]',
        'medium': '/html/body/div/div[2]/div[1]/input[4]',
        'hard': '/html/body/div/div[2]/div[1]/input[5]'
    }


class FetchDataError(Exception):
    '''
    A custom error class to track errors raised in the process of fetching the data
    '''
    pass


def get_driver():
    """
    Initialises a webdriver object for chrome
    :return: webdriver
    """
    driver = webdriver.Chrome()
    return driver


def generate_data(driver: webdriver, difficulty: str, number_of_instances: int, folder: str):
    """
    This function orchestrates the procedure of generating the data. The procedure can be summarised as follows:
        1. Iterate the data generation loop for the number of instances specified.
        2. Generate the button to generate a new puzzle
        3. Wait for a few seconds
        4. Take a screenshot as base64 and save as PNG.
    :param driver: The webdriver object to communicate with tfile_pathhe website and generate data
    :param number_of_instances: Number of data points which are to be generated by the function
    :param folder: the folder which should contain data after generation
    :return: None
    """
    # fetch the page
    driver.get('https://www.kakuro-online.com/generator')

    # set the difficulty of the puzzle
    difficulty_radio_button = driver.find_element(
        By.XPATH, Constants.DIFFICULTY_XPATHS[difficulty]
    )
    difficulty_radio_button.click()

    if not os.path.exists(folder):
        os.mkdir(folder)

    for idx in range(number_of_instances):
        # get the button to generate a new puzzle
        generate_button = driver.find_element(By.XPATH, Constants.GENERATE_BUTTON_XPATH)
        generate_button.click()

        # Giving the website time to load up the puzzle
        time.sleep(Constants.SLEEP_TIME)

        # XPath of the canvas containing the puzzle
        canvas = driver.find_element(By.XPATH, Constants.CANVAS_XPATH)
        screenshot = canvas.screenshot_as_base64

        # decode and write the contents to the file
        decoded = base64.b64decode(screenshot)
        dest_path = os.path.join(folder, f'puzzle_{idx}.png')
        with open(dest_path, 'wb') as f:
            f.write(decoded)

        if (idx + 1) % 10 == 0:
            print(f'{idx} instances generated')


def close_driver(driver: webdriver):
    """
    This function closes the driver which has been opened by the get_driver function.
    :param driver: A selenium webdriver
    :return: None
    """
    driver.close()


# now begins the main part
parser = argparse.ArgumentParser()
basic = parser.add_argument_group('basic')
basic.add_argument('-c', '--count', help='Number of data points which are to be generated', type=int)
basic.add_argument('-d', '--difficulty', help='Difficulty of the problem to be extracted', default='easy')

split = parser.add_argument_group('split')
split.add_argument('-e', '--easy', help='Number of easy samples to fetch', default=0, type=int)
split.add_argument('-m', '--medium', help='Number of medium samples to fetch', default=0, type=int)
split.add_argument('-a', '--hard', help='Number of hard samples to fetch', default=0, type=int)

# parse the arguments supplied
args = parser.parse_args()
data_folder = '../data/puzzles/'
webdriver = get_driver()

if args.count:
    print('basic section')
    generate_data(webdriver, args.difficulty, args.count, data_folder)
elif args.easy or args.medium or args.hard:
    print('split section')
    generate_data(webdriver, 'easy', args.easy, os.path.join(data_folder, args.difficulty))
    generate_data(webdriver, 'medium', args.medium, data_folder)
    generate_data(webdriver, 'hard', args.hard, data_folder)
else:
    print('Argument count has to be supplied')
    raise FetchDataError

close_driver(webdriver)