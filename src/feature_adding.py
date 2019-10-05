import pandas as pd
import math
from nltk.corpus import stopwords

#  load excel preprocessed dataset
play_raw_data = pd.read_csv("../data/processed/excel_preprocessed.csv")

'''
Adding features:
1. whether PlayerLine contains exclamation mark
2. whether PlayerLine contains question mark
3. number of parts in PlayerLine splitted by comma 
4. number of stop words in PlayerLine
5. number of upper case letters in PlayerLine
'''
contain_exclm_mark = []
contain_question_mark = []
num_comma_split = []
stop_words_count = []
stop_words = list(set(stopwords.words('english')))
SPECIAL_MARKS = ['?', ',', '!']
upper_case_count = []
INVALID_PLAYERLINES = ['nan', '', " "]

def remove_special_marks(text):
    result = text
    for mark in SPECIAL_MARKS:
        result = text.replace(mark, '')
    return result


for i, row in play_raw_data.iterrows():
    player_line = row.PlayerLine
    if str(player_line) in INVALID_PLAYERLINES:
        contain_exclm_mark.append(False)
        contain_question_mark.append(False)
        num_comma_split.append(0)
        stop_words_count.append(0)
        upper_case_count.append(0)
        play_raw_data.set_value(i, 'PL_length', 0)

    else:
        contain_exclm_mark.append('!' in player_line)
        contain_question_mark.append('?' in player_line)
        num_comma_split.append(len(player_line.split(',')))
        stop_words_count.append(len([wrd for wrd in remove_special_marks(player_line).split(' ') if wrd.lower() in stop_words]))
        upper_case_count.append(len([letter for letter in player_line if letter.isupper()]))


play_raw_data['PL_contain_!'] = pd.Series(contain_exclm_mark)
play_raw_data['PL_contain_?'] = pd.Series(contain_question_mark)
play_raw_data['PL_#_comma_split'] = pd.Series(num_comma_split)
play_raw_data['PL_#_stop_words'] = pd.Series(stop_words_count)
play_raw_data['PL_#_upper_case'] = pd.Series(upper_case_count)


play_raw_data.to_csv(index=False, path_or_buf="../data/processed/processed.csv")
print 'Adding features completed.'
