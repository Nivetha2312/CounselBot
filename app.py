import numpy as np
import pandas as pd
#import preprocessor as p
import counselor
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from PIL import Image
import streamlit as st
import imagify
from bokeh.plotting import figure, output_file, show
import math
from bokeh.palettes import Greens
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource
#ap = Path.joinpath(Path.cwd(), 'models')
#dsp = Path.joinpath(Path.cwd(), 'dataset')

#model = load_model(Path.joinpath(artifacts_path, 'botmodel.h5'))
#tok = joblib.load(Path.joinpath(artifacts_path, 'tokenizer_t.pkl'))
#words = joblib.load(Path.joinpath(artifacts_path, 'words.pkl'))
#df2 = pd.read_csv(Path.joinpath(datasets_path, 'bot.csv'))

model = load_model('botmodel.h5')
tok = joblib.load('tokenizer_t.pkl')
words = joblib.load('words.pkl')
df2 = pd.read_csv('bot.csv')
flag=1

import string
import re
import json
import nltk
#run on the first time alone :
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():

    lem = WordNetLemmatizer()
    n=1
    def tokenizer(x):
        tokens = x.split()
        rep = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [rep.sub('', i) for i in tokens]
        tokens = [i for i in tokens if i.isalpha()]
        tokens = [lem.lemmatize(i.lower()) for i in tokens]
        tokens = [i.lower() for i in tokens if len(i) > 1]
        return tokens

    def no_stop_inp(tokenizer,df,c):
        no_stop = []
        x = df[c][0]
        tokens = tokenizer(x)
        no_stop.append(' '.join(tokens))
        df[c] = no_stop
        return df

    def inpenc(tok,df,c):
        t = tok
        x = x = [df[c][0]]
        enc = t.texts_to_sequences(x)
        padded = pad_sequences(enc, maxlen=16, padding='post')
        return padded

    def predinp(model,x):
        pred = np.argmax(model.predict(x))
        return pred

    def botp(df3,pred):
        l = df3.user[0].split()
        if len([i for i in l if i in words])==0 :
            pred = 1
        return pred

    def botop(df2,pred):
        x2 = df2.groupby('labels').get_group(pred).shape[0]
        idx1 = np.random.randint(0,x2)
        op = list(df2.groupby('labels').get_group(pred).bot)
        return op[idx1]

    def botans(df3):
        tok = joblib.load('tokenizer_t.pkl')
        word = joblib.load('words.pkl')
        df3 = no_stop_inp(tokenizer, df3, 'user')
        inp = inpenc(tok, df3, 'user')
        pred = predinp(model, inp)
        pred = botp(df3, pred)
        ans = botop(df2, pred)
        return ans

    def get_text():
        x = st.text_input("You : ")
        x=x.lower()
        xx = x[:13]
        if(xx =="start my test"):
            global flag
            flag=0
        input_text  = [x]
        df_input = pd.DataFrame(input_text,columns=['user'])
        return df_input

    #flag=1
    qvals = {"Select an Option": 0, "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2,
             "Strongly Disagree": 1}
    st.title("CounselBot")
    banner=Image.open("img/21.png")
    st.image(banner, use_column_width=True)
    st.write("Hi! I'm CounselBot, your personal career counseling bot. Ask your queries in the text box below and hit enter. If and when you are ready to take our personality test, type \"start my test\" and you're good to go!")

    df3 = get_text()
    if (df3.loc[0, 'user']==""):
        ans = "Hi, I'm CounselBot. \nHow can I help you?"

    elif(flag==0):
        #st.write(flag)
        ans = "Sure, good luck!"
    else:
        ans = botans(df3)

    st.text_area("CounselBot:", value=ans, height=100, max_chars=None)

    if(flag==0):
        #x=start_test()
        #st.text_area("confirm", value="starting test", height=100, max_chars=None)
        st.title("PERSONALITY TEST:")
        #st.write("Would you like to begin with the test?")
        kr = st.selectbox("Would you like to begin with the test?", ["Select an Option", "Yes", "No"])
        if (kr == "Yes"):
            kr1 = st.selectbox("Select level of education",
                               ["Select an Option", "Grade 10", "Grade 12", "Undergraduate"])

            #####################################  GRADE 10  ###########################################

            if(kr1=="Grade 10"):
                lis = []
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("I find writing programs for computer applications interesting")
                    n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        st.header("Question 2")
                        st.write("I can understand mathematical problems with ease")
                        n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            st.header("Question 3")
                            st.write("Learning about the existence of individual chemical components is interesting")
                            n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                st.header("Question 4")
                                st.write("The way plants and animals thrive gets me curious")
                                n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    st.header("Question 5")
                                    st.write("Studying about the way fundamental constituents of the universe interact with each other is fascinating")
                                    n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        st.header("Question 6")
                                        st.write(
                                            "Accounting and business management is my cup of tea")
                                        n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            st.header("Question 7")
                                            st.write(
                                                "I would like to know more about human behaviour, relations and patterns of thinking")
                                            n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                st.header("Question 8")
                                                st.write(
                                                    "I find the need to be aware of stories from the past.")
                                                n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    st.header("Question 9")
                                                    st.write(
                                                        "I see myself as a sportsperson/professional trainer")
                                                    n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        st.header("Question 10")
                                                        st.write(
                                                            "I enjoy creating works of art")
                                                        n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.success("Test Completed")
                                                            #st.write(lis)
                                                            st.title("RESULTS:")
                                                            df = pd.read_csv(r"Subjects.csv")

                                                            input_list = lis

                                                            subjects = {1: "Computers",
                                                                        2: "Mathematics",
                                                                        3: "Chemistry",
                                                                        4: "Biology",
                                                                        5: "Physics",
                                                                        6: "Commerce",
                                                                        7: "Psychology",
                                                                        8: "History",
                                                                        9: "Physical Education",
                                                                        10: "Design"}

                                                            def output(listofanswers):
                                                                class my_dictionary(dict):
                                                                    def __init__(self):
                                                                        self = dict()

                                                                    def add(self, key, value):
                                                                        self[key] = value

                                                                ques = my_dictionary()

                                                                for i in range(0, 10):
                                                                    ques.add(i, input_list[i])

                                                                all_scores = []

                                                                for i in range(9):
                                                                    all_scores.append(ques[i] / 5)

                                                                li = []

                                                                for i in range(len(all_scores)):
                                                                    li.append([all_scores[i], i])
                                                                li.sort(reverse=True)
                                                                sort_index = []
                                                                for x in li:
                                                                    sort_index.append(x[1] + 1)
                                                                all_scores.sort(reverse=True)

                                                                a = sort_index[0:5]
                                                                b = all_scores[0:5]
                                                                s = sum(b)
                                                                d = list(map(lambda x: x * (100 / s), b))

                                                                return a, d

                                                            l, data = output(input_list)

                                                            out = []
                                                            for i in range(0, 5):
                                                                n = l[i]
                                                                c = subjects[n]
                                                                out.append(c)

                                                            output_file("pie.html")

                                                            graph = figure(title="Recommended subjects", height=500,
                                                                           width=500)
                                                            radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                            start_angle = [math.radians(0)]
                                                            prev = start_angle[0]
                                                            for i in radians[:-1]:
                                                                start_angle.append(i + prev)
                                                                prev = i + prev

                                                            end_angle = start_angle[1:] + [math.radians(0)]

                                                            x = 0
                                                            y = 0

                                                            radius = 0.8

                                                            color = Greens[len(out)]
                                                            graph.xgrid.visible = False
                                                            graph.ygrid.visible = False
                                                            graph.xaxis.visible = False
                                                            graph.yaxis.visible = False

                                                            for i in range(len(out)):
                                                                graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                            graph.add_layout(graph.legend[0], 'right')
                                                            st.bokeh_chart(graph, use_container_width=True)
                                                            labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                            graph.add_layout(labels)

                                                            st.header('More information on the subjects')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(subjects[int(l[i])])
                                                                st.write(df['about'][int(l[i]) - 1])

                                                            st.header('Choice of Degrees')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(subjects[int(l[i])])
                                                                st.write(df['further career'][int(l[i]) - 1])

                                                            st.header('Trends over the years')
                                                            # We'll be using a csv file for that
                                                           

                                                            def Convert(string):
                                                                li = list(string.split(","))
                                                                li = list(map(float, li))
                                                                return li

                                                            x = ['2000', '2005', '2010', '2015', '2020']
                                                            y = []
                                                            for i in range(0, 5):
                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                y.append(t)
                                                            output_file("line.html")
                                                            graph2 = figure(title="Trends")

                                                            graph2.line(x, y[0], line_color="Purple",
                                                                        legend_label=out[0])
                                                            graph2.line(x, y[1], line_color="Blue",
                                                                        legend_label=out[1])
                                                            graph2.line(x, y[2], line_color="Green",
                                                                        legend_label=out[2])
                                                            graph2.line(x, y[3], line_color="Magenta",
                                                                        legend_label=out[3])
                                                            graph2.line(x, y[4], line_color="Red",
                                                                        legend_label=out[4])

                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                            st.bokeh_chart(graph2, use_container_width=True)
                                                            banner1 = Image.open("img/coun.png")
                                                            st.image(banner1, use_column_width=True)
                                                            st.header("Contacts of experts from various fields")

                                                            for i in range(0, 5):
                                                                st.subheader(subjects[int(l[i])])
                                                                xl=(df['contacts'][int(l[i]) - 1]).split(",")
                                                                for k in xl:
                                                                    ml=list(k.split(","))
                                                                    for kk in ml:
                                                                        st.write(kk,sep="\n")





        ##########################################  GRADE 12  ########################################################

            elif (kr1 == "Grade 12"):
                lis = []
                st.header("Question 1")
                st.write("I enjoy debating and negotiating issues in public")
                n = imagify.imageify(n)
                inp = st.selectbox("",
                                   ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                    "Strongly Disagree"],
                                   key='1')
                if ((inp != "Select an Option")):
                    lis.append(qvals[inp])
                    st.header("Question 2")
                    st.write("Studying the anatomy of the human body and giving first aid to people is something I'm always looking forward to")
                    n = imagify.imageify(n)
                    inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                             "Strongly Disagree"], key='2')

                    if (inp2 != "Select an Option"):
                        lis.append(qvals[inp2])
                        st.header("Question 3")
                        st.write("I can lead a team and easily manage projects")
                        n = imagify.imageify(n)
                        inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='3')
                        if (inp3 != "Select an Option"):
                            lis.append(qvals[inp3])
                            st.header("Question 4")
                            st.write("Working with tools, equipment, and machinery is enjoyable")
                            n = imagify.imageify(n)
                            inp4 = st.selectbox("",
                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='4')
                            if (inp4 != "Select an Option"):
                                lis.append(qvals[inp4])
                                st.header("Question 5")
                                st.write(
                                    "Budgeting, costing and estimating for a business isn't exhausting")
                                n = imagify.imageify(n)
                                inp5 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                     "Disagree",
                                                     "Strongly Disagree"], key='5')
                                if (inp5 != "Select an Option"):
                                    lis.append(qvals[inp5])
                                    st.header("Question 6")
                                    st.write(
                                        "I can see myself taking part in competitive sporting events to become a professional")
                                    n = imagify.imageify(n)
                                    inp6 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='6')
                                    if (inp6 != "Select an Option"):
                                        lis.append(qvals[inp6])
                                        st.header("Question 7")
                                        st.write(
                                            "I don't burn out while doing translations, reading and correcting language")
                                        n = imagify.imageify(n)
                                        inp7 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='7')
                                        if (inp7 != "Select an Option"):
                                            lis.append(qvals[inp7])
                                            st.header("Question 8")
                                            st.write(
                                                "I would love to act in or direct a play or film")
                                            n = imagify.imageify(n)
                                            inp8 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree",
                                                                 "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='8')
                                            if (inp8 != "Select an Option"):
                                                lis.append(qvals[inp8])
                                                st.header("Question 9")
                                                st.write(
                                                    "Making sketches of people or landscapes is a hobby I see as a career")
                                                n = imagify.imageify(n)
                                                inp9 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='9')
                                                if (inp9 != "Select an Option"):
                                                    lis.append(qvals[inp9])
                                                    st.header("Question 10")
                                                    st.write(
                                                        "I can easily work with numbers and calculations most of the time")
                                                    n = imagify.imageify(n)
                                                    inp10 = st.selectbox("",
                                                                         ["Select an Option", "Strongly Agree", "Agree",
                                                                          "Neutral",
                                                                          "Disagree",
                                                                          "Strongly Disagree"], key='10')
                                                    if (inp10 != "Select an Option"):
                                                        lis.append(qvals[inp10])
                                                        st.header("Question 11")
                                                        st.write(
                                                            "I enjoy doing clerical work i.e. filing, counting stock and issuing receipts")
                                                        n = imagify.imageify(n)
                                                        inp11 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree",
                                                                              "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='11')
                                                        if (inp11 != "Select an Option"):
                                                            lis.append(qvals[inp11])
                                                            st.header("Question 12")
                                                            st.write(
                                                                "I love studying the culture and life style of human societies")
                                                            n = imagify.imageify(n)
                                                            inp12 = st.selectbox("",
                                                                                 ["Select an Option", "Strongly Agree",
                                                                                  "Agree",
                                                                                  "Neutral",
                                                                                  "Disagree",
                                                                                  "Strongly Disagree"], key='12')
                                                            if (inp12 != "Select an Option"):
                                                                lis.append(qvals[inp12])
                                                                st.header("Question 13")
                                                                st.write(
                                                                    "Teaching children and young people is something I see myself doing on a daily basis")
                                                                n = imagify.imageify(n)
                                                                inp13 = st.selectbox("",
                                                                                     ["Select an Option",
                                                                                      "Strongly Agree", "Agree",
                                                                                      "Neutral",
                                                                                      "Disagree",
                                                                                      "Strongly Disagree"], key='13')
                                                                if (inp13 != "Select an Option"):
                                                                    lis.append(qvals[inp13])
                                                                    st.header("Question 14")
                                                                    st.write(
                                                                        "I won't have a problem persevering in the army or police force")
                                                                    n = imagify.imageify(n)
                                                                    inp14 = st.selectbox("",
                                                                                         ["Select an Option",
                                                                                          "Strongly Agree", "Agree",
                                                                                          "Neutral",
                                                                                          "Disagree",
                                                                                          "Strongly Disagree"],
                                                                                         key='14')
                                                                    if (inp14 != "Select an Option"):
                                                                        lis.append(qvals[inp14])
                                                                        st.header("Question 15")
                                                                        st.write(
                                                                            "Introducing consumers to new products and convincing them to buy the same is something that comes with ease")
                                                                        n = imagify.imageify(n)
                                                                        inp15 = st.selectbox("",
                                                                                             ["Select an Option",
                                                                                              "Strongly Agree", "Agree",
                                                                                              "Neutral",
                                                                                              "Disagree",
                                                                                              "Strongly Disagree"],
                                                                                             key='15')
                                                                        if (inp15 != "Select an Option"):
                                                                            lis.append(qvals[inp10])
                                                                            st.success("Test Completed")
                                                                            #st.write(lis)
                                                                            st.title("RESULTS:")
                                                                            df = pd.read_csv(r"Graduate.csv")

                                                                            input_list = lis

                                                                            streams = {1: "Law",
                                                                                       2: "Healthcare",
                                                                                       3: "Management",
                                                                                       4: "Engineering",
                                                                                       5: "Finance",
                                                                                       6: "Sports",
                                                                                       7: "Language and communication",
                                                                                       8: "Performing Arts",
                                                                                       9: "Applied and Visual arts",
                                                                                       10: "Science and math",
                                                                                       11: "Clerical and secretarial",
                                                                                       12: "Social Science",
                                                                                       13: "Education and Social Support",
                                                                                       14: "Armed Forces",
                                                                                       15: "Marketing and sales"}

                                                                            def output(listofanswers):
                                                                                class my_dictionary(dict):
                                                                                    def __init__(self):
                                                                                        self = dict()

                                                                                    def add(self, key, value):
                                                                                        self[key] = value

                                                                                ques = my_dictionary()

                                                                                for i in range(0, 15):
                                                                                    ques.add(i, input_list[i])

                                                                                all_scores = []

                                                                                for i in range(14):
                                                                                    all_scores.append(ques[i] / 5)

                                                                                li = []

                                                                                for i in range(len(all_scores)):
                                                                                    li.append([all_scores[i], i])
                                                                                li.sort(reverse=True)
                                                                                sort_index = []
                                                                                for x in li:
                                                                                    sort_index.append(x[1] + 1)
                                                                                all_scores.sort(reverse=True)

                                                                                a = sort_index[0:5]
                                                                                b = all_scores[0:5]
                                                                                s = sum(b)
                                                                                d = list(
                                                                                    map(lambda x: x * (100 / s), b))

                                                                                return a, d

                                                                            l, data = output(input_list)

                                                                            out = []
                                                                            for i in range(0, 5):
                                                                                n = l[i]
                                                                                c = streams[n]
                                                                                out.append(c)

                                                                            output_file("pie.html")

                                                                            graph = figure(title="Recommended fields",
                                                                                           height=500, width=500)
                                                                            radians = [
                                                                                math.radians((percent / 100) * 360) for
                                                                                percent in data]

                                                                            start_angle = [math.radians(0)]
                                                                            prev = start_angle[0]
                                                                            for i in radians[:-1]:
                                                                                start_angle.append(i + prev)
                                                                                prev = i + prev

                                                                            end_angle = start_angle[1:] + [
                                                                                math.radians(0)]

                                                                            x = 0
                                                                            y = 0

                                                                            radius = 0.8

                                                                            color = Greens[len(out)]
                                                                            graph.xgrid.visible = False
                                                                            graph.ygrid.visible = False
                                                                            graph.xaxis.visible = False
                                                                            graph.yaxis.visible = False

                                                                            for i in range(len(out)):
                                                                                graph.wedge(x, y, radius,
                                                                                            start_angle=start_angle[i],
                                                                                            end_angle=end_angle[i],
                                                                                            color=color[i],
                                                                                            legend_label=out[
                                                                                                             i] + "-" + str(
                                                                                                round(data[i])) + "%")

                                                                            graph.add_layout(graph.legend[0],
                                                                                                'right')
                                                                            st.bokeh_chart(graph,
                                                                                            use_container_width=True)
                                                                            labels = LabelSet(x='text_pos_x',
                                                                                                y='text_pos_y',
                                                                                                text='percentage',
                                                                                                level='glyph',
                                                                                                angle=0,
                                                                                                render_mode='canvas')
                                                                            graph.add_layout(labels)

                                                                            st.header(
                                                                                'More information on the fields')
                                                                            # We'll be using a csv file for that
                                                                            for i in range(0, 5):
                                                                                st.subheader(streams[int(l[i])])
                                                                                st.write(df['About'][int(l[i]) - 1])

                                                                            st.header('Average annual salary')
                                                                            # We'll be using a csv file for that
                                                                            for i in range(0, 5):
                                                                                st.subheader(streams[int(l[i])])
                                                                                st.write("Rs. "+ str(
                                                                                    df['avgsal'][int(l[i]) - 1]))

                                                                            st.header('Trends over the years')
                                                                            # We'll be using a csv file for that
                                                                            

                                                                            def Convert(string):
                                                                                li = list(string.split(","))
                                                                                li = list(map(float, li))
                                                                                return li

                                                                            x = ['2000', '2005', '2010', '2015', '2020']
                                                                            y = []
                                                                            for i in range(0, 5):
                                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                                y.append(t)
                                                                            output_file("line.html")
                                                                            graph2 = figure(title="Trends")

                                                                            graph2.line(x, y[0], line_color="Purple",
                                                                                        legend_label=out[0])
                                                                            graph2.line(x, y[1], line_color="Blue",
                                                                                        legend_label=out[1])
                                                                            graph2.line(x, y[2], line_color="Green",
                                                                                        legend_label=out[2])
                                                                            graph2.line(x, y[3], line_color="Magenta",
                                                                                        legend_label=out[3])
                                                                            graph2.line(x, y[4], line_color="Red",
                                                                                        legend_label=out[4])

                                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                                            st.bokeh_chart(graph2,
                                                                                           use_container_width=True)

                                                                            banner1 = Image.open("img/coun.png")
                                                                            st.image(banner1, use_column_width=True)
                                                                            st.header(
                                                                                "Contacts of experts from various fields")


                                                                            for i in range(0, 5):
                                                                                st.subheader(streams[int(l[i])])
                                                                                xl = (
                                                                                df['contacts'][int(l[i]) - 1]).split(
                                                                                    ",")
                                                                                for k in xl:
                                                                                    ml = list(k.split(","))
                                                                                    for kk in ml:
                                                                                        st.write(kk, sep="\n")




            ######################################  UNDERGRADUATE ##########################################

            elif (kr1 == "Undergraduate"):
                lis = []
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("I can be the person who handles all aspects of information security and protects the virtual data resources of a company")
                    n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        st.header("Question 2")
                        st.write("I enjoy studying business and information requirements of an organisation and using this data to develop processes that help achieve strategic goals.")
                        n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            st.header("Question 3")
                            st.write("I can assess a problem and design a brand new system or improve an existing system to make it better and more efficient. ")
                            n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                st.header("Question 4")
                                st.write("Designing, developing, modifying, editing and working with databases and large datasets is my cup of tea")
                                n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    st.header("Question 5")
                                    st.write(
                                        "I can mine data using BI software tools, compare, visualize and communicate the results with ease")
                                    n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        st.header("Question 6")
                                        st.write(
                                            "Implementing and providing support for Microsoft's Dynamics CRM is a skill I possess")
                                        n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            st.header("Question 7")
                                            st.write(
                                                "I can be innovative and creative when it comes to making user-friendly mobile applications")
                                            n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                st.header("Question 8")
                                                st.write(
                                                    "I can perform well in a varied discipline, combining aspects of psychology, business, market research, design, and technology.")
                                                n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    st.header("Question 9")
                                                    st.write(
                                                        "I am responsible enough to maintain the quality systems, such as laboratory control and document control and training, to ensure control of the manufacturing process.")
                                                    n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        st.header("Question 10")
                                                        st.write(
                                                            "Be it front-end or back-end, I would love designing and developing websites more than anything else")
                                                        n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.success("Test Completed")
                                                            #st.write(lis)

                                                            st.title("RESULTS:")
                                                            df = pd.read_csv(r'Occupations.csv', encoding= 'windows-1252')

                                                            input_list = lis

                                                            professions = {1: "Systems Security Administrator",
                                                                        2: "Business Systems Analyst",
                                                                        3: "Software Systems Engineer",
                                                                        4: "Database Developer",
                                                                        5: "Business Intelligence Analyst",
                                                                        6: "CRM Technical Developer",
                                                                        7: "Mobile Applications Developer",
                                                                        8: "UX Designer",
                                                                        9: "Quality Assurance Associate",
                                                                        10: "Web Developer"}

                                                            def output(listofanswers):
                                                                class my_dictionary(dict):
                                                                    def __init__(self):
                                                                        self = dict()

                                                                    def add(self, key, value):
                                                                        self[key] = value

                                                                ques = my_dictionary()

                                                                for i in range(0, 10):
                                                                    ques.add(i, input_list[i])

                                                                all_scores = []

                                                                for i in range(9):
                                                                    all_scores.append(ques[i] / 5)

                                                                li = []

                                                                for i in range(len(all_scores)):
                                                                    li.append([all_scores[i], i])
                                                                li.sort(reverse=True)
                                                                sort_index = []
                                                                for x in li:
                                                                    sort_index.append(x[1] + 1)
                                                                all_scores.sort(reverse=True)

                                                                a = sort_index[0:5]
                                                                b = all_scores[0:5]
                                                                s = sum(b)
                                                                d = list(map(lambda x: x * (100 / s), b))

                                                                return a, d

                                                            l, data = output(input_list)

                                                            out = []
                                                            for i in range(0, 5):
                                                                n = l[i]
                                                                c = professions[n]
                                                                out.append(c)

                                                            output_file("pie.html")

                                                            graph = figure(title="Recommended professions", height=500,
                                                                           width=500)
                                                            radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                            start_angle = [math.radians(0)]
                                                            prev = start_angle[0]
                                                            for i in radians[:-1]:
                                                                start_angle.append(i + prev)
                                                                prev = i + prev

                                                            end_angle = start_angle[1:] + [math.radians(0)]

                                                            x = 0
                                                            y = 0

                                                            radius = 0.8

                                                            color = Greens[len(out)]
                                                            graph.xgrid.visible = False
                                                            graph.ygrid.visible = False
                                                            graph.xaxis.visible = False
                                                            graph.yaxis.visible = False

                                                            for i in range(len(out)):
                                                                graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                            graph.add_layout(graph.legend[0], 'right')
                                                            st.bokeh_chart(graph, use_container_width=True)
                                                            labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                            graph.add_layout(labels)
                                                            st.header('More information on the professions')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(professions[int(l[i])])
                                                                st.write(df['Information'][int(l[i]) - 1])

                                                            st.header('Monthly Income')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(professions[int(l[i])])
                                                                st.write("Rs. " + str(df['Income'][int(l[i]) - 1]))

                                                            st.header('Trends over the years')
                                                            # We'll be using a csv file for that
                                                        

                                                            def Convert(string):
                                                                li = list(string.split(","))
                                                                li = list(map(float, li))
                                                                return li

                                                            x = ['2000', '2005', '2010', '2015', '2020']
                                                            y = []
                                                            for i in range(0, 5):
                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                y.append(t)
                                                            output_file("line.html")
                                                            graph2 = figure(title="Trends")

                                                            graph2.line(x, y[0], line_color="Purple",
                                                                        legend_label=out[0])
                                                            graph2.line(x, y[1], line_color="Blue", legend_label=out[1])
                                                            graph2.line(x, y[2], line_color="Green",
                                                                        legend_label=out[2])
                                                            graph2.line(x, y[3], line_color="Magenta",
                                                                        legend_label=out[3])
                                                            graph2.line(x, y[4], line_color="Red", legend_label=out[4])

                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                            st.bokeh_chart(graph2, use_container_width=True)
                                                            banner1 = Image.open("img/coun.png")
                                                            st.image(banner1, use_column_width=True)
                                                            st.header("Contacts of experts from various fields")
                                                            for i in range(0, 5):
                                                                st.subheader(professions[int(l[i])])
                                                                xl=(df['contacts'][int(l[i]) - 1]).split(",")
                                                                for k in xl:
                                                                    ml=list(k.split(","))
                                                                    for kk in ml:
                                                                        st.write(kk,sep="\n")



if __name__=="__main__":
    main()