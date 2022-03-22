#pip install tika
#pip install pycaret
import os
from graphviz import render
import tensorflow
import pandas as pd
import keras
import nltk
from nltk.corpus import stopwords
import string
from spacy.pipeline import EntityRuler 
#from wordcloud import WordCloud
#import seaborn as sns
import textract
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()
import tensorflow as tf
#from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, LSTM , Dense, BatchNormalization, Input, Bidirectional, Dropout
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#import nolds
import scipy
#import pyeeg
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow
from keras.layers import Lambda, Dot, Concatenate, Activation, Embedding, add, Conv1D,GlobalMaxPool1D
from keras.models import Sequential
import pickle
import tempfile
from scipy import signal
from mne.time_frequency import psd_array_welch
#from tf.keras.models import Sequential, load_model, save_model, Mode
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from gensim import corpora
import gensim
import re
#spacy
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
import numpy as np
from tika import parser

import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example
## Flask Application

from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
# Initalise the Flask app
app = Flask(__name__, template_folder = '/Users/hritvikgupta/Downloads/flask-app/templates')
UPLOAD_FOLDER = '/Users/hritvikgupta/Downloads/flask-app/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")

def upload():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])

def upload():
    return render_template("index.html")

class resume_spacy_pdf_clean_skills():
    
    def __init__(self, path_to_pdf, cleaning_type):
        self.path = path_to_pdf
        #clean_types = ["mycleaning", "specific_cleaning"]
        self.cleaning_type = cleaning_type
        
        
    def nlp_model_initalization(self):
        nlp = spacy.load("en_core_web_lg")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk("/Users/hritvikgupta/Downloads/flask-app/data/jobs2.jsonl")
        #ruler.from_disk("/kaggle/input/jobsextractor/skills_12_dec_2.json")
        return nlp, ruler
    
    def pdf_to_text(self):
        raw = parser.from_file(self.path)
        text = raw['content']
        return text
    
    def cleaning_texts(self, text):
        if self.cleaning_type == "my_cleaning":
            resumeText = text
            resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
            resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
            resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
            resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
            resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
            resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText) 
            resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
            text = resumeText
            text = "".join([word.lower() for word in text if word not in string.punctuation])
            tokens = re.split('\W+', text)
            text = [lemmer.lemmatize(word) for word in tokens if word not in stopwords]
            review  = " ".join(i for i in text)
        
        if self.cleaning_type == 'specific_cleaning':

                review = re.sub(
                    '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
                    " ",
                    text,
                )
                review = review.lower()
                review = review.split()
                lm = WordNetLemmatizer()
                review = [ lm.lemmatize(word) for word in review if not word in set(stopwords)]
                review = " ".join(review)
        return review

    def get_skills(self, nlp, text):
        doc = nlp(text)
        skills_ = []
        others = []
        for ent in doc.ents:
            if "SKILL" in ent.label_:
                skills_.append(ent.text)
            elif ent.label_ == "ORG":
                others.append(ent.text)
         
        return skills_, others
    
    def get_job_resume_discription(self,text, pattern):
        ans = []
        pattern2 = pattern
        sp = text.split("\n")
        if len(sp) <=3 : 
            text2 = text.split("\xa0")
            for pat in pattern2:
                for t in text2:
                     if re.search(pat, t) != None:
                            ans.append(t)
        else:
            for pat in pattern2:
                for t in text.split('\n'):
                     if re.search(pat, t) != None:
                            ans.append(t)

        ans2 = " ".join([i for i in list(set(ans))])
        #final  = clean_data(ans2)
        return ans2 , list(set(ans))
    
    def get_description_without_re(self, text, pattern):
        sent = []
        text2 = inp.split('\n')
        #p5_ = p5.split(" ")+p6.split(" ")
        for i in p5.split(" "):
            for j in text2:
                if i.lower() in j.lower() and i not in sent:
                    sent.append(j)
                    #print("Text:-{}, pat:-{}".format(j,i))
       
        
        return sent
                

    def get_create_patterns(self, text):
        
        pattern2 = [r'\b(?i:plan)'+r'\b', r'\b(?i:years)'+r'\b',
        r'\b(?i:experience)'+r'\b',
        r'\b(?i:worked)'+r'\b',
        r'\b(?i:willing)'+r'\b',
        r'\b(?i:knowledge)'+r'\b',
        r'\b(?i:interview)'+r'\b', 
        r'\b(?i:applicants)'+r'\b',
        r'\b(?i:interview)'+r'\b',
        r'\b(?i:immediate)'+r'\b',
        r'\b(?i:interested)'+r'\b',
        r'\b(?i:opening)'+r'\b']

        #for i in w:
         #           pattern2.append(r'\b(?i)'+ str(i) + r'\b')
        w2 = text.split(" ")
        for i in w2:
            pattern2.append(r'\b(?i:'+str(i)+')'+r'\b')
    
        return pattern2
    
    def get_description_skill(self,nlp,des):
        skill = []
        des = des.lower()
        d = nlp(des)
        for i in d.ents:
            if 'SKILL' in i.label_:
                skill.append(i.text)
        return set(skill)
    
    def get_salary(self, text_from_pdf):
        pat2 = [r'\b(?i:salary)'+r'\b', r'\b(?i:Rs)'+r'\b', r'\b(?i:rs)'+r'\b']

        sal = []
        for p in pat2:
            for i in text_from_pdf.lower().split("\n"):
                if re.search(p, i)!= None:
                    sal.append(i)
        return sal
    
    
    
    #b ="posting for a computer engineer job in microsoft"
    #v = b.index("engineer")
    def get_job_from_training_spacy_model(self, data,nlp, clean_data):
    ## Making data for training
        TRAIN_DATA = data


        ##Loading model from NLP

        LABEL = "JOB"
        #nlp, ruler = spacy_.nlp_model_initalization()
        pipes1 = nlp.pipe_names
        ner=nlp.get_pipe("ner")
        optimizer = nlp.resume_training()
        move_names = list(ner.move_names)
        #pipe_exceptions = pipes1
        pipe_exceptions = ["ner", "tagger", "tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        #pipe_exceptions = ["ner", "tagger", "tok2vec"]
        
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])



        ### Training the model
        # Import requirements
        import random
        from spacy.util import minibatch, compounding
        from pathlib import Path
        from spacy.training.example import Example

        # TRAINING THE MODEL
        with nlp.disable_pipes(*other_pipes):

          # Training for 30 iterations
          for iteration in range(30):

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            #annotations = [entities for text, entities in batches]
            for batch in batches:
                    texts, annotations = zip(*batch)

                    example = []
                    # Update the model with iterating each text
                    for i in range(len(texts)):
                        doc = nlp.make_doc(texts[i])
                        example.append(Example.from_dict(doc, annotations[i]))

                    # Update the model
                    nlp.update(example, drop=0.5, losses=losses)


        ### Saving the model
        from pathlib import Path
        output_dir=Path('/kaggle/working/model')

        # Saving the model to the output directory
        if not output_dir.exists():
              output_dir.mkdir()
        nlp.meta['name'] = 'my_ner'  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        job = []
        text = nlp(clean_text.lower())
        for i in text.ents:
            if "JOB" in i.label_:
                job.append(i.text)

        return job
    
    def get_number_of_post(self, text_from_pdf):
        post = []
        pat2 =  [r'\b(?i:senior)'+r'\b',r'\b(?i:Trainee)'+r'\b',r'\b(?i:post)'+r'\b',r'\b(?i:reserch fellow)'+r'\b',r'\b(?i:junior)'+r'\b',r'\b(?i:nos)'+r'\b', r'\b(?i:position)'+r'\b', r'\b(?i:required)'+r'\b', r'\b(?i:posting)'+r'\b', r'\b(?i:vocation)'+r'\b',  r'\b(?i:vacancy)'+r'\b',  r'\b(?i:opening)'+r'\b',  r'\b(?i:place)'+r'\b']
        sal = []
        for p in pat2:
            for i in text_from_pdf.lower().split("\n"):
                if re.search(p, i)!= None:
                    post.append(i)

        return post

                
    def get_matching_score(self,req, original):
        req_skills = req
        resume_skills = original
        score = 0
        for x in req_skills:
            if x in resume_skills:
                score += 1
        req_skills_len = len(req_skills)
        match = round(score / req_skills_len * 100, 1)

        print(f"The current Resume is {match}% matched to your requirements")
        return match 

    def train_nlp_model_entity(self,model):
        data = [
              ("Name of the Posts: Programmer ", {"entities": [(19, 29, "JOB")]}),
              ("Requirement for analyst part time in google", {"entities": [(16, 23, "JOB")]}),
              ("Job posting for a writter", {"entities": [(18, 25, "JOB")]}),
              ("vacancy for a manager in tata industries", {"entities": [(14,21, "JOB")]}),
              ("posting for a intern in IIT bhu", {"entities": [(14,20, "JOB")]}),
              ("vacancy for a research intern", {"entities": [(14,22, "JOB")]}),
              ("required a technician for chemistry lab", {"entities": [(11,21, "JOB")]}),
              ("temprary requirement for research fellow urgently", {"entities": [(34,40, "JOB")]}),
              ("position for senior journslist in ABP News", {"entities": [(20,30, "JOB")]}),
              ("employment for a engineer needed urgently", {"entities": [(17,25, "JOB")]}),
              ("medical traineer at aiims delhi part time reqiured", {"entities": [(8,16, "JOB")]}),
              ("post for a screwdriver endevour is empty from our neighbour", {"entities": [(11,22, "JOB")]}),
              ("posting for a computer engineer job in microsoft", {"entities": [(23,31, "JOB")]}),
              ("profession required is a manager in JSW", {"entities": [(25,32, "JOB")]}),
              ("opening for a web developer in india", {"entities": [(18,27, "JOB")]})
              ]
        nlp = model
        ner = nlp.get_pipe("ner")
        for _, annotations in data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        pipe_exceptions = ['ner', "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        from spacy.training.example import Example
        import random
        from spacy.util import minibatch, compounding
        from pathlib import Path

        # TRAINING THE MODEL
        with nlp.disable_pipes(*unaffected_pipes):
            for iteration in range(30):

                # shuufling examples  before every iteration
                random.shuffle(data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    for texts, annotations in batch:
                        doc = nlp.make_doc(texts)
                        example = Example.from_dict(doc , annotations)
                        nlp.update( [example], # batch of annotations
                                    drop=0.5,  # dropout - make it harder to memorise data
                                    losses=losses,

                                )
                        print("Losses", losses)
        return nlp

@app.route('/resume',methods=['GET','POST'])
def resume():
    if request.method == 'POST':  
        file_path = request.files['file']
        ## For Matching resume
    pa  = file_path.filename 
    path_resume = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], pa))
    jobs = request.args.get("jobs")
   # path_resume  ="/Users/hritvikgupta/Downloads/flask-app/data/sid.pdf"
    spacy_2 =  resume_spacy_pdf_clean_skills(path_resume, "specific_cleaning" )
    nlp2, ruler = spacy_2.nlp_model_initalization()
    text_from_pdf2 = spacy_2.pdf_to_text()
    clean_text2 = spacy_2.cleaning_texts(text_from_pdf2)
    get_skills_from_resume2, others= spacy_2.get_skills(nlp2,clean_text2)
    match = spacy_2.get_matching_score(set(jobs), set(get_skills_from_resume2))
    return render_template('index.html', match="Matched resume {}".format(match))

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':  
        file_path = request.files['file']
        filepath2 = request.files['file2']
    #s = str(file_path).index("/Users")
    #e = str(file_path).index(".pdf")
    #path  = str(file_path)[s:e]

    pa  = file_path.filename 
    path = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], pa))
    #path  ="/Users/hritvikgupta/Downloads/flask-app/data/Careers-Sample-job-Ad.pdf"
    spacy_ =  resume_spacy_pdf_clean_skills(path, "specific_cleaning" )
    #text_from_pdf = str(textract.process(path))
    #print(text_from_pdf)
    text_from_pdf = spacy_.pdf_to_text()
    #clean_text = spacy_.cleaning_texts(text_from_pdf)
    p6 = ["agile", "deadline-oriented", "multitask", "pressure","multitasking", "enthusiastic", "high energy", "committed", "proactive", "pressure", "independently", "entrepreneurial", "independent", "resourceful"]
    p5 = "essential salary necessary desirable applicant strong background qualification overtime experience worked knowlegde interview applicants immediate opening required flexible worked working skill skills role roles key full-time part-time well-paid badly paid high-powered stressful challenging rewarding repetitive glamorous plan years experience worked willing knowledge interview applicants interview immediate interested opening responsiblity resposiblities Administrative assistant Customer service Receptionist Part time UPS package handler part time entry level"
    pattern = p5+ "".join([i for i in p6]) #"".join([i for i in common_words2]) 
    nlp, ruler = spacy_.nlp_model_initalization()
    pat = spacy_.get_create_patterns(pattern)
    des , list_des = spacy_.get_job_resume_discription(text_from_pdf, pat)
    sal = spacy_.get_salary(text_from_pdf)
    skills_required = spacy_.get_description_skill(nlp, text_from_pdf)
    number_of_post = spacy_.get_number_of_post(text_from_pdf)
    #jobs = trained_spacy_model_jobs(nlp, data)
    output_dir = Path('/Users/hritvikgupta/Downloads/flask-app/data/nlp_spacy_model')
    #nlp.to_disk(output_dir)
    nlp_updated = spacy.load(output_dir)
    doc1 = nlp_updated(text_from_pdf)
    jobs = [i.text for i in doc1.ents]
    #jobs = spacy_.train_nlp_model_entity(nlp)
    #prediction = [jobs,  des, skills_required , number_of_post, sal]
    prediction = {}
    prediction["Jobs"] = jobs
    prediction['description'] = des
    #jobs_(jobs)


    ## For Matching resume
    pa2  = filepath2.filename 
    path_resume = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], pa2))
    #path_resume  ="/Users/hritvikgupta/Downloads/flask-app/data/sid.pdf"
    spacy_2 =  resume_spacy_pdf_clean_skills(path_resume, "specific_cleaning" )
    nlp2, ruler = spacy_2.nlp_model_initalization()
    text_from_pdf2 = spacy_2.pdf_to_text()
    clean_text2 = spacy_.cleaning_texts(text_from_pdf2)
    get_skills_from_resume2, others= spacy_.get_skills(nlp2,clean_text2)
    match = spacy_.get_matching_score(set(skills_required), set(get_skills_from_resume2))
    return render_template('index.html', prediction_text="Skills Required  : {}".format(set(skills_required)), 
                                        jobs_text = "Jobs Opening : {}".format(set(jobs)),
                                        salary_re = "Salary offered : {}".format(set(sal)),
                                        nop = "Number of Openings : {}".format(number_of_post),
                                        match="Matched resume : {}".format(match),
                                        rs = "Skills from resume : {}".format(set(get_skills_from_resume2)))
    


if __name__ == '__main__':
    app.run(debug=True)