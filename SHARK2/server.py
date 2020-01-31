'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json
import math
import numpy as np
from scipy.spatial import distance



app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

def draw_line(x1,y1,x2,y2,n,i):  #Draws line for all points
    d = (( (x2 - x1)**2 + (y2 - y1)**2 )*0.5)/(n+1)
    d_full = ( (x2 - x1)**2 + (y2 - y1)**2 )*0.5
    points_X = []
    points_Y = []
    count = 0
    if(i == 0):
        points_X.append(round(float(x1), 2))
        points_Y.append(round(float(y1), 2))
        count = count + 1
    if(d_full==0):
        count1 = 0
        for x in range(n):
            count1 = count1 + 1
            if(count1>n):
                break
            points_X.append(round(float(x1), 2))
            points_Y.append(round(float(y1), 2))
    else:
        s = d / d_full
        a = s
        count = 0
        while a < 1:
            if(count > n-1):
                break
            x = (1 - a) * x1 + a * x2
            y = (1 - a) * y1 + a * y2
            points_X.append(round(float(x), 2))
            points_Y.append(round(float(y), 2))
            a += s
            count = count + 1
    points_X.append(round(float(x2), 2))
    points_Y.append(round(float(y2), 2))
    return points_X,points_Y

def getPoints(points_X,points_Y,totalPoints): #Find No of points between two points
    points = []
    for x in range(len(points_X)):
        point =[]
        point.append(points_X[x])
        point.append(points_Y[x])
        points.append(point)
    pointsAlreadyPresent = len(points)
    requiredMore = totalPoints - pointsAlreadyPresent

    divisions = pointsAlreadyPresent - 1
    minValuePerSection = math.floor( requiredMore/divisions )
    valuePerSegments = []
    count = 0
    for x in range(divisions):
        valuePerSegments.append(minValuePerSection)
    requiredMore = requiredMore%divisions
    for x in range(requiredMore):
        valuePerSegments[x] = valuePerSegments[x] + 1
    for x in range(divisions):
            count = count + valuePerSegments[x]

    finalPoints_X = []
    finalPoints_Y = []
    for x in range(divisions):
        firstPoint = points[x]
        lastPoint = points[x+1]
        i=0
        if(x == 0):
            i=0
        if(x != 0):
            i=1
        xz = draw_line(firstPoint[0],firstPoint[1],lastPoint[0],lastPoint[1],valuePerSegments[x],i)
        points1 = xz[0]
        points2 = xz[1]
        for x in range(len(points1)):
            finalPoints_X.append(points1[x])
        for x in range(len(points2)):
            finalPoints_Y.append(points2[x])
    return finalPoints_X,finalPoints_Y

def generate_sample_points(points_X, points_Y): #Generates sample point
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    if isinstance(points_X[0], list):
        points_X = points_X[0]
        points_Y = points_Y[0]
    sample_points_X, sample_points_Y = [], []
    if (len(points_X) < 101):
        val = getPoints(points_X, points_Y, 100)
        sample_points_X = val[0]
        sample_points_Y = val[1]
    else:
        sample_points_X = points_X
        sample_points_Y = points_Y
    # TODO: Start sampling (12 points)


    return sample_points_X, sample_points_Y

# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y): #Removes redundant data
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''

    length=len(template_sample_points_X)
    gg_x,gg_y=generate_sample_points(gesture_points_X,gesture_points_Y)
    length=len(gg_x)

    First_Sample_Gesture_Points_X=gg_x[0]
    First_Sample_Gesture_Points_Y=gg_y[0]
    Last_Sample_Gesture_Points_X=gg_x[length-1]
    Last_Sample_Gesture_Points_Y=gg_y[length-1]


    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 40
    # TODO: Do pruning (12 points)
    for x in range(0, len(template_sample_points_X)):
        length = len(template_sample_points_X[x])
        First_Template_Sample_X = template_sample_points_X[x][0]
        Last_Template_Sample_X = template_sample_points_X[x][length-1]
        First_Template_Sample_Y = template_sample_points_Y[x][0]
        Last_Template_Sample_Y = template_sample_points_Y[x][length-1]
        df_x=distance.euclidean(First_Sample_Gesture_Points_X, First_Template_Sample_X)
        df_y=distance.euclidean(First_Sample_Gesture_Points_Y,First_Template_Sample_Y)
        dl_x=distance.euclidean(Last_Sample_Gesture_Points_X,Last_Template_Sample_X)
        dl_y=distance.euclidean(Last_Sample_Gesture_Points_Y,Last_Template_Sample_Y)
        if(df_x<threshold and df_y<threshold  and dl_x<threshold and dl_y<threshold):

            valid_template_sample_points_X.append(template_sample_points_X[x])
            valid_template_sample_points_Y.append(template_sample_points_Y[x])
            valid_words.append(words[x])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):  #Normalisation
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''



    Mean_Gesture_point_X = sum(gesture_sample_points_X)/ len(gesture_sample_points_X)
    Mean_Gesture_point_Y = sum(gesture_sample_points_Y)/ len(gesture_sample_points_Y)
    tempx,tempy=[],[]
    for x in range(0,len(gesture_sample_points_X)):
            gesture_sample_points_X[x] = gesture_sample_points_X[x] - Mean_Gesture_point_X
            gesture_sample_points_Y[x] = gesture_sample_points_Y[x] - Mean_Gesture_point_Y



    Minimum_gesture_sample_points_X = (min(gesture_sample_points_X))
    Minimum_gesture_sample_points_Y = (min(gesture_sample_points_Y))

    Maximum_gesture_sample_points_X = (max(gesture_sample_points_X))
    Maximum_gesture_sample_points_Y = (max(gesture_sample_points_Y))
    shape_scores = []
    # TODO: Set your own L
    L = 1
    w_Valid_Gesture_Points = Maximum_gesture_sample_points_X - Minimum_gesture_sample_points_X
    h_Valid_Gesture_Points = Maximum_gesture_sample_points_Y - Minimum_gesture_sample_points_Y


    d = max(w_Valid_Gesture_Points, h_Valid_Gesture_Points)
    s1 = L / d
    for x in range(0,len(gesture_sample_points_X)):
        gesture_sample_points_X[x]=s1*gesture_sample_points_X[x]

    for x in range(0,len(gesture_sample_points_Y)):
        gesture_sample_points_Y[x]=s1*gesture_sample_points_Y[x]



    for x in range(len(valid_template_sample_points_X)):
        Median_valid_template_X=sum(valid_template_sample_points_X[x])/len(valid_template_sample_points_X[x])
        for t in range(len(valid_template_sample_points_X[x])):
            valid_template_sample_points_X[x][t]=valid_template_sample_points_X[x][t]-Median_valid_template_X

    for x in range(len(valid_template_sample_points_Y)):
        Median_valid_template_Y = sum(valid_template_sample_points_Y[x]) / len(valid_template_sample_points_Y[x])
        for t in range(len(valid_template_sample_points_Y[x])):
            valid_template_sample_points_Y[x][t] = valid_template_sample_points_Y[x][t] - Median_valid_template_Y


    for x in range(len(valid_template_sample_points_X)):
        Minimum_Valid_template_sample_point_X = min(valid_template_sample_points_X[x])  #Minimum of all valid template of X
        Minimum_Valid_template_sample_point_Y = min(valid_template_sample_points_Y[x])  #Minimum of all valid templates of Y.
        Maximum_Valid_template_sample_point_X = max(valid_template_sample_points_X[x])  #Maximum of all valid template of X
        Maximum_Valid_template_sample_point_Y = max(valid_template_sample_points_Y[x])  #Maximum of all valid template of Y

        w_Valid_template_points = float(Maximum_Valid_template_sample_point_X) - float(Minimum_Valid_template_sample_point_X)
        h_Valid_template_points = float(Maximum_Valid_template_sample_point_Y) - float(Minimum_Valid_template_sample_point_Y)

        g=max(w_Valid_template_points,h_Valid_template_points)
        s2=L/g

        for t in range(len(valid_template_sample_points_X[x])):
            valid_template_sample_points_X[x][t] = s2*valid_template_sample_points_X[x][t]
        for t in range(len(valid_template_sample_points_Y[x])):
            valid_template_sample_points_Y[x][t] = s2*valid_template_sample_points_Y[x][t]

    for x in range(len(valid_template_sample_points_X)):
        Totaldistance = 0
        distance=0
        for t in range(len(valid_template_sample_points_X[x])):
            distance = math.sqrt(((valid_template_sample_points_X[x][t] - gesture_sample_points_X[t]) ** 2) + ((valid_template_sample_points_Y[x][t] -gesture_sample_points_Y[t]) ** 2))
            Totaldistance=Totaldistance+distance
        Xs=Totaldistance/100
        shape_scores.append(Xs)
    # TODO: Calculate shape scores (12 points)

    return shape_scores
def dist(x1,y1,x2,y2): #Euclidean distance

    x = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return x

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y): #Calculates Location Score
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''


    location_scores = []
    radius = 15
    #
    d = []
    d_pq=[]
    D_p=0
    #print(gesture_sample_points_X)
    #print(valid_template_sample_points_X)
    for x in range(0,len(gesture_sample_points_X)):
       for y in range(0,len(valid_template_sample_points_X)):
           d=[]
           for t in range(0,len(valid_template_sample_points_X[y])):
                 d.append(dist(valid_template_sample_points_X[y][t],valid_template_sample_points_Y[y][t],gesture_sample_points_X[x],gesture_sample_points_Y[x]))
                        #print(gesture_sample_points_X[x]*valid_template_sample_points_X[y][t])
           d_pq.append(min(d))
       #print("wp")
       d_pq = [val - radius for val in d_pq]
       b=max(d_pq)
    D_p=D_p+max(b,0)

    d_pqTempgest=[]
    D_pTempgest=0
    for y in range(len(valid_template_sample_points_X)):
        for t in range(len(valid_template_sample_points_X[y])):
            q=[]
            for x in range(len(gesture_sample_points_X)):
                q.append(dist(valid_template_sample_points_X[y][t], valid_template_sample_points_Y[y][t],
                              gesture_sample_points_X[x], gesture_sample_points_Y[x]))

            d_pqTempgest.append(min(q))
        d_pqTempgest = [val - radius for val in d_pqTempgest]
        l=max(d_pqTempgest)
    D_pTempgest = D_pTempgest + max(l , 0)

    XLFinal=[]
    if D_pTempgest==0 and D_p==0:
        location_scores.append(0)
    else:
        XL=0
        for x in range(len(valid_template_sample_points_X)):
            for t in range(len(valid_template_sample_points_X[x])):
                #print(len(valid_template_sample_points_X))
                #print(len(gesture_sample_points_X))
                distance = math.sqrt(((valid_template_sample_points_X[x][t] - gesture_sample_points_X[t]) ** 2) + (
                            (valid_template_sample_points_Y[x][t] - gesture_sample_points_Y[t]) ** 2))
            XL = XL + distance*0.01
            XLFinal.append(XL)



    #TODO: Calculate location scores (12 points)
    location_scores=XLFinal
    return location_scores


def get_integration_scores(shape_scores, location_scores):

    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.8
    # TODO: Set your own location weight
    location_coef = 0.2
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    integration_scores.sort()
    possiblewords=[]
    # TODO: Set your own range.
    n = 4
    for x in range(n):
        possiblewords.append(valid_words[x])
    # TODO: Get the best word (12 points)
    return possiblewords


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)
    shape_scores = get_shape_scores(list(gesture_sample_points_X), list(gesture_sample_points_Y), list(valid_template_sample_points_X), list(valid_template_sample_points_Y))

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + str(best_word) + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'
if __name__ == "__main__":
    app.run()
