def draw_line(x1,y1,x2,y2,n,i):  #Draws line for all points
def getPoints(points_X,points_Y,totalPoints): #Find No of points between two points
def generate_sample_points(points_X, points_Y) #Generates sample points
def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y): #Removes redundant data
def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):  #Normalisation
def dist(x1,y1,x2,y2): #Euclidean distance
def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y): #Calculates Location Score
def get_integration_scores(shape_scores, location_scores) #Integration_score
def get_best_word(valid_words, integration_scores): #Finds the best possible word



