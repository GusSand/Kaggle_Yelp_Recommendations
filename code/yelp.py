
from sklearn import linear_model
from sklearn.metrics import r2_score
from pymongo import MongoClient
from pandas import *
import pandas as pd 
import numpy as np 
import re
from datetime import datetime
from ols import ols
import matplotlib.pyplot as plt
import scipy as sp
import time
import pylab as pl

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


def printSeparator():
	print '=============================================================================='

#def get_reviewTestData() :
	# x = 0
	# printSeparator()
	# print "\nGetting review data... "
	# for review in review_coll.find() :

 
def rmsle(actual, predicted):
	"""
    Computes the root mean squared log error.

    This function computes the root mean squared log error between two lists
    of numbers.

    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value

    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted

    """
	return np.sqrt(msle(actual, predicted))

def msle(actual, predicted):
    """
    Computes the mean squared log error.

    This function computes the mean squared log error between two lists
    of numbers.

    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value

    Returns
    -------
    score : double
            The mean squared log error between actual and predicted

    """
    return np.mean(sle(actual, predicted))

def sle(actual, predicted):
    """
    Computes the squared log error.

    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.

    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value

    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted

    """
    return (np.power(np.log(np.array(actual)+1) - 
            np.log(np.array(predicted)+1), 2))

def ae(actual, predicted):
    """
    Computes the absolute error.

    This function computes the absolute error between two numbers,
    or for element between a pair of lists or numpy arrays.

    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value

    Returns
    -------
    score : double or list of doubles
            The absolute error between actual and predicted

    """
    return np.abs(np.array(actual)-np.array(predicted))

def get_ReviewData(rcoll, rtotal, getUseful=True):

	base_date = datetime(2013, 01, 01)

	x = 0

	#printSeparator()

	review_textArray = [0] * rtotal
	review_wordArray = [0] * rtotal
	review_starsArray = [0] * rtotal
	review_useful_count = [0] * rtotal
	review_dateArray = [0] * rtotal
	review_dateDeltaArray = [0] * rtotal
	review_userId = [0] * rtotal
	review_bizId = [0] * rtotal 
	review_user_useful = [NaN] * rtotal
	review_user_review_count = [0] * rtotal


	print "\nGetting review data... "
	for review in rcoll.find() :
		stars = review['stars']

		#if stars != 3 :		
		#text = review['text']

		if (getUseful) :
			review_useful_count[x] = review['votes']['useful']

		# date on on Mongo is of the form 2011-01-30 
		datestring = review['date']
		review_dateArray[x] = datetime.strptime(datestring, '%Y-%m-%d')
		tdelta = base_date - review_dateArray[x]
		review_dateDeltaArray[x] = tdelta.total_seconds()
		text = review['text']
		review_textArray[x] = len(text)
		review_wordArray[x] = len(re.findall(r'\w+', text))
		review_starsArray[x] = review['stars']

		review_userId[x] = review['user_id']
		review_bizId[x] = review['business_id']

		#review_user_useful[x] = user_useful_dict.get(review_userId[x], mean_user_useful)
		#review_user_review_count[x] = user_review_count_dict.get(review_userId[x], mean_user_review_count)


		x = x + 1
		if x == review_total :
			break

	print "\nDONE: Getting review data"

	# review_bizCount = [0] * rtotal 
	# x = x - 1
	# while (x > 0 ) :
	# 	review_bizCount[x] = review_bizId.count(review_bizId[x])
	# 	x = x - 1

	#print review_bizCount.sort()

	printSeparator()

	review = {'business_id':review_bizId, 'user_id': review_userId, 'review_stars':review_starsArray,  \
		'review_dateDelta': review_dateDeltaArray, 'review_useful': review_useful_count, 'review_text_len':review_textArray,  \
		'review_word_count': review_wordArray}

	review_frame = DataFrame(review)
	return review_frame

def get_UserData(ucoll, utotal, getUseful=True) :

	#print "\ntotal number of users: ", user_count
	#print "\ntotal used for now: ", user_total


	user_userId = [0] * utotal
	user_avgStars = [0] * utotal
	user_reviewCount = [0] * utotal
	user_useful = [NaN] * utotal
	user_cool = [NaN] * utotal
	user_funny = [NaN] * utotal
	user_useful_dict = dict()
	user_review_count_dict = dict()


	printSeparator()

	x = 0
	very_useful = 0
	print "\nGetting user data... "
	for user in ucoll.find() :
		user_userId[x] = user['user_id']
		user_avgStars[x] = user['average_stars']
		user_reviewCount[x] = user['review_count']
		
		if (getUseful) :
			user_useful[x] = user['votes']['useful']
			user_funny[x] = user['votes']['cool']
			user_cool[x] = user['votes']['funny']

			if user_reviewCount[x] < user_useful[x] :
				#print "Review count < UserUseful: weird ...", user_reviewCount[x], user_useful[x]
				very_useful = very_useful + 1


		user_useful_dict[user_userId[x]] = user_useful[x]
		user_review_count_dict[user_userId[x]] = user_reviewCount[x]


		x = x + 1
		if x == user_total :
			break
	print "\nDONE: Getting user data"
	printSeparator()

	user = { 'user_id': user_userId,  'user_useful': user_useful, 'user_reviews' : user_reviewCount, \
		'user_funny': user_funny, 'user_cool': user_cool,  'user_avgStars': user_avgStars}
	user_frame = DataFrame(user)
	return user_frame
	
	#print "TODO: ", user_frame.head()
	#print "very useful ", very_useful
	#print "percent of very useful ", (float(very_useful)/user_total)* 100

def get_BizData(b_coll, b_count) :

	# rint "\ntotal number of biz: ", biz_count
	# print "\ntotal biz used for now ", biz_total

	printSeparator()
	b_bizId = [0] * b_count
	b_reviewCount = [0] * b_count
	b_stars = [0] * b_count

	x = 0
	print "\nGetting biz data... "
	for biz in b_coll.find() :
		b_bizId[x] = biz['business_id']
		b_reviewCount[x] = biz['review_count']
		b_stars[x] = biz['stars']

		x = x + 1
		if x == biz_total :
			break

	print "\nDONE: Getting biz data"
	printSeparator()


	biz = {'business_id': b_bizId, 'biz_stars' : b_stars, 'biz_review_count' : b_reviewCount }
	b_frame = DataFrame(biz)
	return b_frame

def get_CheckinData() :
	# rint "\ntotal number of biz: ", biz_count
	# print "\ntotal biz used for now ", biz_total

	printSeparator()

	x = 0
	print "\nGetting checkin data... "
	for ck in checkin_coll.find() :
		checkin_bizId[x] = ck['business_id']

		x = x + 1
		if x == checkin_total :
			break

	print "\nDONE: Getting biz data"
	printSeparator()

def normalize_Data(dataSet) :

	npdata = np.asarray(dataSet)
	mymin = npdata.min()
	mymax = npdata.max()
	myrange = mymax - mymin

	print "before normalize ", dataSet

	x = 0
	for val in npdata:
		val = (float(val) - mymin)/myrange
		dataSet[x] = val
		x = x+1 


	print "after normalize ", dataSet
	return dataSet

def summarize(arr, str):
	df = DataFrame(arr)
	print "\nsummarizing: ", str
	print df.head()
	print df.describe()


def analyze_UserData(user_frame, hasUseful=True):
	printSeparator()
	print "\nAnalyzing User Data \n"
	#user = { 'user_id' : user_userId, 'num_reviews' : user_reviewCount, 'user_avgStars' : user_avgStars }
	#user = { 'num_reviews' : user_reviewCount, 'user_avgStars' : user_avgStars, 'user_useful': user_useful}


	if (hasUseful) :
		user_frame['useful_pct'] = user_frame['user_useful'].astype(float)/user_frame['user_reviews']
		print "\nuser:  user_useful distribution \n", user_frame['user_useful'].value_counts()
		print "\nuser:  user_percentage distribution\n ", user_frame['useful_pct'].value_counts()

	print "\nuser frame: head \n", user_frame.head()
	print "\nuser frame: describe \n", user_frame.describe()
	print "\nuser:  user_reviews count distribution \n", user_frame['user_reviews'].value_counts()


	#print "\nuser:  duplicates\n", user_frame.duplicated()

	return user_frame

	
	# user_reviews_frame = DataFrame(user)

	# user_reviews_frame[:20].plot(kind='barh', label='users: useful - num reviews')
	# plt.legend()
	# plt.show()

def analyze_ReviewData(review_frame, hasUseful=True):
	printSeparator()
	print "\nAnalyzing Review Data \n"

	#summarize(review_starsArray, 'review_stars')
	#summarize(review_dateDeltaArray, 'review_dateDelta')
	#summarize(review_useful_count, 'review_useful') 


	print "\nreview frame: head \n", review_frame.head()
	print "\nreview frame: describe \n", review_frame.describe()

	#print "\nreview:  date delta distribution \n", review_frame['review_dateDelta'].value_counts()
	print "\nreview:  stars distribution \n", review_frame['review_stars'].value_counts()

	if (hasUseful) :
		print "\nreview:  useful distribution \n", review_frame['review_useful'].value_counts()
	#print "\nreview:  duplicates ? \n", review_frame.duplicated('business_id, user_id')

	return review_frame

def analyze_CheckinData():
	printSeparator()
	print "\nAnalyzing checkin Data \n"

	#ck = {'business_id', checkin_bizId}
	ck_frame = DataFrame(checkin_bizId)
	ck_frame.columns = ['business_id']

	print "\ncheckin frame: head \n", ck_frame.head()
	print "\ncheckin frame: describe \n ", ck_frame.describe()
	print "\ncheckin frame: distribution \n", ck_frame['business_id'].value_counts()
	#print "\ncheckin frame: duplicates \n", ck_frame.duplicated()

	return ck_frame



def analyze_BizData(b_frame):

	printSeparator()
	print "\nAnalyzing Business Data \n"
	#_count = 20

	print "\business frame: head \n", b_frame.head()
	print "\nbusiness frame: describe \n", b_frame.describe()
	#print "\nbusiness frame: duplicated ? \n", biz_frame.duplicated()
	#print "biz frame \n", biz_frame


	print "\nbusiness frame: stars distribution \n", b_frame['biz_stars'].value_counts()
	print "\nbusiness frame: review count distribution \n ", b_frame['biz_review_count'].value_counts()

	#return biz_frame

	#print"\n biz stars distribution \n", bizstars[:_count]

	# bizstars[:].plot(kind='barh', label='Biz star distribution')
	# plt.legend()
	# plt.show()

	# bizreview[:_count].plot(kind='barh', label='Biz review count')
	# plt.legend()
	# plt.show()


def TrainNClassify(urbframe) :
	
	#print "\nafter join review & user: \n", urframe.head()
	#print "\nafter join summary \n", urframe.describe()

	#print "\nafter join review & summary & biz \n", urbframe.head()
	#print "\nafter join summary \n ", urbframe.describe()

	#urframe.drop_duplicates(['user_id', 'business_id'])
	# print "\nafter remove dupes head \n", urframe.head()
	# print "\nafter remove dupes describe \n", urframe.describe()

	## now the machien learning part starts
	#target = urbframe['review_useful']


	#drop user_id, biz_id, review_useful
	del urbframe['user_id']#, 'business_id', 'review_useful']
	del urbframe['business_id']#, 'business_id', 'review_useful']
	#del urbframe['review_useful']#, 'business_id', 'review_useful']

	##############
	#
	# train models
	#
	train = urbframe.dropna()
	target = train['review_useful']
	del train['review_useful']

	print "\n train before calling rfregressor \n", train.describe()
	print "\n target before calling rfregressor \n", target.describe()


	forest = RandomForestRegressor(n_estimators=150, compute_importances=True)
	forest.fit(train, target)
	predictions = forest.predict(review_test)
	print "\npredictions: \n", predictions
	print "\nreview test\n", review_test


#########################################################################################

# declare and initialize globals
NaN = float('nan')
db = MongoClient().test

review_coll = db['training_review']
review_total = review_coll.count()

test_review_coll = db['test_review']
test_review_total = test_review_coll.count()


##
# How many reviews the user has 
#
user_coll = db['training_user']
user_total = user_coll.count()

test_user_coll = db['test_user']
test_user_total = test_user_coll.count()
 
# print user_count

##
# how many reviews the business has
# 
biz_coll = db['training_business']
biz_total = biz_coll.count()

test_biz_coll = db['test_business']
test_biz_total = test_biz_coll.count()


##
# Count the number of checkins in a biz
checkin_coll = db['training_checkin']
checkin_count = checkin_coll.count()
checkin_total = checkin_count

checkin_bizId = [0] * checkin_total
checkin_bizId_dict = dict()


#########################################################################################

## MAIN LOGIC 
doTrain = True
doTest = False #True
doMachineLearning = True


## get all the data from the training database
## 
if (doTrain) :
	bframe = get_BizData(biz_coll, biz_total)
	uframe = get_UserData(user_coll, user_total)
	rframe = get_ReviewData(review_coll, review_total)
	#rframe = get_CheckinData(checkin_coll, checkin_total)

	analyze_BizData(bframe)
	analyze_UserData(uframe)
	analyze_ReviewData(rframe)

	# first merge userframe and reviewframe
	urframe = pd.merge(rframe, uframe, how='left', on= 'user_id')

	# Now joing userreview with bizframe
	df = pd.merge(urframe, bframe)
	df['biz_review_star_delta'] = df['review_stars'].astype(float) - df['biz_stars']


	print "TEST: urb frame", df

	train = df.dropna()
	print "TEST: after dropna", train
	
	target = train['review_useful']

	# delete useless columns
	del train['review_useful']
	del train['business_id']
	del train['user_id']

	#create the delat 
	print "TRAIN: after deleting useless columns: Final columns to use. Train\n", train
	print "TRAIN: after deleting useless columns: Final columns to use. Train\n", train.describe()
	print "TRAIN: after deleting useless columns: Train\n", train.head()




##
## Get all the data from the test database
#
if (doTest) :
	t_bframe = get_BizData(test_biz_coll, test_biz_total)
	t_uframe = get_UserData(test_user_coll, test_user_total, False)
	t_rframe = get_ReviewData(test_review_coll, test_review_total, False)

	analyze_BizData(t_bframe)
	analyze_UserData(t_uframe, False)
	analyze_ReviewData(t_rframe, False)

	t_urframe = pd.merge(t_rframe, t_uframe, how='left', on='user_id')
	t_df = pd.merge(t_urframe, t_bframe)
	t_df['biz_review_star_delta'] = t_df['review_stars'].astype(float) - t_df['biz_stars']

	# delete the ones that have nothing
	del t_df['user_useful']#, 'business_id', 'review_useful']
	del t_df['review_useful']
	del t_df['business_id']
	del t_df['user_id']

	print "TEST: urb frame", t_df
	print "TEST: urb values ", t_df.head()

	review_test = t_df.dropna()

	print "TEST: test df after dropping NA\n", review_test	
	print "TEST: test df after dropping NA\n", review_test.describe()
	print "TEST: test df head after dropping NA\n", review_test.head()


if (doMachineLearning) : 

	#del train['biz_review_star_delta']
	#del train['useful_pct']

	print "\n train before calling rf regressor \n", train
	print "\n train before calling rf regressor \n", train.describe()
	print "\n target before calling rf regressor \n", target.describe()

	# estimators = # of trees
	# oob_score = Whether to use out-of-bag samples to estimate the generalization error.
	regressor = RandomForestRegressor(n_estimators=150, compute_importances=True)

	start = time.clock()

	## 
	## use only a subset to train the predictor
	train_count = int(len(train) * .80)
	trainData = train[:train_count]
	trainTarget = target[:train_count]

	testData = train[train_count:]
	testTarget = target[train_count:]


	#train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
	#trainData, testData = train[train['is_train']==True], train[train['is_train']==False]

	print "starting the regressor.fit...\n"
	regressor.fit(trainData, trainTarget)
	print "done with regressor.fit. Elapsed Time \n", (time.clock() - start)
	
	print"\n Feature importances: ", regressor.feature_importances_ 


	#oobprediction = regressor.oob_prediction_

	#print(your_error_metric(oobprediction, x))

	print "\nstarting predictions..."
	start = time.clock()
	prediction = regressor.predict(testData)
	print "\ndone with predictions. Elapsed Time: \n", time.clock() - start

	dpred = DataFrame(prediction)
	print "\npredictions: \n", dpred, dpred.describe(), dpred.head()
	
	print "\nactual: \n ", target

	# error measurements
	# mse: mean square error
	# rmsle: root mean square log error
	print "\n rmsle ", rmsle(testTarget, prediction)
	## print "\n absolut percent error", np.mean(ae(testTarget, prediction))*100

	#print "\n rmse ", rmse(testTarget, prediction)
	#print "\n mse ", mse(testTarget, prediction)




