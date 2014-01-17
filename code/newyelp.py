import pandas as pd
import time
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


review_frame = pd.read_csv('yelp_training_set//yelp_training_set_review.csv')
review_frame.rename(columns={'votes_cool':'review_cool', 'votes_useful':'review_useful', \
	'votes_funny':'review_funny', 'stars':'review_stars'}, inplace=True)

#review_frame['review_frame_delta'] = review_frame['review_stars'].astype(float) - df['biz_stars']


del review_frame['type']
del review_frame['text']
print "\nReview frame: \n", review_frame
#print "\nSummary: \n", review_frame.describe()

user_frame = pd.read_csv('yelp_training_set//yelp_training_set_user.csv')
user_frame.rename(columns={'votes_cool':'user_cool', 'votes_useful':'user_useful', \
	'votes_funny':'user_funny', 'review_count':'user_review_count'}, inplace=True)

del user_frame['name']
del user_frame['type']
print "\nuser frame: \n", user_frame
#print "\nSummary: \n", user_frame.describe()


#checkin_frame = pd.read_csv('yelp_training_set//yelp_training_set_checkin.csv')
#print "\ncheckin frame: \n", checkin_frame
#print "\nSummary: \n", checkin_frame.describe()

business_frame = pd.read_csv('yelp_training_set//yelp_training_set_business.csv')
business_frame.rename(columns={'review_count':'biz_review_count'}, inplace=True)
del business_frame['neighborhoods']
del business_frame['full_address']
del business_frame['latitude']
del business_frame['longitude']
del business_frame['state']
del business_frame['name']
del business_frame['type']
del business_frame['city']
del business_frame['categories']
del business_frame['open']




print "\nbusiness frame: \n", business_frame
#print "\nSummary: \n", business_frame.describe()

urframe = pd.merge(user_frame, review_frame, on='user_id')

urbframe = pd.merge(urframe, business_frame, on='business_id')

train = urbframe.dropna()

target = train['review_useful']

#delete useless
del train['review_useful']
del train['business_id']
del train['user_id']

print "\nTrain READY: \n", train

# now do the machine learning. 

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
print "\n absolut percent error", np.mean(ae(testTarget, prediction))*100

