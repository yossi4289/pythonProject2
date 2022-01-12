##-------------------------packages-------------------------------------
import random

import pandas as pd
from sklearn.model_selection import train_test_split

###----------------------- loading data ------------------------------
df = pd.read_excel("XY_train.xlsx", engine='openpyxl')


###-----------------------------------------------preprocessing------------------------------------------------------------

##------------------missing values--------------------
# ---------------checking how many missing values we have --------------
def missing_values_checks(df):
    MissingValue = df.isnull().sum()
    print(MissingValue)


missing_values_checks(df)

# --------------deleting samples with 2 or more missing values---------------------
df = df.dropna(thresh=11)

# ------------ delete 48 samples from experience---------
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset=["experience"], inplace=True)

# ------------ delete 163 samples from enrolled_university---------
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset=["enrolled_university"], inplace=True)

# -----------adding MVs to education_level with Primary School-----------
df['education_level'].fillna(value='Primary School', inplace=True)

# ----------------- adding MVs to major_discipline (no major) + deleting 21 samples -----------------
cond = ((df['education_level'] == 'Primary School') | (df['education_level'] == 'High School'))
df['major_discipline'] = df['major_discipline'].fillna(cond.map({True: 'No Major'}))

nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset=["major_discipline"], inplace=True)

# ------------fill MVs of last_new_job with the mean ------------------------
df['last_new_job'].replace(to_replace='never', value=int(0), inplace=True)
df['last_new_job'].replace(to_replace='>4', value=int(5), inplace=True)
df['last_new_job'] = round(df['last_new_job'].fillna(df['last_new_job'].mean()))
df['last_new_job'] = df['last_new_job'].astype("int")


# ---------------adding missing values to gender with out changing the percentegs of each observation------
def random_gender() -> str:
    x = random.choices(['Male', 'Female', 'Other'], weights=(0.9, 0.085, 0.015), k=1)
    return x


for i in range(df.shape[0]):
    if pd.isna(df.iloc[i, 3]):
        df.iloc[i, 3] = random_gender()


# ---------------adding missing values to company_size with out changing the percentegs of each observation---------
def random_company_size() -> str:
    x = random.choices(['<10', '49-10', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'],
                       weights=(0.099045, 0.107725, 0.232327, 0.194619, 0.066834, 0.101938, 0.043784, 0.153727), k=1)
    return x


for i in range(df.shape[0]):
    if pd.isna(df.iloc[i, 9]):
        df.iloc[i, 9] = random_company_size()

# ---------------adding missing values to company_type----------------

df['company_type'].fillna("empty", inplace=True)

##-------------------------------------------- decritization-------------------------------------------------
###----------------(feature represention)----------------------- Convirt data to numerical data -----------------------------------
# -------------------------enrolled_university ----------------------------------
df['enrolled_university'].replace(to_replace='no_enrollment', value=int(1), inplace=True)
df['enrolled_university'].replace(to_replace='Full time course', value=int(3), inplace=True)
df['enrolled_university'].replace(to_replace='Part time course', value=int(2), inplace=True)
# ---------------------education_level-------------------------------------------
df['education_level'].replace(to_replace='Primary School', value=int(1), inplace=True)
df['education_level'].replace(to_replace='High School', value=int(2), inplace=True)
df['education_level'].replace(to_replace='Graduate', value=int(3), inplace=True)
df['education_level'].replace(to_replace='Masters', value=int(4), inplace=True)
df['education_level'].replace(to_replace='Phd', value=int(5), inplace=True)
# ------------------------company_size------------------------------------------
df['company_size'].replace(to_replace='<10', value=int(1), inplace=True)
df['company_size'].replace(to_replace='49-10', value=int(2), inplace=True)
df['company_size'].replace(to_replace='50-99', value=int(3), inplace=True)
df['company_size'].replace(to_replace='100-500', value=int(4), inplace=True)
df['company_size'].replace(to_replace='500-999', value=int(5), inplace=True)
df['company_size'].replace(to_replace='1000-4999', value=int(6), inplace=True)
df['company_size'].replace(to_replace='5000-9999', value=int(7), inplace=True)
df['company_size'].replace(to_replace='10000+', value=int(8), inplace=True)
# ----------------------------------experience----------------------------------
df['experience'].replace(to_replace='<1', value=int(0), inplace=True)
df['experience'].replace(to_replace='>20', value=int(21), inplace=True)
# -------------------------------gender-----------------------------------------
df['gender'].replace(to_replace='Male', value=int(1), inplace=True)
df['gender'].replace(to_replace='Female', value=int(0), inplace=True)
df['gender'].replace(to_replace='Other', value=int(2), inplace=True)
# ----------------------------------relevent_experience----------------------------------
df['relevent_experience'].replace(to_replace='No relevent experience', value=int(0), inplace=True)
df['relevent_experience'].replace(to_replace='Has relevent experience', value=int(1), inplace=True)
# ----------------------------------major_discipline----------------------------------
df['major_discipline'] = df['major_discipline'].map({'STEM': int(1), 'Business Degree': int(0),
                                                     'Arts': int(2), 'Humanities': int(5), 'No Major': int(3),
                                                     'Other': int(4)})
# -----------------------------------company_type--------------------------------------
df['company_type'] = df['company_type'].map({'Pvt Ltd': int(5), 'Funded Startup': int(1),
                                             'Early Stage Startup': int(0), 'Other': int(2),
                                             'Public Sector': int(3), 'NGO': int(4), 'empty': int(6)})

# ------------------------------------city------------------------------------------------
df['city'] = pd.Categorical(df['city'])
df['city'] = df['city'].astype('category').cat.codes

###-------------------------------------------feature extraction-------------------------------------------------------
# ------------------new feature "motivation"----------------
mean_of_TH = df['training_hours'].mean()

df['motivation'] = [1 if x > mean_of_TH else 0 for x in df['training_hours']]

# ------------------new feature "experience_indicator"----------------
df['experience_indicator'] = df['experience']

df['experience_indicator'] = df['experience_indicator'].replace(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
     12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# ------------------new feature "academic_indicator"----------------
df['academic_indicator'] = df['education_level']
df['academic_indicator'] = df['academic_indicator'].replace(
    [1, 2, 3, 4, 5],
    [0, 0, 1, 1, 1])
print(df['academic_indicator'])

# -----------showing the data with the new features-------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
df1 = df.head(3)
print(df1)

###------------------------------------------feature selection---------------------------------------------------------

# ------------------Wrapper Method--------------------------------------


###-----------------------------------------Dimensionality Reduction------------------------------------------------------

# ------------------------------------pca-------------------------------


######--------------------------------------------------PART 2----------------------------------------------###########################

x = df[['city', 'city_development_index', 'gender',
        'relevent_experience', 'enrolled_university', 'education_level',
        'major_discipline', 'experience', 'company_size', 'company_type',
        'last_new_job', 'training_hours', 'motivation', 'experience_indicator', 'academic_indicator'
        ]]
y = df['target']

missing_values_checks(df)
###------------------------------------train the model--------------------------------
##-----------------------K-fold cross-validation -------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#######################################################################################################################
#                                               Decision Trees
#######################################################################################################################


## ------------------Hyperparameter Tuning -----------------

# ----------------- critrion ------------------------

# critrion_list = ['entropy', 'gini']
# res = pd.DataFrame()
# for critrion in critrion_list:
#     model = DecisionTreeClassifier(criterion=critrion, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'critrion': critrion,
#                       'train_acc': roc_auc_score(y_train, model.predict(X_train)),
#                       'test_acc': roc_auc_score(y_test, model.predict(X_test))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['critrion'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['critrion'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.title('critrion')
# plt.show()


# ----------------max depth----------------


# max_depth_list = np.arange(1, 20, 1)
# res = pd.DataFrame()
# for max_depth in max_depth_list:
#     model = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'max_depth': max_depth,
#                       'train_acc': roc_auc_score(y_train, model.predict(X_train)),
#                       'test_acc': roc_auc_score(y_test, model.predict(X_test))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['max_depth'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.title('max_depth')
# plt.show()
#
# print(res.sort_values('test_acc', ascending=False))


# -----------------------max features--------------------------------

# max_featuress_list = np.arange(1, 16, 2)
# res = pd.DataFrame()
# for max_features in max_featuress_list:
#     model = DecisionTreeClassifier(criterion='entropy', max_depth=8, max_features=11, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'max_features': max_features,
#                       'train_acc': roc_auc_score(y_train, model.predict(X_train)),
#                       'test_acc': roc_auc_score(y_test, model.predict(X_test))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['max_features'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['max_features'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.title('max_features')
# plt.show()

# ---------------------------min_samples_leaf----------------------------------

# min_samples_leaf_list = np.arange(1, 30, 3)
# res = pd.DataFrame()
# for min_samples_leaf in min_samples_leaf_list:
#     model = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=8, min_samples_leaf=min_samples_leaf, random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'min_samples_leaf': min_samples_leaf,
#                       'train_acc': roc_auc_score(y_train, model.predict(X_train)),
#                       'test_acc': roc_auc_score(y_test, model.predict(X_test))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['min_samples_leaf'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['min_samples_leaf'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.title('samples_leaf')
# plt.show()


# -------------------------------min_samples_split---------------------------

# min_samples_split_list = np.arange(2, 200, 1)
# res = pd.DataFrame()
# for min_samples_split in min_samples_split_list:
#     model = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=8,min_samples_leaf=13,
#                                    min_samples_split=min_samples_split,random_state=42)
#     model.fit(X_train, y_train)
#     res = res.append({'min_samples_split': min_samples_split,
#                       'train_acc': roc_auc_score(y_train, model.predict(X_train)),
#                       'test_acc': roc_auc_score(y_test, model.predict(X_test))}, ignore_index=True)
#
# plt.figure(figsize=(13, 4))
# plt.plot(res['min_samples_split'], res['train_acc'], marker='o', markersize=4)
# plt.plot(res['min_samples_split'], res['test_acc'], marker='o', markersize=4)
# plt.legend(['Train accuracy', 'Test accuracy'])
# plt.title('samples_split')
# plt.show()


# ------------------The first model------------------------------

# model = DecisionTreeClassifier()
# y_pred = model.predict(X_test)
# print('DecisionTreeClassifier_roc_auc_score on test_set:')
# print(roc_auc_score(y_test, y_pred))
# print('DecisionTreeClassifier_roc_auc_score on train_set:')
# print(roc_auc_score(model.predict(X_train), y_train))


# params = {'max_leaf_nodes': list(range(2, 100,10)), 'min_samples_split': [3,6,9,12,15,18], 'max_features': [3,6,9,12,15],
#           'max_depth':list(range(5,50,5)), 'criterion': ['entropy', 'gini']}
# grid_search_cv = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
# grid_search_cv.fit(X_train, y_train)
# print(print(grid_search_cv.best_params_))


# ------------------The final model------------------------------


# model = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=15,
#                                min_samples_split=6,max_leaf_nodes=42, random_state=42)
# trained_model = model.fit(X_train, y_train)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('DecisionTreeClassifier_roc_auc_score on test_set:')
# print(roc_auc_score(y_test, y_pred))
# print('DecisionTreeClassifier_roc_auc_score on train_set:')
# print(roc_auc_score(model.predict(X_train), y_train))


# ----------------visualization-----------------

# plt.figure(figsize=(12, 8))
# plot_tree(model, filled=True, max_depth=2, fontsize=10, class_names=True)
# plt.show()

# ------------feature importance ------------------


# feature_importances = pd.Series(trained_model.feature_importances_, index=X_train.columns)
# feature_importances.nlargest(12).plot(kind='barh')
# plt.title('DecisionTreeClassifier feature_importances')
# plt.xlabel('feature_importances')
# plt.ylabel('feature')
# plt.show()


#######################################################################################################################
#                                            Artificial Neural Networks
#######################################################################################################################


# scaler = StandardScaler()
# X_train_s=scaler.fit(X_train)
#
# model = MLPClassifier()
# model.fit(X_train, y_train)
# params = {'hidden_layer_sizes': [x for x in itertools.product(np.arange(1,100,2), repeat=2)],
#           'max_iter': [100,200,400,600],
#           'learning_rate_init': [0.001],
#           'activation': ['relu'],
#           'alpha': [0.0001]}
# grid_search_cv = GridSearchCV(MLPClassifier(), params, verbose=1, cv=3)
# grid_search_cv.fit(X_train, y_train)
# print(grid_search_cv.best_params_)
#
#
# scaler = StandardScaler()
# X_train_s = scaler.fit(X_train)
#
# model = MLPClassifier()
# model.fit(X_train, y_train)
#
# print('MLPClassifier_roc_auc_score on train_set:')
# print(roc_auc_score(y_true=y_train.values.ravel(), y_score=model.predict(X_train)))
# print('MLPClassifier_roc_auc_score on test_set:')
# print(roc_auc_score(y_true=y_test.values.ravel(), y_score=model.predict(X_test)))
# print('MLPClassifier confusion_matrix on train_set:')
# print(confusion_matrix(y_true=y_train, y_pred=model.predict(X_train)))
#
# clf_ann = MLPClassifier(random_state=1)
# parameter_space = {'hidden_layer_sizes': [(40, 40), (150, 100), (200, 250)], 'activation': ['relu'],
#                    'max_iter': [75, 100, 200], 'learning_rate_init': [0.001, 0.01], 'early_stopping': [True]}
# scores = GridSearchCV(clf_ann, parameter_space, cv=10, verbose=3, n_jobs=-1).fit(X_train, y_train)
# print('---------------scores-----------')
# print(scores)
# s1 = scores
#
# s2 = zip(s1.cv_results_['mean_test_score'], s1.cv_results_['params'])
# s3 = sorted(s2, key=lambda x: x[0])
#
# s3 = s3[-10:]
# for i in reversed(s3):
#     print(f'{i[0]:.4f} {i[1]}')
#
# val_scores, train_scores = [], []
# for i in reversed(s3):
#     model = MLPClassifier(random_state=1, activation=i[1]['activation'], max_iter=i[1]['max_iter'],
#                           hidden_layer_sizes=i[1]['hidden_layer_sizes'],
#                           learning_rate_init=i[1]['learning_rate_init'])
#     s = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
#     val_scores.append(s['test_score'].mean())
#     train_scores.append(s['train_score'].mean())
#
# fig = plt.figure(figsize=(8, 5))
# ax = plt.axes()
# ax.plot(np.arange(1, 11), train_scores, label='Train Score')
# ax.plot(np.arange(1, 11), val_scores, label='Validation Score')
# plt.show()
#
# ax.set_ylim(0.88, 0.94)
# ax.set_xticks(np.arange(1, 11))
# ax.set_title("Decision Tree - Cross Validation")
# ax.legend(loc='lower right')
# ax.set_ylabel('Score')
# ax.set_xlabel('Parameters combination')
# plt.show()
#
# ann_clf = MLPClassifier(random_state=1, activation='relu', hidden_layer_sizes=4, learning_rate='adaptive',
#                         learning_rate_init=0.01, max_iter=200).fit(X_train, y_train)
#
# plt.plot(ann_clf.loss_curve_)
# plt.title('Chosen Model Loss Curve')
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
#
# clf_ann = MLPClassifier(random_state=1, activation='relu', hidden_layer_sizes=(40, 20, 10, 30),
#                         learning_rate_init=0.01, max_iter=200).fit(X_train, y_train)
# plot_confusion_matrix(clf_ann, X_test, y_test, values_format='d')

# ----------------------------------------------------------------------------------------------------
#
# train_accs = []
# test_accs = []
# for size_ in range(1, 100, 5):
#     print(f"size: {size_}")
#     model = MLPClassifier(hidden_layer_sizes=size_)
#     model.fit(scaler.transform(X_train), y_train)
#     train_acc = model.score(scaler.transform(X_train), y_train)
#     train_accs.append(train_acc)
#     test_acc = model.score(scaler.transform(X_test), y_test)
#     test_accs.append(test_acc)
#
#
#
# plt.figure(figsize=(7, 4))
# plt.plot(range(1, 100, 5), train_accs, label='Train')
# plt.plot(range(1, 100, 5), test_accs, label='Test')
# plt.legend()
# plt.xlabel('# neurons')
# plt.ylabel('auc_roc_score')
# plt.title('Tuning the number of neurons in 2 layers networks ')
# plt.show()
#
#
# # -----------------------------------------------------------------------------------------------
#
# train_accs = []
# test_accs = []
# for size_ in range(1, 100, 10):
#     print(f"size: {size_}")
#     model = MLPClassifier(hidden_layer_sizes=36,
#                           max_iter=size_)
#     model.fit(scaler.transform(X_train), y_train)
#     train_acc = model.score(scaler.transform(X_train), y_train)
#     train_accs.append(train_acc)
#     test_acc = model.score(scaler.transform(X_test), y_test)
#     test_accs.append(test_acc)
#
#
#
# plt.figure(figsize=(7, 4))
# plt.plot(range(1, 100, 10), train_accs, label='Train')
# plt.plot(range(1, 100, 10), test_accs, label='Test')
# plt.legend()
# plt.xlabel('# max_iter')
# plt.ylabel('auc_roc_score')
# plt.title('Tuning the number of iter')
# plt.show()
#
#
# # ---------------------------------------------------------------------------------------------------------------
#
# train_accs = []
# test_accs = []
# for size_ in [0.01, 0.001, 0.0001, 0.00001]:
#     print(f"size: {size_}")
#     model = MLPClassifier(random_state=1,
#                           hidden_layer_sizes=36,
#                           max_iter=11,
#                           learning_rate_init=size_,)
#     model.fit(scaler.transform(X_train), y_train)
#     train_acc = model.score(scaler.transform(X_train), y_train)
#     train_accs.append(train_acc)
#     test_acc = model.score(scaler.transform(X_test), y_test)
#     test_accs.append(test_acc)
#
#
#
# plt.figure(figsize=(7, 4))
# plt.plot([0.01, 0.001, 0.0001, 0.00001], train_accs, label='Train')
# plt.plot([0.01, 0.001, 0.0001, 0.00001], test_accs, label='Test')
# plt.legend()
# plt.xlabel('# learning_rate_init')
# plt.ylabel('auc_roc')
# plt.title('Tuning the number of learning_rate_init')
# plt.show()
#
#
# model = MLPClassifier(hidden_layer_sizes=(30,30,40,10),
#                       max_iter=55,
#                       learning_rate_init=0.01,
#                       )
# model.fit(X_train, y_train)
#
# y_pred=model.predict(X_train)
# print('MLPClassifier_roc_auc_score on train_set:')
# print(roc_auc_score(y_true=y_train.values.ravel(), y_score=model.predict(X_train)))
# print('MLPClassifier_roc_auc_score on test_set:')
# print(roc_auc_score(y_true=y_test.values.ravel(), y_score=model.predict(X_test)))
# print('MLPClassifier confusion_matrix on train_set:')
# print(confusion_matrix(y_true=y_train, y_pred=model.predict(X_train)))
#
#
#
# model = MLPClassifier(hidden_layer_sizes=(10,40,40,20),
#                       max_iter=55,
#                       learning_rate_init=0.001, alpha=0.0001, activation= 'logistic' )
# model.fit(X_train, y_train)
# y_pred=model.predict(X_train)
# print('MLPClassifier_roc_auc_score on train_set:')
# print(roc_auc_score(y_true=y_train.values.ravel(), y_score=model.predict(X_train)))
# print('MLPClassifier_roc_auc_score on test_set:')
# print(roc_auc_score(y_true=y_test.values.ravel(), y_score=model.predict(X_test)))
# print('MLPClassifier confusion_matrix on train_set:')
# print(confusion_matrix(y_true=y_train, y_pred=model.predict(X_train)))


# params = {'hidden_layer_sizes': [x for x in itertools.product((10,20,30,40,50,100),repeat=4)],
#           'max_iter': [5,15,25,35, 45, 55,65,75],
#           'learning_rate_init': [0.01, 0.001],
#           'activation': ['identity', 'logistic', 'tanh', 'relu'],
#           'early_stopping': [True, False],
#           'alpha': [0.0001, 0.00001]}
# grid_search_cv = RandomizedSearchCV(MLPClassifier(), params, verbose=1, cv=3)
# grid_search_cv.fit(X_train, y_train)
# print(grid_search_cv.best_params_)


#######################################################################################################################
#                                                    SVM
#######################################################################################################################

# from sklearn.svm import LinearSVC, SVC
#
# model = LinearSVC()
# model.fit(X_train, y_train)
#
#
# # params = {'C': np.arange(1, 20, 0.1), 'loss': ['squared_hinge']}
# # grid_search = GridSearchCV(model, params, refit=True, verbose=3)
# # grid_search.fit(X_train, y_train)
# # print(grid_search.best_params_)
#
#
# # params = {'C': [9.3], 'max_iter': np.arange(628, 1200, 1),'loss': ['squared_hinge']}
# # grid_search = GridSearchCV(model, params, refit=True, verbose=3)
# # grid_search.fit(X_train, y_train)
# # print(grid_search.best_params_)
#
#
# model = LinearSVC()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('SVM roc_auc_score on test_set:', roc_auc_score(y_test, y_pred))
# print('SVM roc_auc_score on train_set:', roc_auc_score(y_train, model.predict(X_train)))
#
#
#
# model = LinearSVC(C=9.3, loss='squared_hinge', max_iter=1000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('SVM roc_auc_score on test_set:', roc_auc_score(y_test, y_pred))
# print('SVM roc_auc_score on train_set:', roc_auc_score(y_train, model.predict(X_train)))
#
#
# print('w = ', model.coef_)
# print('b = ', model.intercept_)
#
#
# def f_importances(coef, names):
#     imp = coef
#     imp, names = zip(*sorted(zip(imp, names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.xlabel('feature_importances')
#     plt.ylabel('feature')
#     plt.title('SVM feature_importances')
#     plt.show()
#
#
#
# features_names = ['city', 'city_development_index', 'gender',
#                   'relevent_experience', 'enrolled_university', 'education_level',
#                   'major_discipline', 'experience', 'company_size', 'company_type',
#                   'last_new_job', 'training_hours', 'motivation', 'experience_indicator', 'academic_indicator']
#
#
# model = LinearSVC(C=9.3, loss='squared_hinge', max_iter=1000)
# model.fit(X_train, y_train)
#
# for i in model.coef_:
#     f_importances(i, features_names)


#######################################################################################################################
#                                Unsupervised Learning - Clustering


########################################################################################################################
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# todo: change x below
# todo: download pyclustering package using the command (in the terminal) : pip install pyclustering
x = X_train.copy()
u = list(x.columns)

metric = distance_metric(type_metric.GOWER, max_range=x.max(axis=0))

dbi_list = []
sil_list = []

max_n_clusters = 8

# # this part should take you about 40-60 minutes of calculations (maybe more - depends on your computer)
for n_clusters in tqdm(range(2, max_n_clusters, 1)):
    initial_medoids = kmeans_plusplus_initializer(x, n_clusters).initialize(return_index=True)
    kmedoids_instance = kmedoids(data=x, initial_index_medoids=initial_medoids, metric=metric)
    kmedoids_instance.process()

    assignment = kmedoids_instance.predict(x)
    sil = silhouette_score(x, assignment)
    dbi = davies_bouldin_score(x, assignment)
    dbi_list.append(dbi)
    sil_list.append(sil)

plt.plot(range(2, max_n_clusters, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, max_n_clusters, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()

# from kmodes.kmodes import KModes
# # random categorical data
#
#
# km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
#
# clusters = km.fit_predict(X_train)
#
# # Print the cluster centroids
# print(km.cluster_centroids_)
#
#


# from sklearn_extra.cluster import KMedoids
# X_train_cluster = X_train.copy()
# kmedoid=KMedoids(n_clusters=6,random_state=0).fit(X_train_cluster)
# labels=kmedoid.labels_
# print(labels)
# unique_labels = set(labels)
# colors = [
#     plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
# ]

# for k, col in zip(unique_labels, colors):
#
#     class_member_mask = labels == k
#
#     xy = X[class_member_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=6,
#     )
#
# plt.plot(
#     cobj.cluster_centers_[:, 0],
#     cobj.cluster_centers_[:, 1],
#     "o",
#     markerfacecolor="cyan",
#     markeredgecolor="k",
#     markersize=6,
# )
#
# plt.title("KMedoids clustering. Medoids are represented in cyan.")
#


# from pyclustering.cluster.kmedoids import kmedoids
# from pyclustering.cluster import cluster_visualizer
# import pyclustering.core.kmedoids_wrapper as wrapper
# from pyclustering.core.wrapper import ccore_library
# from pyclustering.core.metric_wrapper import metric_wrapper
#
# # from pyclustering.utils import read_sample
#
#
# # Load list of points for cluster analysis.
# X_train_cluster = X_train.copy()
# # Set random initial medoids.
# initial_medoids = [0, 1]
# # Create instance of K-Medoids algorithm.
# # create Minkowski distance metric with degree equals to '2'
# metric = distance_metric(type_metric.MINKOWSKI, degree=2)
# kmedoids_instance = kmedoids(data=X_train_cluster,
#                                       initial_index_medoids=initial_medoids)
# kmedoids_instance = kmedoids.__init__(self=kmedoids_instance, data=X_train_cluster,
#                                       initial_index_medoids=initial_medoids, tolerance=0.001, ccore=True,
#                                       metric=metric)
# # Run cluster analysis and obtain results.
# kmedoids_instance.process(self=kmedoids_instance)
# clusters = kmedoids_instance.get_clusters()
# # Show allocated clusters.
# print(clusters)
# # Display clusters.
# visualizer = cluster_visualizer()
# visualizer.append_clusters(clusters, X_train_cluster)
# visualizer.show()
