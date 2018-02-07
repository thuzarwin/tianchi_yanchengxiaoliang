#encoding=UTF8
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import copy

# 路径设置为None就不会输出文件
predict_output_path = None

xgb_param = {'max_depth':4, 'eta':0.02, 'silent':1, 'objective':'reg:linear', "subsample": 0.8, "num_round": 1000,
             "early_stopping_rounds": 100}

train_origin = pd.read_csv("train_origin.csv")

ori_label_name = "sale_quantity"

'''-------------------------构造训练集，验证集-------------------'''
print "-------------------------构造训练集，验证集-------------------"
# 建模方式如下：
# 特征是历史的销售报表，标签是这个月的销售量。就像作为销售经理，用当月报表预测下个月汽车销量一样。
# “历史”这个词可以有多种跨度：上个月，过去半年，过去一年，过去三年，历史全部。需要注意有些车型没有上市这么长时间（最短只上了2个月）
# 暂时只考虑过去一个月的销售情况，构造特征的时候可以增加上个月销量和其他信息的交叉特征
# 注意验证集构造，可以用原始训练集最后一个月的销售情况作验证集，其中有4个上市两个月的车型被排除在这个验证集外，可以最后再单独考虑这四个车型。


# 预处理
# tmp_df = pd.DataFrame()
# class_id_lst = train_origin["class_id"].unique()
# for class_id in class_id_lst:
#     sales_tmp = train_origin[(train_origin["class_id"] == class_id)]
#     sales = sales_tmp[ori_label_name]
#     sales_std = np.std(sales)
#     sales_avg = np.mean(sales)
#     train_filter = sales_tmp[((sales_tmp[ori_label_name] >= sales_avg - 2 * sales_std) &
#                              (sales_tmp[ori_label_name] <= sales_avg + 2 * sales_std)) |
#                              (sales_tmp["sale_date"] >= 69) | (sales_tmp["sale_date"] == 58)]
#     tmp_df = pd.concat([tmp_df, train_filter])
# print "过滤前的训练集长度：%d" % train_origin.shape[0]
# train_origin = tmp_df.copy()
# print "过滤后的训练集长度：%d" % train_origin.shape[0]
# del tmp_df

train_X = train_origin[train_origin["label"] >= 0].copy()
train_watch_list_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 10)
                                & (train_origin["sale_date"] == 22) & (train_origin["sale_date"] == 34)
                                & (train_origin["sale_date"] == 46) & (train_origin["sale_date"] == 58)].copy()

print train_X.columns
train_y = train_X["label"]
train_watch_list_y = train_watch_list_X["label"]
test_X = train_origin[train_origin["sale_date"] == 70].copy()
class_id = test_X["class_id"]
# 构造第1个验证集
cv1_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69)].copy()
cv1_train_y = cv1_train_X["label"]
cv1_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 69)].copy()
cv1_test_y = cv1_test_X["label"]
# 构造第2个验证集
cv2_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 70) & (train_origin["sale_date"] != 58)].copy()
cv2_train_y = cv2_train_X["label"]
cv2_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 58)].copy()
cv2_test_y = cv2_test_X["label"]
# 构造第3个验证集
cv3_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69) & (train_origin["sale_date"] != 58)].copy()
cv3_train_y = cv3_train_X["label"]
cv3_test_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] == 69)].copy()
cv3_test_y = cv3_test_X["label"]
# 构造第4个验证集
cv4_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69) & (train_origin["sale_date"] != 58)].copy()
cv4_train_y = cv4_train_X["label"]
cv4_test_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] == 58)].copy()
cv4_test_y = cv4_test_X["label"]
# 构造第5个验证集
cv5_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69)].copy()
cv5_train_y = cv5_train_X["label"]
cv5_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 69) &
                          (train_origin["label"] < 1000)].copy()
cv5_test_y = cv5_test_X["label"]
# 构造第6个验证集
cv6_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 70) & (train_origin["sale_date"] != 58)].copy()
cv6_train_y = cv6_train_X["label"]
cv6_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 58) &
                          (train_origin["label"] < 1000)].copy()
cv6_test_y = cv6_test_X["label"]
# 构造第7个验证集
cv7_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69) & (train_origin["sale_date"] != 58)].copy()
cv7_train_y = cv7_train_X["label"]
cv7_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 69) &
                          (train_origin["label"] < 1000)].copy()
cv7_test_y = cv7_test_X["label"]
# 构造第8个验证集
cv8_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69) & (train_origin["sale_date"] != 58)].copy()
cv8_train_y = cv8_train_X["label"]
cv8_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 58) &
                          (train_origin["label"] < 1000)].copy()
cv8_test_y = cv8_test_X["label"]
# 构造第9个验证集
cv9_train_X = train_origin[(train_origin["label"] >= 0) &
                          (train_origin["sale_date"] < 69) & (train_origin["sale_date"] != 58)].copy()
cv9_train_y = cv9_train_X["label"] - cv9_train_X["sales"]
cv9_test_X = train_origin[(train_origin["label"] >= 0) & (train_origin["sale_date"] == 58) &
                          (train_origin["label"] < 1000)].copy()
cv9_test_y = cv9_test_X["label"]



'''---------------------训练模型-------------------'''
print "---------------------训练模型-------------------"
# 用训练集训练的模型
# feature_used = ["sales", "car_length_label_mean", "if_charging_count", "rated_passenger_mean", "cylinder_number_mean",
#                "TR_count", "min_price_mean", "fuel_type_id_count", "driven_type_id_count", "gearbox_type_count",
#                "newenergy_type_id_count", "min_price_label_mean", "car_height_mean", "engine_torque_mean",
#                "rated_passenger_label_mean", "displacement_count", "displacement_mean", "max_price_count",
#                "car_length_mean", "rated_passenger_count", "max_price_median", "min_price_median", "avg_price_median",
#                "sales_is_increase", "is_price_smaller_min_price", "is_price_in_price_level",
#                'type_id_1_label_sum','type_id_3_label_sum', 'newenergy_1_increase','newenergy_2_increase',
#                'newenergy_4_increase', 'engine_torque_history_mean','cylinder_number_history_mean',
#                'car_length_history_mean','equipment_quality_history_mean','car_height_history_mean',
#                'compartment_history_mean']
# dtrain = xgb.DMatrix(train_X[feature_used].values, train_y - train_X["sales"])
# dtest = xgb.DMatrix(test_X[feature_used].values)
# dwatch = xgb.DMatrix(train_watch_list_X[feature_used].values, train_watch_list_y)
# watchlist = [(dtrain, "watch_set")]
# bst = xgb.train(xgb_param, dtrain, xgb_param['num_round'], watchlist, verbose_eval=10)
# test_y = bst.predict(dtest) + test_X["sales"]
# result_df = train_origin[train_origin["sale_date"] == 70][["class_id"]].copy()
# result_df["predict_quantity"] = test_y
# result_df["predict_date"] = "201711"
# result_df["predict_quantity"] = result_df["predict_quantity"].apply(lambda x: int(x))
# result_df["class_id"] = result_df["class_id"].apply(lambda x: int(x))
# if predict_output_path:
#     result_df[["predict_date", "class_id", "predict_quantity"]].to_csv(predict_output_path, index=None)
# feature_score = bst.get_score(importance_type='gain')
# feature_score_df = pd.DataFrame()
# feature_score_df["feature_name"] = train_X[feature_used].columns
# feature_score_df["score"] = 0
# print feature_score
# for idx in range(len(feature_score_df["feature_name"])):
#     name = train_X[feature_used].columns[idx]
#     if "f%d" % idx in feature_score:
#         feature_score_df.loc[idx, "score"] = feature_score["f%d" % idx]
# print feature_score_df.sort_values(["score"], ascending=False)
# corr_df = train_X[["sales"]].copy()
# corr_df["label"] = train_y
# print corr_df[["label", "sales"]].corr()


def cross_validation(feature_tmp):
    # cv_train_X_lst = [cv1_train_X, cv2_train_X, cv3_train_X, cv4_train_X]
    # cv_train_y_lst = [cv1_train_y, cv2_train_y, cv3_train_y, cv4_train_y]
    # cv_test_X_lst = [cv1_test_X, cv2_test_X, cv3_test_X, cv4_test_X]
    # cv_test_y_lst = [cv1_test_y, cv2_test_y, cv3_test_y, cv4_test_y]
    cv_train_X_lst = [cv8_train_X]
    cv_train_y_lst = [cv8_train_y]
    cv_test_X_lst = [cv8_test_X]
    cv_test_y_lst = [cv8_test_y]
    error_lst = []
    for i in range(len(cv_train_X_lst)):
        print "========第 %d 次交叉验证==========" % i
        dtrain_cv = xgb.DMatrix(cv_train_X_lst[i][feature_tmp].values, cv_train_y_lst[i])
        dtest_cv = xgb.DMatrix(cv_test_X_lst[i][feature_tmp].values, cv_test_y_lst[i])
        watchlist_cv = [(dtest_cv, "test_cv")]
        model_cv = xgb.train(xgb_param, dtrain_cv, xgb_param['num_round'], watchlist_cv, verbose_eval=10)
        result = np.array(model_cv.predict(dtest_cv))
        error_lst.append(np.sqrt(np.mean(np.power(result - cv_test_y_lst[i], 2))))

        model_cv.get_score(importance_type='gain')
        feature_score = model_cv.get_score(importance_type='gain')
        # 计算特征的信息增益
        feature_score_df = pd.DataFrame()
        feature_score_df["feature_name"] = cv_train_X_lst[i][feature_tmp].columns
        feature_score_df["score"] = 0
        for idx in range(len(feature_score_df["feature_name"])):
            if "f%d" % idx in feature_score:
                feature_score_df.loc[idx, "score"] = feature_score["f%d" % idx]
        print feature_score_df.sort_values(["score"], ascending=False)
        # 计算特征和标签的相关性
        # cv1_corr_train = cv_train_X_lst[i][["sales"]].copy()
        # cv1_corr_train["label"] = cv_train_y_lst[i]
        # cv1_corr_test = cv_test_X_lst[i][["sales"]].copy()
        # cv1_corr_test["label"] = cv_test_y_lst[i]
        # print cv1_corr_train[["label", "sales"]].corr()
        # print cv1_corr_test[["label", "sales"]].corr()
        # print model_cv.predict(dtest_cv)
    return error_lst

# feature_tmp = ["sales", "car_length_label_mean", "if_charging_count", "rated_passenger_mean", "cylinder_number_mean",
#                "TR_count", "min_price_mean", "fuel_type_id_count", "driven_type_id_count", "gearbox_type_count",
#                "newenergy_type_id_count", "min_price_label_mean", "car_height_mean", "engine_torque_mean",
#                "rated_passenger_label_mean", "displacement_count", "displacement_mean", "max_price_count",
#                "car_length_mean", "rated_passenger_count", "max_price_median", "min_price_median", "avg_price_median",
#                "sales_is_increase", "is_price_smaller_min_price", "is_price_in_price_level",
#                'type_id_1_label_sum','type_id_3_label_sum', 'newenergy_1_increase','newenergy_2_increase',
#                'newenergy_4_increase', 'engine_torque_history_mean','cylinder_number_history_mean',
#                'car_length_history_mean','equipment_quality_history_mean','car_height_history_mean',
#                'compartment_history_mean']
# error_lst = cross_validation(feature_tmp=feature_tmp)
# print error_lst


# 如果增加一组特征，将集合内所有特征组合都进行交叉验证
# def all_subset(lst):
#     result = []
#     n = len(lst)
#     for i in range(2 ** n):
#         e = list(bin(i))[2:]
#         e = np.array(e) == '1'
#         result.append(lst[n - len(e):][e])
#     return result
# feature_append_names = ["sales_predict_linear", "sales_predict_ridge", "sales_predict_lasso",
#                         "price_predict_linear", "price_predict_ridge", "price_predict_lasso",
#                         "sales", "price_mean"]
# feature_append_lst = all_subset(np.array(feature_append_names))
# with open("feature_subset.txt", "a") as f:
#     for feature_append in feature_append_lst:
#         feature_tmp2 = copy.copy(feature_tmp)
#         feature_tmp2.extend(feature_append)
#         error_lst = cross_validation(feature_tmp2)
#         print_names = ["'" + name + "'" for name in feature_append]
#         print_error = [str(error) for error in error_lst]
#         f.write(",".join(print_names) + "\n")
#         f.write("[" + ",".join(print_error) + "]" + "\n")


# 选择特征
# 从少到多
feature_lst = []
prediction_lst = []
feature_tmp = []
with open("feature_used.txt") as f:
    feature_tmp = f.readline().split(",")
feature_used = feature_tmp
for i in range(len(feature_used) - 1):
    best_score = 10000
    best_feature = None
    print "=========start round %d=========" % i
    for feature in feature_used:
        feature_tmp = []
        if len(feature_lst) == 0:
            feature_tmp.append(feature)
        else:
            feature_tmp = copy.copy(feature_lst[-1])
            feature_tmp.append(feature)
        dtrain_cv = xgb.DMatrix(cv9_train_X[feature_tmp].values, cv9_train_y)
        dtest_cv = xgb.DMatrix(cv9_test_X[feature_tmp].values, cv9_test_y)
        # watchlist_cv = [(dtest_cv, "test_cv")]
        model_cv = xgb.train(xgb_param, dtrain_cv, xgb_param['num_round'], verbose_eval=10)
        result = np.array(model_cv.predict(dtest_cv))
        result = result + cv9_test_X["sales"]
        if np.sqrt(np.average(np.power(result - cv9_test_y.values, 2))) < best_score:
            best_score = np.sqrt(np.average(np.power(result - cv9_test_y.values, 2)))
            best_feature = feature
    feature_add = []
    if len(feature_lst) == 0:
        feature_add.append(best_feature)
    else:
        feature_add = copy.copy(feature_lst[-1])
        feature_add.append(best_feature)
    feature_lst.append(feature_add)
    prediction_lst.append(best_score)
    feature_used.remove(best_feature)
    with open("./20170207/feature_selection1.txt", "a") as f:
        f.write(",".join(feature_add))
        f.write("\n")
        f.write(str(best_score))
        f.write("\n")

# 从多到少
feature_lst = [["sales", "car_length_label_mean", "if_charging_count", "rated_passenger_mean", "cylinder_number_mean",
               "TR_count", "min_price_mean", "fuel_type_id_count", "driven_type_id_count", "gearbox_type_count",
               "newenergy_type_id_count", "min_price_label_mean", "car_height_mean", "engine_torque_mean",
               "rated_passenger_label_mean", "displacement_count", "displacement_mean", "max_price_count",
               "car_length_mean", "rated_passenger_count", "max_price_median", "min_price_median", "avg_price_median",
               "sales_is_increase", "is_price_smaller_min_price", "is_price_in_price_level",
               'type_id_1_label_sum','type_id_3_label_sum', 'newenergy_1_increase','newenergy_2_increase',
               'newenergy_4_increase', 'engine_torque_history_mean','cylinder_number_history_mean',
               'car_length_history_mean','equipment_quality_history_mean','car_height_history_mean',
               'compartment_history_mean']]
for i in range(len(feature_used) - 1):
    best_score = 10000
    best_feature = None
    print "=========start round %d=========" % i
    for feature in feature_lst[-1]:
        feature_tmp = copy.copy(feature_lst[-1])
        feature_tmp.remove(feature)
        dtrain_cv = xgb.DMatrix(cv9_train_X[feature_tmp].values, cv9_train_y)
        dtest_cv = xgb.DMatrix(cv9_test_X[feature_tmp].values, cv9_test_y)
        # watchlist_cv = [(dtest_cv, "test_cv")]
        model_cv = xgb.train(xgb_param, dtrain_cv, xgb_param['num_round'], verbose_eval=10)
        result = np.array(model_cv.predict(dtest_cv))
        result = result + cv9_test_X["sales"]
        if np.sqrt(np.average(np.power(result - cv9_test_y.values, 2))) < best_score:
            best_score = np.sqrt(np.average(np.power(result - cv9_test_y.values, 2)))
            best_feature = feature
    feature_add = copy.copy(feature_lst[-1])
    feature_add.remove(best_feature)
    feature_lst.append(feature_add)
    feature_used.remove(best_feature)
    with open("./20170207/feature_selection3.txt", "a") as f:
        f.write(",".join(feature_add))
        f.write("\n")
        f.write(str(best_score))
        f.write("\n")