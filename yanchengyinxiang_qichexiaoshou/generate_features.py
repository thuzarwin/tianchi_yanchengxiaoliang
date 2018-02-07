#encoding=UTF8
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import copy
from sklearn import linear_model
import math

# 路径设置为None就不会输出文件
predict_output_path = "./20180121/result2.csv"

xgb_param = {'max_depth':4, 'eta':0.02, 'silent':1, 'objective':'reg:linear', "subsample": 0.8, "num_round": 1000,
             "early_stopping_rounds": 100}

# 读取训练数据
origin_train_df = pd.read_csv("[new] yancheng_train_20171226.csv")
# 读取预测数据
origin_test_df = pd.read_csv("yancheng_testA_20171225.csv")


'''-------------------------预处理-----------------------'''
print "-------------------------预处理-----------------------"
# 预处理
def charge_price_level(level):
    if "-" in level:
        # 价格区间
        min_price = int(level.split("-")[0])
        max_price = int(level.split("-")[1][:-1])
        return (min_price, max_price, (min_price + max_price) / 2.0)
    else:
        # 低于……价格
        return (0, int(level[:-2]), float(level[:-2]) / 2)


def charge_power(power):
    power = str(power)
    if power == "-":
        return -1
    if "/" in power:
        # 分数表示法（两种类型）
        return (float(power.split("/")[0]) + float(power.split("/")[1])) / 2
    else:
        return float(power)


def charge_rated_passenger(value):
    value = str(value)
    if "-" in value:
        return (float(value.split("-")[1]) + float(value.split("-")[0])) / 2
    else:
        return int(value)


def charge_price(price):
    if price == '-':
        return np.NaN
    else:
        return float(price)

origin_train_df["level_id"] = origin_train_df["level_id"].apply(lambda x: 0 if x == '-' else int(x))
origin_train_df["min_price"] = origin_train_df["price_level"].apply(lambda x: charge_price_level(x)[0])
origin_train_df["max_price"] = origin_train_df["price_level"].apply(lambda x: charge_price_level(x)[1])
origin_train_df["avg_price"] = origin_train_df["price_level"].apply(lambda x: charge_price_level(x)[2])
origin_train_df["driven_type_id"] = origin_train_df["driven_type_id"].apply(lambda x: str(x))
origin_train_df["fuel_type_id"] = origin_train_df["fuel_type_id"].apply(lambda x: str(x))
origin_train_df["newenergy_type_id"] = origin_train_df["newenergy_type_id"].apply(lambda x: str(x))
origin_train_df["emission_standards_id"] = origin_train_df["emission_standards_id"].apply(lambda x: str(x))
origin_train_df["power"] = origin_train_df["power"].apply(charge_power)
origin_train_df["engine_torque"] = origin_train_df["engine_torque"].apply(charge_power)
origin_train_df["rated_passenger"] = origin_train_df["rated_passenger"].apply(charge_rated_passenger)
origin_train_df["price"] = origin_train_df["price"].apply(charge_price)


'''-------------------------构造特征--------------------'''
print "-------------------------构造特征--------------------"
# 标签列
ori_label_name = "sale_quantity"
# 需要聚合后计算均值的列
# 分成三类
# 第一类，和汽车外观有关："compartment", "car_length", "car_width", "car_height", "total_quality", "equipment_quality", "rated_passenger"
# 第二类，和汽车性能有关："power", "cylinder_number", "engine_torque", "wheelbase", "front_track", "rear_track"
# 第三类，和汽车品牌与价格有关："avg_price", "max_price", "min_price", "type_id", "level_id", "price"
mean_lst_names = ["sale_date", "class_id", "compartment", "displacement", "avg_price", "power",
                  "cylinder_number", "engine_torque", "car_length", "car_width", "car_height", "total_quality",
                  "equipment_quality", "rated_passenger", "wheelbase", "front_track", "rear_track", "max_price",
                  "min_price", "type_id", "level_id", "price"]

# 需要聚合后计算种类数的字段
#
count_lst_names = ["sale_date", "class_id", "compartment", "TR",
                   "gearbox_type", "displacement", "if_charging", "driven_type_id", "fuel_type_id",
                   "newenergy_type_id", "emission_standards_id", "rated_passenger",
                   "max_price", "price"]

# 加权平均的列
# 第一类，和汽车外观有关："compartment", "car_length", "car_width", "car_height", "total_quality", "equipment_quality", "rated_passenger"
# 第二类，和汽车性能有关："power", "cylinder_number", "engine_torque", "wheelbase", "front_track", "rear_track"
# 第三类，和汽车品牌与价格有关："max_price", "min_price", "avg_price"
weight_mean_lst_name = ["compartment", "displacement", "power", "cylinder_number", "engine_torque", "car_length", "car_width",
                "car_height", "total_quality", "equipment_quality", "rated_passenger", "wheelbase", "front_track",
                "rear_track", "max_price", "min_price", "avg_price"]
label_df = origin_train_df[["sale_date", "class_id", ori_label_name]].groupby(["sale_date", "class_id"]).sum()
# 取字段均值
mean_df = origin_train_df[mean_lst_names].groupby(["sale_date", "class_id"]).aggregate(np.mean)
# 取字段类型计数，例如在sale_date，class_id下有多少种箱型
count_df = origin_train_df[count_lst_names].groupby(["sale_date", "class_id"]).aggregate(
    lambda lst: len(set(lst.values)))
# 取字段和label的加权平均
weight_mean_lst_name_tmp = [name + "_label" for name in weight_mean_lst_name]
origin_train_tmp = origin_train_df.copy()
for name in weight_mean_lst_name:
    origin_train_tmp[name + "_label"] = origin_train_tmp[name] * origin_train_tmp[ori_label_name]
weight_mean_names = copy.copy(weight_mean_lst_name_tmp)
weight_mean_names.append(ori_label_name)
weight_mean_names.append("sale_date")
weight_mean_names.append("class_id")
weight_mean_df = origin_train_tmp[weight_mean_names].groupby(["sale_date", "class_id"]).sum().copy()
for name in weight_mean_lst_name_tmp:
    weight_mean_df[name] = weight_mean_df[name] / weight_mean_df[ori_label_name]

# bool型特征
bool_lst_name = ["if_MPV_id", "if_luxurious_id"]
bool_lst_tmp = copy.copy(bool_lst_name)
bool_lst_tmp.append("sale_date")
bool_lst_tmp.append("class_id")
bool_df = origin_train_df[bool_lst_tmp].groupby(["sale_date", "class_id"]).aggregate(np.average)

# bool特征和销量交叉
bool_label_lst_name = ["if_MPV_label", "if_luxurious_label"]
bool_label_df = origin_train_df[["sale_date", "class_id", "if_MPV_id", "if_luxurious_id"]].copy()
bool_label_df = bool_label_df.groupby(["sale_date", "class_id"]).aggregate(np.average)
bool_label_df["if_MPV_id"] = bool_label_df["if_MPV_id"].apply(lambda x: str(x))
bool_label_df["if_luxurious_id"] = bool_label_df["if_luxurious_id"].apply(lambda x: str(x))
bool_label_df.columns = bool_label_lst_name
bool_label_df = pd.get_dummies(bool_label_df)
bool_label_df[ori_label_name] = label_df[ori_label_name]
for column in bool_label_df.columns:
    bool_label_df[column] = bool_label_df[column] * bool_label_df[ori_label_name]
del bool_label_df[ori_label_name]
bool_label_lst_name = bool_label_df.columns.values

# 无对比关系的离散特征
category_lst_name = ["type_id", "level_id", "brand_id", "department_id"]
category_df = origin_train_df[category_lst_name].copy()
for item in category_lst_name:
    category_df[item] = category_df[item].apply(lambda field: str(field))
category_df = pd.get_dummies(category_df)
category_df["sale_date"] = origin_train_df["sale_date"]
category_df["class_id"] = origin_train_df["class_id"]
category_df = category_df.groupby(["sale_date", "class_id"]).aggregate(np.average)
category_lst_name = category_df.columns.values.tolist()

# 另一种离散特征处理方式，直接转成字符串格式传入，同一个sale_date和class_id内该字符串只出现过一种
category_str_lst_name = ["type_id", "level_id", "brand_id", "department_id"]
category_str_df = origin_train_df[category_str_lst_name].copy()
category_str_df["sale_date"] = origin_train_df["sale_date"]
category_str_df["class_id"] = origin_train_df["class_id"]
category_str_df = category_str_df.groupby(["sale_date", "class_id"]).aggregate(lambda series: str(series.values[0]))

# 针对聚合后不唯一的离散特征，将该离散特征和销量交叉生成新的特征，原始特征和交叉销量特征分别求和，计算各离散值对应销量在聚合后总销量里的占比
category_label_lst_name = ["fuel_type_id", "newenergy_type_id", "emission_standards_id", "type_id",
                           "level_id", "brand_id", "department_id", "gearbox_type", "if_charging"]
category_label_df = origin_train_df[category_label_lst_name].copy()
for name in category_label_lst_name:
    category_label_df[name] = category_label_df[name].apply(lambda field: str(field))
category_label_df = pd.get_dummies(category_label_df)
category_label_ratio_lst_name = category_label_df.columns.values
category_label_lst_name = category_label_df.columns.values
for name in category_label_lst_name:
    category_label_df[name + "_label"] = category_label_df[name] * origin_train_df[ori_label_name]
category_label_df["sale_date"] = origin_train_df["sale_date"]
category_label_df["class_id"] = origin_train_df["class_id"]
category_label_df = category_label_df.groupby(["sale_date", "class_id"]).aggregate(np.sum)
category_label_lst_name = category_label_df.columns.values
for i in range(len(category_label_ratio_lst_name)):
    ratio_name = category_label_ratio_lst_name[i] + "_label_ratio"
    label_name = category_label_ratio_lst_name[i] + "_label"
    category_label_df[ratio_name] = category_label_df[label_name] * 1.0 / label_df[ori_label_name]
category_label_ratio_lst_name = [name + "_label_ratio" for name in category_label_ratio_lst_name]

# 聚合后计算中位数的列
# 第一类，和汽车外观有关："compartment", "car_length", "car_width", "car_height", "total_quality", "equipment_quality", "rated_passenger"
# 第二类，和汽车性能有关："power", "cylinder_number", "engine_torque", "wheelbase", "front_track", "rear_track"
# 第三类，和汽车品牌与价格有关："max_price", "min_price", "avg_price"
def cal_median(series):
    tmp = series.values
    tmp.sort()
    return tmp[len(tmp) / 2]


median_lst_name = ["compartment", "displacement", "power", "cylinder_number", "engine_torque", "car_length", "car_width",
                "car_height", "total_quality", "equipment_quality", "rated_passenger", "wheelbase", "front_track",
                "rear_track", "max_price", "min_price", "avg_price"]
median_df = origin_train_df[median_lst_name].copy()
median_df["sale_date"] = origin_train_df["sale_date"]
median_df["class_id"] = origin_train_df["class_id"]
median_df = median_df.groupby(["sale_date", "class_id"]).aggregate(cal_median)

# 历史所有值聚合后均值，类型计数，加权平均，中位数
# 历史所有值的加权平均，权重POW(0.9, x)其中x为该车型的第几个月
history_median_df = origin_train_df[["sale_date", "class_id"]].copy()
history_mean_df = origin_train_df[["sale_date", "class_id"]].copy()
history_count_df = origin_train_df[["sale_date", "class_id"]].copy()
history_weight_mean_df = origin_train_df[["sale_date", "class_id"]].copy()
decay_mean_df = origin_train_df[["sale_date", "class_id"]].copy()
for column in median_lst_name:
    history_median_df[column] = np.NaN
for column in mean_lst_names[2:]:
    history_mean_df[column] = np.NaN
for column in count_lst_names[2:]:
    history_count_df[column] = np.NaN
for column in weight_mean_lst_name:
    history_weight_mean_df[column] = np.NaN
for column in weight_mean_lst_name:
    decay_mean_df[column] = np.NaN
history_median_df = history_median_df.groupby(["sale_date", "class_id"]).mean()
history_mean_df = history_mean_df.groupby(["sale_date", "class_id"]).mean()
history_count_df = history_count_df.groupby(["sale_date", "class_id"]).mean()
history_weight_mean_df = history_weight_mean_df.groupby(["sale_date", "class_id"]).mean()
decay_mean_df = decay_mean_df.groupby(["sale_date", "class_id"]).mean()
for (sale_date, class_id) in history_median_df.index:
    tmp = origin_train_df[(origin_train_df["sale_date"] <= sale_date) & (origin_train_df["class_id"] == class_id)].copy()
    sales_tmp = origin_train_df[(origin_train_df["sale_date"] <= sale_date) &
                                (origin_train_df["class_id"] == class_id)][ori_label_name]
    history_median_df.loc[(sale_date, class_id), :] = tmp[median_lst_name].apply(cal_median)
    history_mean_df.loc[(sale_date, class_id), :] = np.mean(tmp[mean_lst_names[2:]])
    history_count_df.loc[(sale_date, class_id), :] = tmp[count_lst_names[2:]].apply(lambda x: len(set(x)))
    weight_tmp = tmp.copy()
    for column in weight_mean_lst_name:
        weight_tmp[column] = weight_tmp[column] * sales_tmp
    history_weight_mean_df.loc[(sale_date, class_id), :] = np.sum(weight_tmp[weight_mean_lst_name]) / np.sum(sales_tmp)
    decay_tmp = tmp.copy()
    decay_tmp["date"] = decay_tmp["sale_date"].apply(lambda x: (x / 1000 - 2012) * 12 + x % 100)
    decay_tmp["date"] = np.max(decay_tmp["date"]) - decay_tmp["date"]
    decay_tmp["decay"] = np.exp(decay_tmp["date"] * math.log(0.9))
    for column in weight_mean_lst_name:
        decay_tmp[column] = decay_tmp[column] * decay_tmp["decay"]
    decay_mean_df.loc[(sale_date, class_id), :] = np.sum(decay_tmp[weight_mean_lst_name]) / np.sum(decay_tmp["decay"])



# # 聚合后计算最大值的列
# def cal_max(series):
#     tmp = series.values
#     return np.max(tmp)
#
# max_lst_name = ["price"]
# max_df = origin_train_df[max_lst_name].copy()
# max_df["sale_date"] = origin_train_df["sale_date"]
# max_df["class_id"] = origin_train_df["class_id"]
# max_df = max_df.groupby(["sale_date", "class_id"]).aggregate(cal_max)
#
#
# # 聚合后计算最小值的列
# def cal_min(series):
#     tmp = series.values
#     return np.min(tmp)
#
# min_lst_name = ["price"]
# min_df = origin_train_df[min_lst_name].copy()
# min_df["sale_date"] = origin_train_df["sale_date"]
# min_df["class_id"] = origin_train_df["class_id"]
# min_df = min_df.groupby(["sale_date", "class_id"]).aggregate(cal_min)

# 时间列
time_lst_name = ["month", "is_eleven", "is_ten", "year"]
time_df = origin_train_df[["sale_date", "class_id"]].copy()
time_df["sale_date"] = time_df["sale_date"].apply(lambda x: int(x))
time_df["month"] = time_df["sale_date"].apply(lambda date: date % 100)
time_df["year"] = time_df["sale_date"].apply(lambda date: date / 100)
time_df["is_eleven"] = time_df["sale_date"].apply(lambda date: 1 if (date % 100 == 11) else 0)
time_df["is_ten"] = time_df["sale_date"].apply(lambda date: 1 if (date % 100 == 10) else 0)
time_df = time_df.groupby(["sale_date", "class_id"]).aggregate(np.mean)

# 品牌偏好特征
brand_lst_name = ["brand_sales_one_month", "brand_prefer_one_month"]
brand_df = origin_train_df[["sale_date", "class_id", "brand_id"]].copy()
brand_df = brand_df.groupby(["sale_date", "class_id"]).aggregate(np.average)
brand_df["brand_sales_one_month"] = 0
brand_df["brand_prefer_one_month"] = 0
for sale_date, class_id in brand_df.index:
    brand_id = brand_df.loc[(sale_date, class_id), "brand_id"]
    tmp_df = origin_train_df[
        (origin_train_df["sale_date"] == sale_date)].copy()
    brand_sales = np.sum(tmp_df[tmp_df["brand_id"] == brand_id][ori_label_name])
    brand_df.loc[(sale_date, class_id), "brand_sales_one_month"] = brand_sales
    total_sales = sum(tmp_df[ori_label_name])
    brand_df.loc[(sale_date, class_id), "brand_prefer_one_month"] = brand_sales * 1.0 / total_sales

# 用户上个月对某品牌，车型级别，车型类别的偏好情况
prefer_lst_name = ["type_id", "brand_id", "level_id", "department_id"]
prefer_df = origin_train_df[prefer_lst_name].copy()
prefer_df["sale_date"] = origin_train_df["sale_date"]
prefer_df["class_id"] = origin_train_df["class_id"]
prefer_df = prefer_df.groupby(["sale_date", "class_id"], as_index=False).mean()
prefer_df["sales"] = label_df[ori_label_name].values
for column in prefer_lst_name:
    tmp = prefer_df[["sale_date", "sales", column]].groupby(["sale_date", column], as_index=False).sum()
    tmp = tmp.rename(columns={"sales": column + "_prefer"})
    tmp = pd.merge(tmp, prefer_df[["sale_date", "sales"]].groupby(["sale_date"], as_index=False).sum(),
                   how="left", on=["sale_date"])
    tmp[column + "_prefer"] = tmp[column + "_prefer"] / tmp["sales"]
    prefer_df = pd.merge(prefer_df, tmp[["sale_date", column, column + "_prefer"]], on=["sale_date", column], how="left")
prefer_df.index = label_df.index

# 大趋势，计算该车系，品牌，所有销量数较上个月是否增长，首先计算该车系，品牌等每个月的销量
trend_lst_name = ["type_id", "brand_id", "level_id", "department_id"]
trend_df = origin_train_df[trend_lst_name].copy()
trend_df["sale_date"] = origin_train_df["sale_date"]
trend_df["class_id"] = origin_train_df["class_id"]
trend_df = trend_df.groupby(["sale_date", "class_id"], as_index=False).mean()
trend_df["sales"] = label_df[ori_label_name].values
for column in trend_lst_name:
    tmp = trend_df[["sale_date", column, "sales"]].groupby(["sale_date", column], as_index=False).sum()
    tmp = tmp.rename(columns={"sales": column + "_sales"})
    trend_df = pd.merge(trend_df, tmp, on=["sale_date", column], how="left")
trend_df.index = label_df.index

# 每一款车型的上市时长
purchase_time_name = ["purchase_time"]
purchase_df = origin_train_df[["sale_date", "class_id"]].copy()
purchase_df["date"] = purchase_df["sale_date"].apply(lambda x: (x / 100 - 2012) * 12 + x % 100)
purchase_df = purchase_df.groupby(["sale_date", "class_id"], as_index=False).mean()
purchase_tmp = pd.DataFrame()
for class_id in purchase_df["class_id"].unique():
    tmp = purchase_df[purchase_df["class_id"] == class_id].copy()
    tmp["purchase_time"] = tmp["date"] - np.min(tmp["date"])
    purchase_tmp = pd.concat([purchase_tmp, tmp])
purchase_df = pd.merge(purchase_df, purchase_tmp, on=["sale_date", "class_id"], how="left")
purchase_df.index = label_df.index

# 每一车型下款式数量
item_name = "car_count"
item_df = origin_train_df[["sale_date", "class_id"]].copy()
item_df["car_count"] = 1
item_df = item_df.groupby(["sale_date", "class_id"]).sum()

# 整合
train_df = label_df.copy()
mean_train_names = [name + "_mean" for name in mean_lst_names[2:]]
history_mean_train_names = [name + "_history_mean" for name in mean_lst_names[2:]]
count_train_names = [name + "_count" for name in count_lst_names[2:]]
history_count_train_names = [name + "_history_count" for name in count_lst_names[2:]]
weight_mean_train_names = [name + "_mean" for name in weight_mean_lst_name_tmp]
history_weight_mean_train_names = [name + "_label_history_mean" for name in weight_mean_lst_name]
median_train_names = [name + "_median" for name in median_lst_name]
history_median_train_names = [name + "_history_median" for name in median_lst_name]
category_label_train_names = [name + "_sum" for name in category_label_lst_name]
decay_train_names = [name + "_decay" for name in weight_mean_lst_name]
prefer_train_names = [name + "_prefer" for name in prefer_lst_name]
trend_train_names = [name + "_sales" for name in trend_lst_name]
# max_train_names = [name + "_max" for name in max_lst_name]
# min_train_names = [name + "_min" for name in min_lst_name]
train_df[mean_train_names] = mean_df[mean_lst_names[2:]]
train_df[history_mean_train_names] = history_mean_df[mean_lst_names[2:]]
train_df[count_train_names] = count_df[count_lst_names[2:]]
train_df[history_count_train_names] = history_count_df[count_lst_names[2:]]
train_df[weight_mean_train_names] = weight_mean_df[weight_mean_lst_name_tmp]
train_df[history_weight_mean_train_names] = history_weight_mean_df[weight_mean_lst_name]
train_df[bool_lst_name] = bool_df[bool_lst_name]
train_df[bool_label_lst_name] = bool_label_df[bool_label_lst_name]
train_df[category_lst_name] = category_df[category_lst_name]
train_df[category_str_lst_name] = category_str_df[category_str_lst_name]
train_df[median_train_names] = median_df[median_lst_name]
train_df[history_median_train_names] = history_median_df[median_lst_name]
train_df[time_lst_name] = time_df[time_lst_name]
train_df[brand_lst_name] = brand_df[brand_lst_name]
train_df[item_name] = item_df[item_name]
train_df[category_label_train_names] = category_label_df[category_label_lst_name]
train_df[category_label_ratio_lst_name] = category_label_df[category_label_ratio_lst_name]
train_df[decay_train_names] = decay_mean_df[weight_mean_lst_name]
train_df[prefer_train_names] = prefer_df[prefer_train_names]
train_df[trend_train_names] = trend_df[trend_train_names]
train_df[purchase_time_name] = purchase_df[purchase_time_name]

# train_df[max_train_names] = max_df[max_lst_name]
# train_df[min_train_names] = min_df[min_lst_name]
train_df["sales"] = label_df[ori_label_name]
# 整合当前时间点以前第二个月的销量，售价，不同能源类型的汽车销量
train_df["sales2"] = np.NaN
train_df["price2"] = np.NaN
for i in range(1, 5, 1):
    train_df["newenergy_type_id_%d_label_sum2" % i] = np.NaN
for i in [1, 2, 3, 5]:
    train_df["emission_standards_id_%d_label_sum2" % i] = np.NaN
for i in range(1, 5, 1):
    train_df["level_id_%d_label_sum2" % i] = np.NaN
for name in prefer_lst_name:
    train_df[name + "_prefer2"] = np.NaN
for name in trend_lst_name:
    train_df[name + "_sales2"] = np.NaN
for sale_date, class_id in train_df.index:
    sale_date_trans = (sale_date / 100 - 2012) * 12 + sale_date % 100
    sale_date_trans = sale_date_trans - 1
    if sale_date_trans >= 0:
        sale_date_trans = (2012 + sale_date_trans / 12) * 100 + sale_date_trans % 12
        if (sale_date_trans, class_id) in label_df.index:
            train_df.loc[(sale_date, class_id), "sales2"] = label_df.loc[(sale_date_trans, class_id), ori_label_name]
            train_df.loc[(sale_date, class_id), "price2"] = train_df.loc[(sale_date_trans, class_id), "price_mean"]
            for i in range(1, 5, 1):
                train_df.loc[(sale_date, class_id), "newenergy_type_id_%d_label_sum2" % i] = \
                    train_df.loc[(sale_date_trans, class_id), "newenergy_type_id_%d_label_sum" % i]
            for i in [1, 2, 3, 5]:
                train_df.loc[(sale_date, class_id), "emission_standards_id_%d_label_sum2" % i] = \
                    train_df.loc[(sale_date_trans, class_id), "emission_standards_id_%d_label_sum" % i]
            for i in range(1, 5, 1):
                train_df.loc[(sale_date, class_id), "level_id_%d_label_sum2" % i] = \
                    train_df.loc[(sale_date_trans, class_id), "level_id_%d_label_sum" % i]
            for name in prefer_lst_name:
                train_df.loc[(sale_date, class_id), name + "_prefer2"] = \
                    train_df.loc[(sale_date_trans, class_id), name + "_prefer"]
            for name in trend_lst_name:
                train_df.loc[(sale_date, class_id), name + "_sales2"] = \
                    train_df.loc[(sale_date_trans, class_id), name + "_sales"]


def charge_sales_comparison(theta):
    if theta > 0:
        return 1
    elif theta < 0:
        return -1
    else:
        return theta
train_df["sales_is_increase"] = (train_df["sales"] - train_df["sales2"]).apply(charge_sales_comparison)
train_df["sales_increase"] = train_df["sales"] - train_df["sales2"]
train_df["sales_increase_abs"] = np.abs(train_df["sales"] - train_df["sales2"])
train_df["sales_increase_ratio"] = (train_df["sales"] - train_df["sales2"]) * 1.0 / train_df["sales2"]
train_df["sales_increase_ratio_abs"] = np.abs((train_df["sales"] - train_df["sales2"]) * 1.0 / train_df["sales2"])
train_df["price_is_increase"] = (train_df["price_mean"] - train_df["price2"]).apply(charge_sales_comparison)
train_df["price_increase"] = train_df["price_mean"] - train_df["price2"]
for i in range(1, 5, 1):
    train_df["newenergy_%d_is_increase" % i] = (train_df["newenergy_type_id_%d_label_sum" % i] -
                                        train_df["newenergy_type_id_%d_label_sum2" % i]).apply(charge_sales_comparison)
    train_df["newenergy_%d_increase" % i] = (train_df["newenergy_type_id_%d_label_sum" % i] -
                                            train_df["newenergy_type_id_%d_label_sum2" % i])
for i in [1, 2, 3, 5]:
    train_df["emission_%d_is_increase" % i] = (train_df["emission_standards_id_%d_label_sum" % i] -
                                    train_df["emission_standards_id_%d_label_sum2" % i]).apply(charge_sales_comparison)
    train_df["emission_%d_increase" % i] = (train_df["emission_standards_id_%d_label_sum" % i] -
                                               train_df["emission_standards_id_%d_label_sum2" % i])
for i in range(1, 5, 1):
    train_df["level_id_%d_is_increase" % i] = (train_df["level_id_%d_label_sum" % i] -
                                    train_df["level_id_%d_label_sum2" % i]).apply(charge_sales_comparison)
    train_df["level_id_%d_increase" % i] = (train_df["level_id_%d_label_sum" % i] -
                                               train_df["level_id_%d_label_sum" % i])
for name in prefer_lst_name:
    train_df[name + "_is_increase"] = (train_df[name + "_prefer"] -
                                       train_df[name + "_prefer2"]).apply(charge_sales_comparison)
    train_df[name + "_increase"] = train_df[name + "_prefer"] - train_df[name + "_prefer2"]
for name in trend_lst_name:
    train_df[name + "_sales_is_increase"] = (train_df[name + "_sales"] -
                                             train_df[name + "_sales2"]).apply(charge_sales_comparison)
    train_df[name + "_sales_increase"] = train_df[name + "_sales"] - train_df[name + "_sales2"]


# 增加price是否比price_level的最小值小，最大值大，正处于price_level之间
train_df["is_price_larger_max_price"] = (train_df["price_mean"] > train_df["max_price_mean"]).apply(lambda flag: 1 if flag else 0)
train_df["is_price_smaller_min_price"] = (train_df["price_mean"] < train_df["min_price_mean"]).apply(lambda flag: 1 if flag else 0)
train_df["is_price_in_price_level"] = ((train_df["price_mean"] < train_df["max_price_mean"]) & \
                                      (train_df["price_mean"] > train_df["min_price_mean"])).apply(lambda flag: 1 if flag else 0)

# 将id类特征one-hot编码，放到线性回归模型中，分别预测销量和价格
category_names = ["sale_date", "class_id", "brand_id", "type_id", "level_id", "department_id", "if_MPV_id",
                  "if_luxurious_id"]
# 训练预测销量的模型
train_category = origin_train_df[category_names].copy()
train_category["month"] = train_category["sale_date"].apply(lambda x: x % 100)
train_category["year"] = train_category["sale_date"].apply(lambda x: x / 100)
for column in train_category.columns:
    train_category[column] = train_category[column].apply(lambda x: str(x))
col_dummies = copy.copy(train_category.columns.values.tolist())
col_dummies.remove("sale_date")
train_category = pd.get_dummies(train_category[col_dummies])


# 将sale_date减一个月，为了方便后续结果的合并
def reduce_month(date):
    numeric_date = (date / 100 - 2012) * 12 + date % 100
    numeric_date -= 1
    # return numeric_date / 12 + 2012 + numeric_date % 12
    return numeric_date
train_category["sale_date"] = origin_train_df["sale_date"].apply(lambda x: reduce_month(x))
train_category["class_id"] = origin_train_df["class_id"]
train_category_sales_X = train_category.groupby(["sale_date", "class_id"]).mean()
train_category_sales_y = origin_train_df[["sale_date", "class_id", ori_label_name]].copy()
train_category_sales_y["sale_date"] = train_category_sales_y["sale_date"].apply(reduce_month)
train_category_sales_y = train_category_sales_y.groupby(["sale_date", "class_id"]).sum()
# 降了1个月后，所以取sale_date为201709的数据，实际上是201710的数据
test_category_sales_X = train_category[train_category["sale_date"] == 69].copy()
test_category_sales_X["sale_date"] = 70
test_category_sales_X["month_10"] = 0
test_category_sales_X["month_11"] = 1
test_category_sales_X = test_category_sales_X.groupby(["sale_date", "class_id"]).mean()
test_category_sales_X = pd.concat([train_category_sales_X, test_category_sales_X])
linear = linear_model.LinearRegression()
linear.fit(train_category_sales_X.values, train_category_sales_y.values)
result_linear = linear.predict(test_category_sales_X.values)
ridge = linear_model.Ridge(alpha=0.5)
ridge.fit(train_category_sales_X.values, train_category_sales_y.values)
result_ridge = ridge.predict(test_category_sales_X.values)
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(train_category_sales_X.values, train_category_sales_y.values)
result_lasso = lasso.predict(test_category_sales_X.values)
test_category_sales_X["sales_predict_linear"] = result_linear
test_category_sales_X["sales_predict_ridge"] = result_ridge
test_category_sales_X["sales_predict_lasso"] = result_lasso

# 训练预测价格的模型
price_df = origin_train_df[["sale_date", "class_id", "price"]].copy()
price_df["sale_date"] = price_df["sale_date"].apply(reduce_month)
price_df = price_df.groupby(["sale_date", "class_id"]).mean()
train_category_price_X = train_category.groupby(["sale_date", "class_id"]).mean()[~price_df["price"].isnull()]
train_category_price_y = price_df[~price_df["price"].isnull()]["price"]
# 降了1个月后，所以取sale_date为201709的数据，实际上是201710的数据
test_category_price_X = train_category[train_category["sale_date"] == 69].copy()
test_category_price_X["sale_date"] = 70
test_category_price_X["month_10"] = 0
test_category_price_X["month_11"] = 1
test_category_price_X = test_category_price_X.groupby(["sale_date", "class_id"]).mean()
test_category_price_X = pd.concat([train_category_price_X, test_category_price_X])
linear = linear_model.LinearRegression()
linear.fit(train_category_price_X.values, train_category_price_y.values)
result_linear = linear.predict(test_category_price_X.values)
ridge = linear_model.Ridge(alpha=0.5)
ridge.fit(train_category_price_X.values, train_category_price_y.values)
result_ridge = ridge.predict(test_category_price_X.values)
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(train_category_price_X.values, train_category_price_y.values)
result_lasso = lasso.predict(test_category_price_X.values)
test_category_price_X["price_predict_linear"] = result_linear
test_category_price_X["price_predict_ridge"] = result_ridge
test_category_price_X["price_predict_lasso"] = result_lasso

# 为每个样本添加标签
train_df = train_df.swaplevel(0, 1, axis=0)
train_df["index"] = train_df.index
train_df["sale_date"] = train_df["index"].apply(lambda item: item[1])
train_df["class_id"] = train_df["index"].apply(lambda item: item[0])
train_df["sale_date"] = train_df["sale_date"].apply(lambda date: (date / 100 - 2012) * 12 + date % 100)
del train_df["index"]
train_tmp = train_df.copy()
train_tmp["label"] = -1
# train_tmp["sales_predict_linear"] = np.NaN
# train_tmp["sales_predict_ridge"] = np.NaN
# train_tmp["sales_predict_lasso"] = np.NaN
# train_tmp["price_predict_linear"] = np.NaN
# train_tmp["price_predict_ridge"] = np.NaN
# train_tmp["price_predict_lasso"] = np.NaN
label_df = train_tmp[["sale_quantity", "sale_date", "class_id"]]
for index in train_df.index:
    date = (index[1] / 100 - 2012) * 12 + index[1] % 100
    class_id = index[0]
    label = label_df[(label_df["sale_date"] == date + 1) & (label_df["class_id"] == class_id)]["sale_quantity"].values
    if len(label) > 0:
        train_tmp.loc[index, "label"] = label[0]
    if (date, class_id) in test_category_sales_X.index:
        sales_linear = test_category_sales_X.loc[(date, class_id), "sales_predict_linear"]
        train_tmp.loc[index, "sales_predict_linear"] = sales_linear
        sales_ridge = test_category_sales_X.loc[(date, class_id), "sales_predict_ridge"]
        train_tmp.loc[index, "sales_predict_ridge"] = sales_ridge
        sales_lasso = test_category_sales_X.loc[(date, class_id), "sales_predict_lasso"]
        train_tmp.loc[index, "sales_predict_lasso"] = sales_lasso

    if (date, class_id) in test_category_price_X.index:
        price_linear = test_category_price_X.loc[(date, class_id), "price_predict_linear"]
        train_tmp.loc[index, "price_predict_linear"] = price_linear
        price_ridge = test_category_price_X.loc[(date, class_id), "price_predict_ridge"]
        train_tmp.loc[index, "price_predict_ridge"] = price_ridge
        price_lasso = test_category_price_X.loc[(date, class_id), "price_predict_lasso"]
        train_tmp.loc[index, "price_predict_lasso"] = price_lasso

# train_tmp[["price_predict_linear", "price_predict_ridge", "price_predict_lasso"]] = \
#     test_category_price_X[["price_predict_linear", "price_predict_ridge", "price_predict_lasso"]]
# train_tmp[["sales_predict_linear", "sales_predict_ridge", "sales_predict_lasso"]] = \
#     test_category_sales_X[["sales_predict_linear", "sales_predict_ridge", "sales_predict_lasso"]]

print train_tmp.head()


train_origin = pd.DataFrame(train_tmp.values)
train_origin.columns = train_tmp.columns

train_origin.to_csv("train_origin.csv", index=None)
