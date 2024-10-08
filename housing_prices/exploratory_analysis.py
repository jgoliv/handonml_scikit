from housing_prices.splitting_data import strat_train_set
housing = strat_train_set.copy()

# Visualizing Geographical Data
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# housing prices and population density are very much related to the location
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

housing.plot(
    kind="scatter", x="longitude", y="latitude", alpha=0.4
    , s=housing["population"]/100, label="population"
    , figsize=(10,7), c="median_house_value"
    , cmap=plt.get_cmap("jet"), colorbar=True
)
plt.legend()
plt.show()

# looking for correlations
corr_matrix = housing.drop("ocean_proximity", axis=1).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

"""
median house value ~ strong positive correlation with median_income
median house value ~ small negative correlation with latitude
"""

from pandas.plotting import scatter_matrix

attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age"
]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# Most promising attribute to predict the median house value is the median income
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# Creating more interesting attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.drop("ocean_proximity", axis=1).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

