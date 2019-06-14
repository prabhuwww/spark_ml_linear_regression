# https://towardsdatascience.com/apache-spark-mllib-tutorial-ec6f1cb336a9
# https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning
# http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

#export PYSPARK_DRIVER_PYTHON="/usr/local/ipython/bin/ipython"
# export SPARK_HOME="/usr/local/spark/"

# Set a fixed value for the hash seed secret
#export PYTHONHASHSEED=0

# Set an alternate Python executable
#export PYSPARK_PYTHON=/usr/local/ipython/bin/ipython

# Augment the default search path for shared libraries
#export LD_LIBRARY_PATH=/usr/local/ipython/bin/ipython

# Augment the default search path for private libraries 
#export PYTHONPATH=$SPARK_HOME/python/lib/py4j-*-src.zip:$PYTHONPATH:$SPARK_HOME/python/

#import findspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression


# Build the SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("Linear Regression Model") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()

sc = spark.sparkContext

# Load in the data
rdd = sc.textFile('/Users/yourName/Downloads/CaliforniaHousing/cal_housing.data')
rdd.take(2)

# Split lines on commas
rdd = rdd.map(lambda line: line.split(","))
# Inspect the first 2 lines 
rdd.take(2)

# Load in the header
header = sc.textFile('/Users/yourName/Downloads/CaliforniaHousing/cal_housing.domain')
header.collect()


# Map the RDD to a DF
df = rdd.map(lambda line: Row(longitude=line[0], 
                              latitude=line[1], 
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5], 
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()


df.show()
# Print the schema of `df`
df.printSchema()

# All columns are still of data type string
# rectify this situation and assign “better” or more accurate data types to all columns
df = df.withColumn("longitude", df["longitude"].cast(FloatType())) \
   .withColumn("latitude", df["latitude"].cast(FloatType())) \
   .withColumn("housingMedianAge",df["housingMedianAge"].cast(FloatType())) \
   .withColumn("totalRooms", df["totalRooms"].cast(FloatType())) \ 
   .withColumn("totalBedRooms", df["totalBedRooms"].cast(FloatType())) \ 
   .withColumn("population", df["population"].cast(FloatType())) \ 
   .withColumn("households", df["households"].cast(FloatType())) \ 
   .withColumn("medianIncome", df["medianIncome"].cast(FloatType())) \ 
   .withColumn("medianHouseValue", df["medianHouseValue"].cast(FloatType()))

df.select('population','totalBedRooms').show(10)
df.groupBy("housingMedianAge").count().sort("housingMedianAge",ascending=False).show()

# Feature engineering - add features

# Divide `totalRooms` by `households`
roomsPerHousehold = df.select(col("totalRooms")/col("households"))

# Divide `population` by `households`
populationPerHousehold = df.select(col("population")/col("households"))

# Divide `totalBedRooms` by `totalRooms`
bedroomsPerRoom = df.select(col("totalBedRooms")/col("totalRooms"))

# Add the new columns to `df`
df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households")) \
   .withColumn("populationPerHousehold", col("population")/col("households")) \
   .withColumn("bedroomsPerRoom", col("totalBedRooms")/col("totalRooms"))

# Separate label and features - first column is the label rest

# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `df` with the new DataFrame
df = spark.createDataFrame(input_data, ["label", "features"])
   
# Normalize/standardize feature columns
# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(df)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(df)


# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8,.2],seed=1234)

# Import `LinearRegression`
from pyspark.ml.regression import LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the data to the model
linearModel = lr.fit(train_data)

# Save the model
model_1.save('my_cal_house_lin_model')  # Ensure correct path here

# Load my model
LinearRegression.load('my_cal_house_lin_model')   # Ensure correct path here


# Generate predictions
predicted = linearModel.transform(test_data)

# Extract the predictions and the "known" correct labels
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])

# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()

# Print out first 5 instances of `predictionAndLabel` 
predictionAndLabel[:5]

# Coefficients for the model
linearModel.coefficients

# Intercept for the model
linearModel.intercept

# Get the RMSE
linearModel.summary.rootMeanSquaredError

# Get the R2
linearModel.summary.r2

spark.stop()
