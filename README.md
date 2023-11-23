# Mini-Project
### WEATHER ANALYSIS
### AIM:
To Perform weather Analysis on the given dataset and save the data to a file.

### ALGORITHM
### STEP 1 
Read the given Dataset

### STEP 2
Perform Data Cleaning operations and Outlier Detection and Removal

### STEP 3
Perform Univariate Analysis and Multivariate Analysis.

### STEP 4
Apply ,Feature Encoding, Feature Scaling and Feature transformation and selection techniques to all the features of the data set.

### STEP 5
Apply data visualization techniques to identify the patterns of the data.

### Program And Output:

Importing necessary packages:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
read the data set:
```
df=pd.read_csv("weather.csv")
df
```


![202074920-223a336a-361e-440b-b31e-7cca3a1c8a53](https://github.com/divyakumars/Mini-Project/assets/119393621/4fd40274-537e-4bc7-bc68-c9131dd6c20b)
```
df.head()
```

![202075017-6ab7c3b0-cbe9-4f3e-9f6e-56d3fcdcd8be](https://github.com/divyakumars/Mini-Project/assets/119393621/176a3742-e9aa-4ce3-879e-555bd2843a33)
```
df.info()
```
![202075224-10709e9c-0f94-4a01-b056-47661fad990b](https://github.com/divyakumars/Mini-Project/assets/119393621/e360fbcb-dcda-4328-a20a-5b35c676abb5)
```
df.tail()
```
![202075270-aeb27ea2-cefb-45c4-a909-a60d8a9bb336](https://github.com/divyakumars/Mini-Project/assets/119393621/cf52c5f5-15f3-43c9-897d-ea22270e2798)
```
df.describe()
```
![202075365-f97cbbe5-31f6-4cff-9fd9-ebff97e80e50](https://github.com/divyakumars/Mini-Project/assets/119393621/bfe5c9a7-90c6-497c-aadf-87d1fe7d2c2f)

```
df.shape
```
![202075419-7e9ba26a-104f-45bb-b05e-04ca0eb3bdab](https://github.com/divyakumars/Mini-Project/assets/119393621/eed1f15b-1da1-40e4-93d4-6b59f1fa9cba)
```
df['weather'].value_counts()
```


![202075488-ee9c5852-bb17-45b0-984a-ecdc225e558d](https://github.com/divyakumars/Mini-Project/assets/119393621/c4a0eb0a-3b36-4a21-9e74-81c9455b551b)


LABEL ENCODER:
```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['wind'] = le.fit_transform(df['weather'])
df.head(10)
```


![202075544-2ff2b185-77a9-4c4e-a563-9ae28f9cbfc0](https://github.com/divyakumars/Mini-Project/assets/119393621/26172ea3-38ec-4196-ab00-3d0a23d664fa)

DATA CLEANING:

```
df.isnull().sum()

```
![202075615-c47bf868-a9e7-4c67-8730-0cabf28a9031](https://github.com/divyakumars/Mini-Project/assets/119393621/990a20da-ee01-490c-a124-4e2d6169aa77)

```
missing_percentage = (df.isnull().sum())/(df.shape[0])*100
missing_percentage
```



![202075672-562ded93-db62-40c9-82b7-22ffd1cbb391](https://github.com/divyakumars/Mini-Project/assets/119393621/083e64ed-1606-4b0c-9847-40cc94ebed8c)
```
df.duplicated().value_counts()
```
![202075848-abb5493b-bd9f-4f0c-9cc8-9be3065e9787](https://github.com/divyakumars/Mini-Project/assets/119393621/929eca98-d28b-4aed-82f9-a2c71d691c75)

UNIVARIATE ANALYSIS
```
sns.boxplot(y="wind",data=df)
```

![202076084-b5248e03-6180-4c82-8a94-a33c47486781](https://github.com/divyakumars/Mini-Project/assets/119393621/834cbceb-8a10-4a0a-876e-92d5fbafcdc1)
```
sns.countplot(y="weather",data=df)

```

![202076467-57071e75-3cc2-42f9-8c88-6d3fdcb09e57](https://github.com/divyakumars/Mini-Project/assets/119393621/4f60112f-82de-4513-8066-6dccd90b5f58)

```
sns.histplot(y="wind",data=df)
```


![202076570-7d0e696e-32eb-44a4-adc5-f50e0a78beff](https://github.com/divyakumars/Mini-Project/assets/119393621/d5444264-6783-4ac9-b73b-1967551cbdd3)


MULTIVARIATE ANALYSIS
```
sns.scatterplot(df['wind'],df['weather'])
```


![202077081-7637e653-4d25-4849-b2d0-3c2ebcbd90f9](https://github.com/divyakumars/Mini-Project/assets/119393621/f867303c-2ed8-4e04-8a37-6f11060dcfdd)
```
sns.barplot(data=df, x='wind', y='precipitation')
```

![202077186-363af693-a67a-490c-8d23-ed695fdf3330](https://github.com/divyakumars/Mini-Project/assets/119393621/f590d30c-ede6-44db-9875-e4ff0ef9ba5e)

```
df.corr()
```
![202077255-46a8858c-e616-4b90-84ab-d983e569436f](https://github.com/divyakumars/Mini-Project/assets/119393621/2ed861d3-26fb-4dd0-9d97-c12c317812bb)
```
sns.heatmap(df.corr(),annot=True)
```


![202077324-79db0496-ba14-41c2-afaf-4c1fddaab9de](https://github.com/divyakumars/Mini-Project/assets/119393621/826aafbf-f7a8-4aef-9895-83a809718f8f)


DATA VISUALIZATION:
```
plt.figure(figsize=(20, 7))
sns.lineplot(data=df, x='temp_min', y='temp_max')
plt.show()
```

![202077469-d20438b6-2b8e-4ee0-b845-0fee37ea002d](https://github.com/divyakumars/Mini-Project/assets/119393621/c783db96-896c-4c79-9427-5ada31386751)
```
sns.pointplot(x=df['temp_max'],y=df['temp_min'])
```

![202077572-280474f5-d721-4857-8009-82959be22820](https://github.com/divyakumars/Mini-Project/assets/119393621/84978953-0c6d-4067-b222-273ecebbde6c)
```
sns.kdeplot(x=df['wind'],data=df)
```

![202077649-689749e0-9b44-43de-9f90-48e45724fb2e](https://github.com/divyakumars/Mini-Project/assets/119393621/6900e24e-10ca-4d28-bc98-745c9e1cef0e)
```
sns.countplot(y="precipitation",data=df)
```

![202077705-a325ad73-9ac3-4ca4-a694-607793f5e798](https://github.com/divyakumars/Mini-Project/assets/119393621/2acb2f7b-3040-45a7-9694-7d5c7f18c1a7)


### Result:
Hence the program to analyze the data set using data science is applied sucessfully.




















