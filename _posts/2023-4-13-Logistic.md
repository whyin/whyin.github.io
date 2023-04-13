로지스틱 회기 분류기 
---

로지스틱 회기 분석은 개별 클래스 집합에 대한 관찰을 예측하는 데 사용되는 지도 학습 분류 알고리즘입니다.분류 문제를 해결하는 데 사용되는 가장 단순하고 간단하며 다용도의 분류 알고리즘 중 하나입니다.

통계학에서 로지스틱 회귀 모형은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모형입니다. 즉, 관측치 집합이 주어지면 로지스틱 회귀 알고리즘을 사용하여 관측치를 두 개 이상의 이산 클래스로 분류할 수 있습니다. 따라서 대상 변수는 본질적으로 이산적입니다.

####  선형 방정식 구현

로지스틱 회귀 분석 알고리즘은 반응 값을 예측하기 위해 독립 변수 또는 설명 변수가 있는 선형 방정식을 구현하는 방식으로 작동합니다.

만약 우리가 하나의 설명 변수(x1)와 하나의 반응 변수(z)를 가지고 있다면, 선형 방정식은 다음과 같은 방정식으로 수학적으로 주어질 것입니다


```python
z = β0 + β1x1    
```

여기서 계수 β0과 β1은 모형의 모수입니다.

설명 변수가 여러 개인 경우, 위의 방정식은 다음과 같이 확장될 수 있습니다


```python
z = β0 + β1x1+ β2x2+……..+ βnxn
```

여기서 계수 β0, β1, β2 및 βn은 모델의 매개변수입니다.

따라서 예측 반응 값은 위의 방정식에 의해 주어지며 z로 표시됩니다.

#### 시그모이드 함수

z로 표시된 이 예측 반응 값은 0과 1 사이에 있는 확률 값으로 변환됩니다. 우리는 예측 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용합니다. 그런 다음 이 시그모이드 함수는 실제 값을 0과 1 사이의 확률 값으로 매핑합니다.

기계 학습에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용됩니다. 시그모이드 함수는 S자형 곡선을 가지고 있습니다. 그것은 시그모이드 곡선이라고도 불립니다.

#### 의사결정경계

시그모이드 함수는 0과 1 사이의 확률 값을 반환합니다. 그런 다음 이 확률 값은 "0" 또는 "1"인 이산 클래스에 매핑됩니다. 이 확률 값을 이산 클래스에 매핑하기 위해 임계값을 선택합니다. 이 임계값을 의사결정 경계라고 합니다. 이 임계값을 초과하면 확률 값을 클래스 1에 매핑하고 클래스 0에 매핑합니다.

수학적으로 다음과 같이 표현할 수 있습니다.

p ◦ 0.5 => 클래스 = 1 \
p < 0.5 => 클래스 = 0

일반적으로 의사 결정 경계는 0.5로 설정됩니다. 따라서 확률 값이 0.8(> 0.5)이면 이 관측치를 클래스 1에 매핑합니다. 마찬가지로 확률 값이 0.2(< 0.5)이면 이 관측치를 클래스 0에 매핑합니다. 

#### 예측하기 

로지스틱 회귀 분석의 예측 함수는 관측치가 양수, 예 또는 참일 확률을 반환합니다. 이를 클래스 1이라고 하며 P(클래스 = 1)로 표시합니다. 확률이 1에 가까우면 관측치가 클래스 1에 있고 그렇지 않으면 클래스 0에 있다는 것을 모형에 대해 더 확신할 수 있습니다.

#### 로지스틱 회귀 분석의 가정

로지스틱 회귀 분석 모형에는 몇 가지 주요 가정이 필요합니다.

- 로지스틱 회귀 분석 모형에서는 종속 변수가 이항, 다항식 또는 순서형이어야 합니다.
- 관측치가 서로 독립적이어야 합니다. 따라서 관측치는 반복적인 측정에서 나와서는 안 됩니다.
- 로지스틱 회귀 분석 알고리즘에는 독립 변수 간의 다중 공선성이 거의 또는 전혀 필요하지 않습니다. 즉, 독립 변수들이 서로 너무 높은 상관 관계를 맺어서는 안 됩니다.
- 로지스틱 회귀 모형은 독립 변수와 로그 승산의 선형성을 가정합니다.
- 로지스틱 회귀 분석 모형의 성공 여부는 표본 크기에 따라 달라집니다. 일반적으로 높은 정확도를 얻으려면 큰 표본 크기가 필요합니다.

#### 로지스틱 회귀 분석의 유형

로지스틱 회귀 분석 모형은 대상 변수 범주를 기준으로 세 그룹으로 분류할 수 있습니다.

1. 이항 로지스틱 회귀 분석
이항 로지스틱 회귀 분석에서 대상 변수에는 두 가지 범주가 있습니다. 범주의 일반적인 예는 예 또는 아니오, 양호 또는 불량, 참 또는 거짓, 스팸 또는 스팸 없음, 통과 또는 실패입니다.

2. 다항 로지스틱 회귀 분석
다항 로지스틱 회귀 분석에서 대상 변수에는 특정 순서가 아닌 세 개 이상의 범주가 있습니다. 따라서 세 개 이상의 공칭 범주가 있습니다. 그 예들은 사과, 망고, 오렌지 그리고 바나나와 같은 과일의 종류를 포함합니다.

3. 순서형 로지스틱 회귀 분석
순서형 로지스틱 회귀 분석에서 대상 변수에는 세 개 이상의 순서형 범주가 있습니다. 그래서, 범주와 관련된 본질적인 순서가 있습니다. 예를 들어, 학생들의 성적은 불량, 평균, 양호, 우수로 분류될 수 있습니다.

### 라이브러리 가져오기


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```


```python
import warnings

warnings.filterwarnings('ignore')
```

### 데이터 집합 가져오기


```python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```

### 탐색적 데이터 분석

데이터를 탐색할 것입니다.


```python
# view dimensions of dataset

df.shape
```

output:

(142193, 24)

우리는 데이터 세트에 142193개의 인스턴스와 24개의 변수가 있음을 알 수 있습니다.


```python
col_names = df.columns

col_names
```

#### RISK_MM 변수 삭제
데이터 세트 설명에서 RISK_MM 기능 변수를 삭제해야 한다는 내용이 데이터 세트 설명에 나와 있습니다. 우리는 다음과 같이 그것을 떨어트려야 합니다.


```python
df.drop(['RISK_MM'], axis=1, inplace=True)
```


```python
# view summary of dataset

df.info()
```

#### 변수 유형
이 섹션에서는 데이터 세트를 범주형 변수와 숫자 변수로 분리합니다. 데이터 집합에는 범주형 변수와 숫자 변수가 혼합되어 있습니다. 범주형 변수에는 데이터 유형 개체가 있습니다. 숫자 변수의 데이터 유형은 float64입니다.

우선 범주형 변수를 찾아보겠습니다


```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

#### 범주형 변수 요약

- 날짜 변수가 있습니다. 날짜 열로 표시됩니다.
- 6개의 범주형 변수가 있습니다. 이것들은 위치, 윈드 구스트 다이어, 윈드 다이어 9am, 윈드 다이어 3pm, 비 투데이 그리고 비 투데이에 의해 주어집니다.
- 두 개의 이진 범주형 변수인 RainToday와 RainTomorrow가 있습니다.
- 내일 비가 목표 변수입니다.

#### 범주형 변수 내의 문제 탐색
먼저 범주형 변수에 대해 알아보겠습니다.


```python
# check missing values in categorical variables

df[categorical].isnull().sum()
```


```python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

데이터 세트에 결측값이 포함된 범주형 변수는 4개뿐임을 알 수 있습니다. 윈드구스트디어, 윈드디어9am, 윈드디어3pm, 레인투데이입니다.

범주형 변수의 빈도 카운트

이제 범주형 변수의 빈도 수를 확인하겠습니다.


```python
# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
```

Output:


```python
2014-04-15    49
2013-08-04    49
2014-03-18    49
2014-07-08    49
2014-02-27    49
              ..
2007-11-01     1
2007-12-30     1
2007-12-12     1
2008-01-20     1
2007-12-05     1
Name: Date, Length: 3436, dtype: int64
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Ballarat            3028
Launceston          3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cobar               2988
Cairns              2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
NorfolkIsland       2964
Penrith             2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
No     109332
Yes     31455
Name: RainToday, dtype: int64
No     110316
Yes     31877
Name: RainTomorrow, dtype: int64
```


```python
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

Output:


```python
2014-04-15    0.000345
2013-08-04    0.000345
2014-03-18    0.000345
2014-07-08    0.000345
2014-02-27    0.000345
                ...   
2007-11-01    0.000007
2007-12-30    0.000007
2007-12-12    0.000007
2008-01-20    0.000007
2007-12-05    0.000007
Name: Date, Length: 3436, dtype: float64
Canberra            0.024038
Sydney              0.023468
Perth               0.022455
Darwin              0.022448
Hobart              0.022420
Brisbane            0.022230
Adelaide            0.021731
Bendigo             0.021337
Townsville          0.021330
AliceSprings        0.021316
MountGambier        0.021309
Ballarat            0.021295
Launceston          0.021295
Albany              0.021211
Albury              0.021175
PerthAirport        0.021161
MelbourneAirport    0.021161
Mildura             0.021147
SydneyAirport       0.021133
Nuriootpa           0.021112
Sale                0.021098
Watsonia            0.021091
Tuggeranong         0.021084
Portland            0.021070
Woomera             0.021028
Cobar               0.021014
Cairns              0.021014
Wollongong          0.020979
GoldCoast           0.020957
WaggaWagga          0.020929
NorfolkIsland       0.020845
Penrith             0.020845
SalmonGums          0.020782
Newcastle           0.020782
CoffsHarbour        0.020768
Witchcliffe         0.020761
Richmond            0.020753
Dartmoor            0.020697
NorahHead           0.020599
BadgerysCreek       0.020592
MountGinini         0.020444
Moree               0.020071
Walpole             0.019825
PearceRAAF          0.019424
Williamtown         0.017954
Melbourne           0.017125
Nhil                0.011034
Katherine           0.010964
Uluru               0.010697
Name: Location, dtype: float64
W      0.068780
SE     0.065467
E      0.063794
N      0.063526
SSE    0.063245
S      0.062936
WSW    0.062598
SW     0.061867
SSW    0.060552
WNW    0.056726
NW     0.056283
ENE    0.056205
ESE    0.051374
NE     0.049651
NNW    0.046142
NNE    0.045241
Name: WindGustDir, dtype: float64
N      0.080123
SE     0.064434
E      0.063463
SSE    0.063055
NW     0.060144
S      0.059729
W      0.058090
SW     0.057928
NNE    0.055896
NNW    0.055136
ENE    0.054398
ESE    0.053153
NE     0.052935
SSW    0.052380
WNW    0.050593
WSW    0.048125
Name: WindDir9am, dtype: float64
SE     0.074990
W      0.069701
S      0.067500
WSW    0.065608
SW     0.064574
SSE    0.064293
N      0.060952
WNW    0.060875
NW     0.059553
ESE    0.058948
E      0.058667
NE     0.057415
SSW    0.056332
NNW    0.054384
ENE    0.054321
NNE    0.045319
Name: WindDir3pm, dtype: float64
No     0.768899
Yes    0.221213
Name: RainToday, dtype: float64
No     0.775819
Yes    0.224181
Name: RainTomorrow, dtype: float64
```

#### 레이블 수: cardinality

범주형 변수 내의 레이블 수를 카디널리티라고 합니다. 변수 내의 레이블 수가 많은 것을 고 카디널리티라고 합니다. 높은 카디널리티는 기계 학습 모델에서 몇 가지 심각한 문제를 일으킬 수 있습니다. 그래서 카디널리티가 높은지 확인해보겠습니다.


```python
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

Output:


```python
Date  contains  3436  labels
Location  contains  49  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  2  labels
```

우리는 전처리가 필요한 날짜 변수가 있다는 것을 알 수 있습니다. 저는 다음 섹션에서 전처리를 할 것입니다.

다른 모든 변수에는 상대적으로 적은 수의 변수가 포함되어 있습니다.

#### 날짜 변수의 피쳐 엔지니어링


```python
df['Date'].dtypes
```

날짜 변수의 데이터 유형이 개체임을 알 수 있습니다. 현재 객체로 코딩된 날짜를 datetime 형식으로 구문 분석하겠습니다.


```python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```


```python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```

Output:


```python
0    2008
1    2008
2    2008
3    2008
4    2008
Name: Year, dtype: int64
```


```python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```

Output:


```python
0    12
1    12
2    12
3    12
4    12
Name: Month, dtype: int64
```


```python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```

Output:


```python
0    1
1    2
2    3
3    4
4    5
Name: Day, dtype: int64
```


```python
# again view the summary of dataset

df.info()
```

Output:


```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 26 columns):
Date             142193 non-null datetime64[ns]
Location         142193 non-null object
MinTemp          141556 non-null float64
MaxTemp          141871 non-null float64
Rainfall         140787 non-null float64
Evaporation      81350 non-null float64
Sunshine         74377 non-null float64
WindGustDir      132863 non-null object
WindGustSpeed    132923 non-null float64
WindDir9am       132180 non-null object
WindDir3pm       138415 non-null object
WindSpeed9am     140845 non-null float64
WindSpeed3pm     139563 non-null float64
Humidity9am      140419 non-null float64
Humidity3pm      138583 non-null float64
Pressure9am      128179 non-null float64
Pressure3pm      128212 non-null float64
Cloud9am         88536 non-null float64
Cloud3pm         85099 non-null float64
Temp9am          141289 non-null float64
Temp3pm          139467 non-null float64
RainToday        140787 non-null object
RainTomorrow     142193 non-null object
Year             142193 non-null int64
Month            142193 non-null int64
Day              142193 non-null int64
dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
memory usage: 28.2+ MB
```

날짜 변수에서 추가로 세 개의 열이 생성된 것을 확인할 수 있습니다. 이제 데이터 집합에서 원래 날짜 변수를 삭제하겠습니다.


```python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```

#### 범주형 변수 탐색

이제 범주형 변수를 하나씩 살펴보도록 하겠습니다.


```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

우리는 데이터 세트에 6개의 범주형 변수가 있다는 것을 알 수 있습니다. 날짜 변수가 제거되었습니다. 먼저 범주형 변수의 결측값을 확인하겠습니다.


```python
# check for missing values in categorical variables 

df[categorical].isnull().sum()
```

WindGustDir, WindDir9am, WindDir3pm, RainToday 변수에 결측값이 포함되어 있음을 알 수 있습니다. 이 변수들을 하나씩 탐색할 것입니다.

#### 위치 변수 탐색

다음 코드를 실행하여 위치를 탐색해줍니다. 


```python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```


```python
# check labels in location variable

df.Location.unique()
```


```python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```


```python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
```

#### WindGustDir 다이어 변수 탐색

마찬가지로 다음 코드를 실행하여 위치를 탐색해줄 것입니다. 


```python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```


```python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```


```python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```


```python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

Output을 확인하여 WindGustDir 변수에는 9330개의 결측값이 있음을 알 수 있습니다.

#### WindDir9am 변수 탐색


```python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```


```python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```


```python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```


```python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

WindDir9am 변수에 결측값이 10013개 있음을 알 수 있습니다.

#### Explore WindDir3pm variable


```python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```


```python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```


```python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```


```python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

WindDir3pm 변수에 3778개의 결측값이 있습니다.

#### RainToday 변수 탐색


```python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```


```python
# check labels in WindGustDir variable

df['RainToday'].unique()
```


```python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```


```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

RainToday 변수에는 1406개의 결측값이 있습니다.

#### Numerical 변수 탐색


```python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

#### 수치 변수 요약
- 16개의 숫자 변수가 있습니다.
- 이것들은 MinTemp, MaxTemp, 강우량, 증발, 햇빛, 풍속, 풍속 9am, 풍속 3pm, 습도 9am, 습도 3pm, 압력 9am, 구름 3pm, 구름 3pm, 온도 9am 및 온도 3pm에 의해 제공됩니다.
- 모든 숫자 변수는 연속형입니다.

#### 수치 변수 내의 문제 탐색
이제 수치 변수를 살펴보겠습니다.

숫자 변수의 결측값


```python
# check missing values in numerical variables

df[numerical].isnull().sum()
```

16개의 수치 변수에 결측값이 모두 포함되어 있음을 알 수 있습니다.

#### 숫자 변수의 특이치


```python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

자세히 살펴보면 강우량, 증발량, 풍속 9am 및 풍속 3pm 열에 특이치가 포함되어 있을 수 있습니다.

상자 그림을 그려 위 변수의 특이치를 시각화합니다.


```python
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

#### 변수 분포 확인

이제 히스토그램을 그려 분포가 정규 분포인지 치우쳐 있는지 확인합니다. 변수가 정규 분포를 따르는 경우 극단값 분석을 수행하고, 그렇지 않은 경우 치우친 경우 IQR을 찾습니다.


```python
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

네 가지 변수가 모두 치우쳐 있음을 알 수 있습니다. 따라서 특이치를 찾기 위해 분위수 범위를 사용합니다.


```python
Logistic Regression Classifier Tutorial with Python
Hello friends,

In this kernel, I implement Logistic Regression with Python and Scikit-Learn. I build a Logistic Regression classifier to predict whether or not it will rain tomorrow in Australia. I train a binary classification model using Logistic Regression.

As always, I hope you find this kernel useful and your UPVOTES would be highly appreciated.


Table of Contents
Introduction to Logistic Regression
Logistic Regression intuition
Assumptions of Logistic Regression
Types of Logistic Regression
Import libraries
Import dataset
Exploratory data analysis
Declare feature vector and target variable
Split data into separate training and test set
Feature engineering
Feature scaling
Model training
Predict results
Check accuracy score
Confusion matrix
Classification metrices
Adjusting the threshold level
ROC - AUC
k-Fold Cross Validation
Hyperparameter optimization using GridSearch CV
Results and conclusion
References
1. Introduction to Logistic Regression 
Table of Contents

When data scientists may come across a new classification problem, the first algorithm that may come across their mind is Logistic Regression. It is a supervised learning classification algorithm which is used to predict observations to a discrete set of classes. Practically, it is used to classify observations into different categories. Hence, its output is discrete in nature. Logistic Regression is also called Logit Regression. It is one of the most simple, straightforward and versatile classification algorithms which is used to solve classification problems.

2. Logistic Regression intuition 
Table of Contents

In statistics, the Logistic Regression model is a widely used statistical model which is primarily used for classification purposes. It means that given a set of observations, Logistic Regression algorithm helps us to classify these observations into two or more discrete classes. So, the target variable is discrete in nature.

The Logistic Regression algorithm works as follows -

Implement linear equation
Logistic Regression algorithm works by implementing a linear equation with independent or explanatory variables to predict a response value. For example, we consider the example of number of hours studied and probability of passing the exam. Here, number of hours studied is the explanatory variable and it is denoted by x1. Probability of passing the exam is the response or target variable and it is denoted by z.

If we have one explanatory variable (x1) and one response variable (z), then the linear equation would be given mathematically with the following equation-

z = β0 + β1x1    

Here, the coefficients β0 and β1 are the parameters of the model.

If there are multiple explanatory variables, then the above equation can be extended to

z = β0 + β1x1+ β2x2+……..+ βnxn

Here, the coefficients β0, β1, β2 and βn are the parameters of the model.

So, the predicted response value is given by the above equations and is denoted by z.

Sigmoid Function
This predicted response value, denoted by z is then converted into a probability value that lie between 0 and 1. We use the sigmoid function in order to map predicted values to probability values. This sigmoid function then maps any real value into a probability value between 0 and 1.

In machine learning, sigmoid function is used to map predictions to probabilities. The sigmoid function has an S shaped curve. It is also called sigmoid curve.

A Sigmoid function is a special case of the Logistic function. It is given by the following mathematical formula.

Graphically, we can represent sigmoid function with the following graph.

Sigmoid Function
Sigmoid Function

Decision boundary
The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called Decision boundary. Above this threshold value, we will map the probability values into class 1 and below which we will map values into class 0.

Mathematically, it can be expressed as follows:-

p ≥ 0.5 => class = 1

p < 0.5 => class = 0

Generally, the decision boundary is set to 0.5. So, if the probability value is 0.8 (> 0.5), we will map this observation to class 1. Similarly, if the probability value is 0.2 (< 0.5), we will map this observation to class 0. This is represented in the graph below-

Decision boundary in sigmoid function

Making predictions
Now, we know about sigmoid function and decision boundary in logistic regression. We can use our knowledge of sigmoid function and decision boundary to write a prediction function. A prediction function in logistic regression returns the probability of the observation being positive, Yes or True. We call this as class 1 and it is denoted by P(class = 1). If the probability inches closer to one, then we will be more confident about our model that the observation is in class 1, otherwise it is in class 0.

3. Assumptions of Logistic Regression 
Table of Contents

The Logistic Regression model requires several key assumptions. These are as follows:-

Logistic Regression model requires the dependent variable to be binary, multinomial or ordinal in nature.

It requires the observations to be independent of each other. So, the observations should not come from repeated measurements.

Logistic Regression algorithm requires little or no multicollinearity among the independent variables. It means that the independent variables should not be too highly correlated with each other.

Logistic Regression model assumes linearity of independent variables and log odds.

The success of Logistic Regression model depends on the sample sizes. Typically, it requires a large sample size to achieve the high accuracy.

4. Types of Logistic Regression 
Table of Contents

Logistic Regression model can be classified into three groups based on the target variable categories. These three groups are described below:-

1. Binary Logistic Regression
In Binary Logistic Regression, the target variable has two possible categories. The common examples of categories are yes or no, good or bad, true or false, spam or no spam and pass or fail.

2. Multinomial Logistic Regression
In Multinomial Logistic Regression, the target variable has three or more categories which are not in any particular order. So, there are three or more nominal categories. The examples include the type of categories of fruits - apple, mango, orange and banana.

3. Ordinal Logistic Regression
In Ordinal Logistic Regression, the target variable has three or more ordinal categories. So, there is intrinsic order involved with the categories. For example, the student performance can be categorized as poor, average, good and excellent.

5. Import libraries 
Table of Contents

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv
import warnings

warnings.filterwarnings('ignore')
6. Import dataset 
Table of Contents

data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
7. Exploratory data analysis 
Table of Contents

Now, I will explore the data to gain insights about the data.

# view dimensions of dataset

df.shape
(142193, 24)
We can see that there are 142193 instances and 24 variables in the data set.

# preview the dataset

df.head()
Date	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RISK_MM	RainTomorrow
0	2008-12-01	Albury	13.4	22.9	0.6	NaN	NaN	W	44.0	W	...	22.0	1007.7	1007.1	8.0	NaN	16.9	21.8	No	0.0	No
1	2008-12-02	Albury	7.4	25.1	0.0	NaN	NaN	WNW	44.0	NNW	...	25.0	1010.6	1007.8	NaN	NaN	17.2	24.3	No	0.0	No
2	2008-12-03	Albury	12.9	25.7	0.0	NaN	NaN	WSW	46.0	W	...	30.0	1007.6	1008.7	NaN	2.0	21.0	23.2	No	0.0	No
3	2008-12-04	Albury	9.2	28.0	0.0	NaN	NaN	NE	24.0	SE	...	16.0	1017.6	1012.8	NaN	NaN	18.1	26.5	No	1.0	No
4	2008-12-05	Albury	17.5	32.3	1.0	NaN	NaN	W	41.0	ENE	...	33.0	1010.8	1006.0	7.0	8.0	17.8	29.7	No	0.2	No
5 rows × 24 columns

col_names = df.columns

col_names
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'],
      dtype='object')
Drop RISK_MM variable
It is given in the dataset description, that we should drop the RISK_MM feature variable from the dataset description. So, we should drop it as follows-

df.drop(['RISK_MM'], axis=1, inplace=True)
# view summary of dataset

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 23 columns):
Date             142193 non-null object
Location         142193 non-null object
MinTemp          141556 non-null float64
MaxTemp          141871 non-null float64
Rainfall         140787 non-null float64
Evaporation      81350 non-null float64
Sunshine         74377 non-null float64
WindGustDir      132863 non-null object
WindGustSpeed    132923 non-null float64
WindDir9am       132180 non-null object
WindDir3pm       138415 non-null object
WindSpeed9am     140845 non-null float64
WindSpeed3pm     139563 non-null float64
Humidity9am      140419 non-null float64
Humidity3pm      138583 non-null float64
Pressure9am      128179 non-null float64
Pressure3pm      128212 non-null float64
Cloud9am         88536 non-null float64
Cloud3pm         85099 non-null float64
Temp9am          141289 non-null float64
Temp3pm          139467 non-null float64
RainToday        140787 non-null object
RainTomorrow     142193 non-null object
dtypes: float64(16), object(7)
memory usage: 25.0+ MB
Types of variables
In this section, I segregate the dataset into categorical and numerical variables. There are a mixture of categorical and numerical variables in the dataset. Categorical variables have data type object. Numerical variables have data type float64.

First of all, I will find categorical variables.

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
There are 7 categorical variables

The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
# view the categorical variables

df[categorical].head()
Date	Location	WindGustDir	WindDir9am	WindDir3pm	RainToday	RainTomorrow
0	2008-12-01	Albury	W	W	WNW	No	No
1	2008-12-02	Albury	WNW	NNW	WSW	No	No
2	2008-12-03	Albury	WSW	W	WSW	No	No
3	2008-12-04	Albury	NE	SE	E	No	No
4	2008-12-05	Albury	W	ENE	NW	No	No
Summary of categorical variables
There is a date variable. It is denoted by Date column.
There are 6 categorical variables. These are given by Location, WindGustDir, WindDir9am, WindDir3pm, RainToday and RainTomorrow.
There are two binary categorical variables - RainToday and RainTomorrow.
RainTomorrow is the target variable.
Explore problems within categorical variables
First, I will explore the categorical variables.

Missing values in categorical variables
# check missing values in categorical variables

df[categorical].isnull().sum()
Date                0
Location            0
WindGustDir      9330
WindDir9am      10013
WindDir3pm       3778
RainToday        1406
RainTomorrow        0
dtype: int64
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
WindGustDir     9330
WindDir9am     10013
WindDir3pm      3778
RainToday       1406
dtype: int64
We can see that there are only 4 categorical variables in the dataset which contains missing values. These are WindGustDir, WindDir9am, WindDir3pm and RainToday.

Frequency counts of categorical variables
Now, I will check the frequency counts of categorical variables.

# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
2014-04-15    49
2013-08-04    49
2014-03-18    49
2014-07-08    49
2014-02-27    49
              ..
2007-11-01     1
2007-12-30     1
2007-12-12     1
2008-01-20     1
2007-12-05     1
Name: Date, Length: 3436, dtype: int64
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Ballarat            3028
Launceston          3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cobar               2988
Cairns              2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
NorfolkIsland       2964
Penrith             2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
No     109332
Yes     31455
Name: RainToday, dtype: int64
No     110316
Yes     31877
Name: RainTomorrow, dtype: int64
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
2014-04-15    0.000345
2013-08-04    0.000345
2014-03-18    0.000345
2014-07-08    0.000345
2014-02-27    0.000345
                ...   
2007-11-01    0.000007
2007-12-30    0.000007
2007-12-12    0.000007
2008-01-20    0.000007
2007-12-05    0.000007
Name: Date, Length: 3436, dtype: float64
Canberra            0.024038
Sydney              0.023468
Perth               0.022455
Darwin              0.022448
Hobart              0.022420
Brisbane            0.022230
Adelaide            0.021731
Bendigo             0.021337
Townsville          0.021330
AliceSprings        0.021316
MountGambier        0.021309
Ballarat            0.021295
Launceston          0.021295
Albany              0.021211
Albury              0.021175
PerthAirport        0.021161
MelbourneAirport    0.021161
Mildura             0.021147
SydneyAirport       0.021133
Nuriootpa           0.021112
Sale                0.021098
Watsonia            0.021091
Tuggeranong         0.021084
Portland            0.021070
Woomera             0.021028
Cobar               0.021014
Cairns              0.021014
Wollongong          0.020979
GoldCoast           0.020957
WaggaWagga          0.020929
NorfolkIsland       0.020845
Penrith             0.020845
SalmonGums          0.020782
Newcastle           0.020782
CoffsHarbour        0.020768
Witchcliffe         0.020761
Richmond            0.020753
Dartmoor            0.020697
NorahHead           0.020599
BadgerysCreek       0.020592
MountGinini         0.020444
Moree               0.020071
Walpole             0.019825
PearceRAAF          0.019424
Williamtown         0.017954
Melbourne           0.017125
Nhil                0.011034
Katherine           0.010964
Uluru               0.010697
Name: Location, dtype: float64
W      0.068780
SE     0.065467
E      0.063794
N      0.063526
SSE    0.063245
S      0.062936
WSW    0.062598
SW     0.061867
SSW    0.060552
WNW    0.056726
NW     0.056283
ENE    0.056205
ESE    0.051374
NE     0.049651
NNW    0.046142
NNE    0.045241
Name: WindGustDir, dtype: float64
N      0.080123
SE     0.064434
E      0.063463
SSE    0.063055
NW     0.060144
S      0.059729
W      0.058090
SW     0.057928
NNE    0.055896
NNW    0.055136
ENE    0.054398
ESE    0.053153
NE     0.052935
SSW    0.052380
WNW    0.050593
WSW    0.048125
Name: WindDir9am, dtype: float64
SE     0.074990
W      0.069701
S      0.067500
WSW    0.065608
SW     0.064574
SSE    0.064293
N      0.060952
WNW    0.060875
NW     0.059553
ESE    0.058948
E      0.058667
NE     0.057415
SSW    0.056332
NNW    0.054384
ENE    0.054321
NNE    0.045319
Name: WindDir3pm, dtype: float64
No     0.768899
Yes    0.221213
Name: RainToday, dtype: float64
No     0.775819
Yes    0.224181
Name: RainTomorrow, dtype: float64
Number of labels: cardinality
The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is known as high cardinality. High cardinality may pose some serious problems in the machine learning model. So, I will check for high cardinality.

# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
Date  contains  3436  labels
Location  contains  49  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  2  labels
We can see that there is a Date variable which needs to be preprocessed. I will do preprocessing in the following section.

All the other variables contain relatively smaller number of variables.

Feature Engineering of Date Variable
df['Date'].dtypes
dtype('O')
We can see that the data type of Date variable is object. I will parse the date currently coded as object into datetime format.

# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
0    2008
1    2008
2    2008
3    2008
4    2008
Name: Year, dtype: int64
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
0    12
1    12
2    12
3    12
4    12
Name: Month, dtype: int64
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
0    1
1    2
2    3
3    4
4    5
Name: Day, dtype: int64
# again view the summary of dataset

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 26 columns):
Date             142193 non-null datetime64[ns]
Location         142193 non-null object
MinTemp          141556 non-null float64
MaxTemp          141871 non-null float64
Rainfall         140787 non-null float64
Evaporation      81350 non-null float64
Sunshine         74377 non-null float64
WindGustDir      132863 non-null object
WindGustSpeed    132923 non-null float64
WindDir9am       132180 non-null object
WindDir3pm       138415 non-null object
WindSpeed9am     140845 non-null float64
WindSpeed3pm     139563 non-null float64
Humidity9am      140419 non-null float64
Humidity3pm      138583 non-null float64
Pressure9am      128179 non-null float64
Pressure3pm      128212 non-null float64
Cloud9am         88536 non-null float64
Cloud3pm         85099 non-null float64
Temp9am          141289 non-null float64
Temp3pm          139467 non-null float64
RainToday        140787 non-null object
RainTomorrow     142193 non-null object
Year             142193 non-null int64
Month            142193 non-null int64
Day              142193 non-null int64
dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
memory usage: 28.2+ MB
We can see that there are three additional columns created from Date variable. Now, I will drop the original Date variable from the dataset.

# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
# preview the dataset again

df.head()
Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	WindDir3pm	...	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RainTomorrow	Year	Month	Day
0	Albury	13.4	22.9	0.6	NaN	NaN	W	44.0	W	WNW	...	1007.1	8.0	NaN	16.9	21.8	No	No	2008	12	1
1	Albury	7.4	25.1	0.0	NaN	NaN	WNW	44.0	NNW	WSW	...	1007.8	NaN	NaN	17.2	24.3	No	No	2008	12	2
2	Albury	12.9	25.7	0.0	NaN	NaN	WSW	46.0	W	WSW	...	1008.7	NaN	2.0	21.0	23.2	No	No	2008	12	3
3	Albury	9.2	28.0	0.0	NaN	NaN	NE	24.0	SE	E	...	1012.8	NaN	NaN	18.1	26.5	No	No	2008	12	4
4	Albury	17.5	32.3	1.0	NaN	NaN	W	41.0	ENE	NW	...	1006.0	7.0	8.0	17.8	29.7	No	No	2008	12	5
5 rows × 25 columns

Now, we can see that the Date variable has been removed from the dataset.

Explore Categorical Variables
Now, I will explore the categorical variables one by one.

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
There are 6 categorical variables

The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
We can see that there are 6 categorical variables in the dataset. The Date variable has been removed. First, I will check missing values in categorical variables.

# check for missing values in categorical variables 

df[categorical].isnull().sum()
Location            0
WindGustDir      9330
WindDir9am      10013
WindDir3pm       3778
RainToday        1406
RainTomorrow        0
dtype: int64
We can see that WindGustDir, WindDir9am, WindDir3pm, RainToday variables contain missing values. I will explore these variables one by one.

Explore Location variable
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
Location contains 49 labels
# check labels in location variable

df.Location.unique()
array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)
# check frequency distribution of values in Location variable

df.Location.value_counts()
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Ballarat            3028
Launceston          3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cobar               2988
Cairns              2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
NorfolkIsland       2964
Penrith             2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
Albany	Albury	AliceSprings	BadgerysCreek	Ballarat	Bendigo	Brisbane	Cairns	Canberra	Cobar	...	Townsville	Tuggeranong	Uluru	WaggaWagga	Walpole	Watsonia	Williamtown	Witchcliffe	Wollongong	Woomera
0	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 48 columns

Explore WindGustDir variable
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
WindGustDir contains 17 labels
# check labels in WindGustDir variable

df['WindGustDir'].unique()
array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
       'S', 'NW', 'SE', 'ESE', nan, 'E', 'SSW'], dtype=object)
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
ENE	ESE	N	NE	NNE	NNW	NW	S	SE	SSE	SSW	SW	W	WNW	WSW	NaN
0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
1	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0
3	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
ENE    7992
ESE    7305
N      9033
NE     7060
NNE    6433
NNW    6561
NW     8003
S      8949
SE     9309
SSE    8993
SSW    8610
SW     8797
W      9780
WNW    8066
WSW    8901
NaN    9330
dtype: int64
We can see that there are 9330 missing values in WindGustDir variable.

Explore WindDir9am variable
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
WindDir9am contains 17 labels
# check labels in WindDir9am variable

df['WindDir9am'].unique()
array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
       'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
ENE	ESE	N	NE	NNE	NNW	NW	S	SE	SSE	SSW	SW	W	WNW	WSW	NaN
0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
1	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0
2	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
3	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0
4	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
ENE     7735
ESE     7558
N      11393
NE      7527
NNE     7948
NNW     7840
NW      8552
S       8493
SE      9162
SSE     8966
SSW     7448
SW      8237
W       8260
WNW     7194
WSW     6843
NaN    10013
dtype: int64
We can see that there are 10013 missing values in the WindDir9am variable.

Explore WindDir3pm variable
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
WindDir3pm contains 17 labels
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
       'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
ENE	ESE	N	NE	NNE	NNW	NW	S	SE	SSE	SSW	SW	W	WNW	WSW	NaN
0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0
2	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0
3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
ENE     7724
ESE     8382
N       8667
NE      8164
NNE     6444
NNW     7733
NW      8468
S       9598
SE     10663
SSE     9142
SSW     8010
SW      9182
W       9911
WNW     8656
WSW     9329
NaN     3778
dtype: int64
There are 3778 missing values in the WindDir3pm variable.

Explore RainToday variable
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
RainToday contains 3 labels
# check labels in WindGustDir variable

df['RainToday'].unique()
array(['No', 'Yes', nan], dtype=object)
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
No     109332
Yes     31455
Name: RainToday, dtype: int64
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
Yes	NaN
0	0	0
1	0	0
2	0	0
3	0	0
4	0	0
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
Yes    31455
NaN     1406
dtype: int64
There are 1406 missing values in the RainToday variable.

Explore Numerical Variables
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
There are 19 numerical variables

The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
# view the numerical variables

df[numerical].head()
MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustSpeed	WindSpeed9am	WindSpeed3pm	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	Year	Month	Day
0	13.4	22.9	0.6	NaN	NaN	44.0	20.0	24.0	71.0	22.0	1007.7	1007.1	8.0	NaN	16.9	21.8	2008	12	1
1	7.4	25.1	0.0	NaN	NaN	44.0	4.0	22.0	44.0	25.0	1010.6	1007.8	NaN	NaN	17.2	24.3	2008	12	2
2	12.9	25.7	0.0	NaN	NaN	46.0	19.0	26.0	38.0	30.0	1007.6	1008.7	NaN	2.0	21.0	23.2	2008	12	3
3	9.2	28.0	0.0	NaN	NaN	24.0	11.0	9.0	45.0	16.0	1017.6	1012.8	NaN	NaN	18.1	26.5	2008	12	4
4	17.5	32.3	1.0	NaN	NaN	41.0	7.0	20.0	82.0	33.0	1010.8	1006.0	7.0	8.0	17.8	29.7	2008	12	5
Summary of numerical variables
There are 16 numerical variables.
These are given by MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am and Temp3pm.
All of the numerical variables are of continuous type.
Explore problems within numerical variables
Now, I will explore the numerical variables.

Missing values in numerical variables
# check missing values in numerical variables

df[numerical].isnull().sum()
MinTemp            637
MaxTemp            322
Rainfall          1406
Evaporation      60843
Sunshine         67816
WindGustSpeed     9270
WindSpeed9am      1348
WindSpeed3pm      2630
Humidity9am       1774
Humidity3pm       3610
Pressure9am      14014
Pressure3pm      13981
Cloud9am         53657
Cloud3pm         57094
Temp9am            904
Temp3pm           2726
Year                 0
Month                0
Day                  0
dtype: int64
We can see that all the 16 numerical variables contain missing values.

Outliers in numerical variables
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
        MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
count  141556.0  141871.0  140787.0      81350.0   74377.0       132923.0   
mean       12.0      23.0       2.0          5.0       8.0           40.0   
std         6.0       7.0       8.0          4.0       4.0           14.0   
min        -8.0      -5.0       0.0          0.0       0.0            6.0   
25%         8.0      18.0       0.0          3.0       5.0           31.0   
50%        12.0      23.0       0.0          5.0       8.0           39.0   
75%        17.0      28.0       1.0          7.0      11.0           48.0   
max        34.0      48.0     371.0        145.0      14.0          135.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
count      140845.0      139563.0     140419.0     138583.0     128179.0   
mean           14.0          19.0         69.0         51.0       1018.0   
std             9.0           9.0         19.0         21.0          7.0   
min             0.0           0.0          0.0          0.0        980.0   
25%             7.0          13.0         57.0         37.0       1013.0   
50%            13.0          19.0         70.0         52.0       1018.0   
75%            19.0          24.0         83.0         66.0       1022.0   
max           130.0          87.0        100.0        100.0       1041.0   

       Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
count     128212.0   88536.0   85099.0  141289.0  139467.0  142193.0   
mean        1015.0       4.0       5.0      17.0      22.0    2013.0   
std            7.0       3.0       3.0       6.0       7.0       3.0   
min          977.0       0.0       0.0      -7.0      -5.0    2007.0   
25%         1010.0       1.0       2.0      12.0      17.0    2011.0   
50%         1015.0       5.0       5.0      17.0      21.0    2013.0   
75%         1020.0       7.0       7.0      22.0      26.0    2015.0   
max         1040.0       9.0       9.0      40.0      47.0    2017.0   

          Month       Day  
count  142193.0  142193.0  
mean        6.0      16.0  
std         3.0       9.0  
min         1.0       1.0  
25%         3.0       8.0  
50%         6.0      16.0  
75%         9.0      23.0  
max        12.0      31.0   2
On closer inspection, we can see that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns may contain outliers.

I will draw boxplots to visualise outliers in the above variables.

# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
Text(0, 0.5, 'WindSpeed3pm')

The above boxplots confirm that there are lot of outliers in these variables.

Check the distribution of variables
Now, I will plot the histograms to check distributions to find out if they are normal or skewed. If the variable follows normal distribution, then I will do Extreme Value Analysis otherwise if they are skewed, I will find IQR (Interquantile range).

# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
Text(0, 0.5, 'RainTomorrow')

We can see that all the four variables are skewed. So, I will use interquantile range to find outliers.


```

강우량의 경우 최소값과 최대값은 0.0과 371.0입니다. 따라서 특이치는 3.2보다 큰 값입니다.


```python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

증발의 경우 최소값과 최대값은 0.0과 145.0입니다. 따라서 특이치는 21.8보다 큰 값입니다.


```python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

풍속 9am의 경우 최소값과 최대값은 0.0과 130.0입니다. 따라서 특이치는 55.0보다 큰 값입니다.


```python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

풍속 3pm의 경우 최소값과 최대값은 0.0과 87.0입니다. 따라서 특이치는 57.0보다 큰 값입니다.

### 피쳐 벡터 및 대상 변수 선언


```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

### 데이터를 별도의 교육 및 테스트 세트로 분할


```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```

### 피쳐 엔지니어링

기능 엔지니어링은 원시 데이터를 유용한 기능으로 변환하여 모델을 더 잘 이해하고 예측력을 높이는 데 도움이 됩니다. 저는 다양한 유형의 변수에 대해 피쳐 엔지니어링을 수행할 것입니다.

먼저 범주형 변수와 숫자형 변수를 다시 별도로 표시하겠습니다.


```python
# check data types in X_train

X_train.dtypes
```


```python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```


```python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

#### 숫자 변수의 결측값 엔지니어링


```python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```


```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```


```python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

추정
데이터가 랜덤으로 완전히 누락되었다고 가정합니다(MCAR). 결측값을 귀속시키는 데 사용할 수 있는 두 가지 방법이 있습니다. 하나는 평균 또는 중위수 귀책이고 다른 하나는 랜덤 표본 귀책입니다. 데이터 집합에 특이치가 있을 경우 중위수 귀책을 사용해야 합니다. 중위수 귀인은 특이치에 강하므로 중위수 귀인을 사용합니다.

결측값을 데이터의 적절한 통계적 측도(이 경우 중위수)로 귀속시킵니다. 귀속은 교육 세트에 대해 수행된 다음 테스트 세트에 전파되어야 합니다. 즉, 트레인과 테스트 세트 모두에서 결측값을 채우기 위해 사용되는 통계적 측정값은 트레인 세트에서만 추출되어야 합니다. 이는 과적합을 방지하기 위한 것입니다.


```python
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True) 
```


```python
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```


```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

이제 훈련 및 테스트 세트의 숫자 열에 결측값이 없음을 알 수 있습니다.

#### 범주형 변수의 결측값 엔지니어링


```python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```


```python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```


```python
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```


```python
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()
```


```python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```

마지막으로 X_train과 X_test에서 결측값을 확인하겠습니다.


```python
# check missing values in X_train

X_train.isnull().sum()
```


```python
# check missing values in X_test

X_test.isnull().sum()
```

코드의 실행을 통해 X_train 및 X_test에서 결측값이 없음을 알 수 있습니다.

#### 숫자 변수의 공학적 특이치

강우량, 증발량, 풍속 9am 및 풍속 3pm 열에 특이치가 포함되어 있는 것을 확인했습니다. 최상위 코드화 방법을 사용하여 최대값을 상한으로 설정하고 위 변수에서 특이치를 제거합니다.


```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```


```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```


```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```


```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```


```python
X_train[numerical].describe()
```

이제 강우량, 증발량, 풍속 9am 및 풍속 3pm 열의 특이치가 상한선임을 알 수 있습니다.

#### 범주형 변수 인코딩


```python
categorical
```


```python
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```

코드의 실행을 통해 RainToday_0 및 RainToday_1 변수 두 개가 RainToday 변수에서 생성되었음을 알 수 있습니다.

이제 X_train 교육 세트를 생성하겠습니다.


```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```

마찬가지로 X_test testing set도 만들겠습니다.


```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```

이제 모델 구축을 위한 교육 및 테스트가 준비되었습니다. 그 전에 모든 형상 변수를 동일한 척도에 매핑해야 합니다. 이를 형상 스케일링이라고 합니다. 다음과 같이 하겠습니다.

### Feature Scaling


```python
X_train.describe()
```


```python
cols = X_train.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```


```python
X_train = pd.DataFrame(X_train, columns=[cols])
```


```python
X_test = pd.DataFrame(X_test, columns=[cols])
```


```python
X_train.describe()
```

이제 X_train 데이터 세트를 로지스틱 회귀 분류기에 입력할 준비가 되었습니다. 다음과 같이 하겠습니다.

### Model training


```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

### Predict results


```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

#### predict_proba 방법
predict_proba 메서드는 이 경우 대상 변수(0 및 1)에 대한 확률을 배열 형식으로 제공합니다.

0은 비가 오지 않을 확률이고 1은 비가 올 확률입니다.


```python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```


```python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```

### Check accuracy score 


```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

여기서 y_test는 참 클래스 레이블이고 y_pred_test는 테스트 세트의 예측 클래스 레이블입니다.

열차 세트와 테스트 세트 정확도 비교
이제 트레인 세트와 테스트 세트 정확도를 비교하여 과적합 여부를 확인하겠습니다.


```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```


```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

#### 과적합 및 과소적합 여부 점검


```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

교육 세트 정확도 점수는 0.8476인 반면 테스트 세트 정확도는 0.8501입니다. 이 두 값은 상당히 비슷합니다. 따라서 과적합의 문제는 없습니다.

로지스틱 회귀 분석에서는 C = 1의 기본값을 사용합니다. 교육 및 테스트 세트 모두에서 약 85%의 정확도로 우수한 성능을 제공합니다. 그러나 교육 및 테스트 세트의 모델 성능은 매우 유사합니다. 그것은 아마도 부족한 경우일 것입니다.

C를 늘리고 좀 더 유연한 모델을 맞출 것입니다.


```python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```


```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

Output:

Training set score: 0.8478
Test set score: 0.8505

C=100이 테스트 세트 정확도를 높이고 교육 세트 정확도를 약간 높인다는 것을 알 수 있습니다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘해야 한다는 결론을 내릴 수 있습니다.

이제 C=0.01을 설정하여 기본값인 C=1보다 정규화된 모델을 사용하면 어떻게 되는지 알아보겠습니다.


```python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```


```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

Output:

Training set score: 0.8409
Test set score: 0.8448

 C=0.01을 설정하여 보다 정규화된 모델을 사용하면 교육 및 테스트 세트 정확도가 기본 매개 변수에 비해 모두 감소합니다.

#### 모델 정확도와 null 정확도 비교
모형 정확도는 0.8501입니다. 그러나 위의 정확도에 근거하여 모델이 매우 좋다고 말할 수는 없습니다. 우리는 그것을 null 정확도와 비교해야 합니다. Null 정확도는 항상 가장 빈도가 높은 클래스를 예측하여 얻을 수 있는 정확도입니다.

우리는 먼저 테스트 세트의 클래스 분포를 확인해야 합니다.


```python
# check class distribution in test set

y_test.value_counts()
```

Output:

No     22067
Yes     6372
Name: RainTomorrow, dtype: int64

우리는 가장 빈번한 수업의 발생 횟수가 22067회임을 알 수 있습니다. 따라서 22067을 총 발생 횟수로 나누어 null 정확도를 계산할 수 있습니다.


```python
# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

Output:

Null accuracy score: 0.7759

모델 정확도 점수는 0.8501이지만 null 정확도 점수는 0.7759임을 알 수 있습니다. 따라서 로지스틱 회귀 분석 모형이 클래스 레이블을 예측하는 데 매우 효과적이라는 결론을 내릴 수 있습니다.

이제 위의 분석을 바탕으로 분류 모델 정확도가 매우 우수하다는 결론을 내릴 수 있습니다. 우리 모델은 클래스 레이블을 예측하는 측면에서 매우 잘 수행하고 있습니다.

그러나 기본적인 값 분포는 제공하지 않습니다. 또한, 그것은 우리 반 학생들이 저지르는 오류의 유형에 대해서는 아무 것도 말해주지 않습니다.

Confusion matrix 라고 불리는 또 다른 도구를 가지고 있습니다.

### Confusion matrix

혼동 행렬은 분류 알고리즘의 성능을 요약하는 도구입니다. 혼동 행렬은 분류 모델 성능과 모델에 의해 생성되는 오류 유형에 대한 명확한 그림을 제공합니다. 각 범주별로 분류된 정확한 예측과 잘못된 예측의 요약을 제공합니다. 요약은 표 형식으로 표시됩니다.

분류 모델 성능을 평가하는 동안 네 가지 유형의 결과가 가능합니다.

- 참 양성(TP) – 참 양성은 관측치가 특정 클래스에 속하고 관측치가 실제로 해당 클래스에 속한다고 예측할 때 발생합니다.

- True Negatives(TN) – True Negatives는 관측치가 특정 클래스에 속하지 않고 실제로 관측치가 해당 클래스에 속하지 않을 때 발생합니다.

- False Positives(FP) – False Positives는 관측치가 특정 클래스에 속하지만 실제로는 해당 클래스에 속하지 않는다고 예측할 때 발생합니다. 이러한 유형의 오류를 유형 I 오류라고 합니다.

- FN(False Negatives) – 관측치가 특정 클래스에 속하지 않지만 실제로는 해당 클래스에 속한다고 예측할 때 False Negatives가 발생합니다. 이것은 매우 심각한 오류이며 Type II 오류라고 합니다.

이 네 가지 결과는 아래에 제시된 혼동 매트릭스로 요약됩니다.


```python
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

혼동 행렬은 20892 + 3285 = 24177 정확한 예측과 3087 + 1175 = 4262 부정확한 예측을 나타냅니다.

이 경우,

- 참 양성(실제 양성:1 및 예측 양성:1) - 20892
- 참 음수(실제 음수:0 및 예측 음수:0) - 3285
- 거짓 양성(실제 음성: 0이지만 예측 양성: 1) - 1175(유형 I 오류)
- 거짓 음성(실제 양의 1이지만 예측 음의 0) - 3087(타입 II 오류)


```python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

### Classification metrices

#### 분류 보고서
분류 보고서는 분류 모델 성능을 평가하는 또 다른 방법입니다. 모형의 정밀도, 호출, f1 및 지원 점수가 표시됩니다. 저는 이 용어들을 나중에 설명했습니다.

다음과 같이 분류 보고서를 인쇄할 수 있습니다


```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

Output:

              precision    recall  f1-score   support

          No       0.87      0.95      0.91     22067
         Yes       0.74      0.52      0.61      6372

    accuracy                           0.85     28439
   macro avg       0.80      0.73      0.76     28439
weighted avg       0.84      0.85      0.84     28439

#### Classification accuracy


```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```


```python
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

Output:

Classification accuracy : 0.8502

#### Classification error


```python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

Output:

Classification error : 0.1498

#### 정확
정밀도는 모든 예측 긍정 결과 중 정확하게 예측된 긍정 결과의 백분율로 정의할 수 있습니다. 참 및 거짓 양성의 합계에 대한 참 양성(TP + FP)의 비율로 지정할 수 있습니다.

따라서 정밀도는 정확하게 예측된 양성 결과의 비율을 식별합니다. 그것은 부정적인 계층보다 긍정적인 계층에 더 관심이 있습니다.

수학적으로 정밀도는 TP 대 (TP + FP)의 비율로 정의할 수 있습니다.


```python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```

Output:

Precision : 0.9468    

#### 리콜
리콜은 모든 실제 긍정적 결과 중 정확하게 예측된 긍정적 결과의 비율로 정의할 수 있습니다. 참 양성과 거짓 음성의 합(TP + FN)에 대한 참 양성(TP)의 비율로 지정할 수 있습니다. 리콜은 민감도라고도 합니다.

호출은 정확하게 예측된 실제 긍정의 비율을 식별합니다.

수학적으로 호출은 TP 대 (TP + FN)의 비율로 지정할 수 있습니다.

Output:

Recall or Sensitivity : 0.8713   

#### True Positive Rate
True Positive Rate는 Recall과 동의어입니다.


```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

Output:

True Positive Rate : 0.8713

#### False Positive Rate


```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```


```python
Output:

False Positive Rate : 0.2634
```

#### Specificity


```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

Output:

Specificity : 0.7366

#### f1-점수 
f1-점수는 정밀도와 호출의 가중 조화 평균입니다. 가능한 최상의 f1-점수는 1.0이고 최악의 f1-점수는 0.0입니다. f1-점수는 정밀도와 호출의 조화 평균입니다. 따라서 f1-점수는 정확도와 리콜을 계산에 포함시키기 때문에 정확도 측도보다 항상 낮습니다. f1-점수의 가중 평균은 전역 정확도가 아닌 분류기 모델을 비교하는 데 사용되어야 합니다.

### Adjusting the threshold level 


```python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

각 행에서 숫자는 1이 됩니다.

2개의 클래스(0 및 1)에 해당하는 2개의 열이 있습니다.
- 클래스 0 - 내일 비가 오지 않을 확률을 예측합니다.
- 클래스 1 - 내일 비가 올 확률을 예측합니다.

예측 확률의 중요성
- 비가 오거나 오지 않을 확률로 관측치의 순위를 매길 수 있습니다.

predict_proba 공정
- 확률을 예측합니다
- 확률이 가장 높은 클래스 선택

분류 임계값 레벨
- 분류 임계값 레벨은 0.5입니다.
- 클래스 1 - 확률이 0.5 이상일 경우 비가 올 확률이 예측됩니다.
- 클래스 0 - 확률이 0.5 미만일 경우 비가 오지 않을 확률이 예측됩니다.


```python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```


```python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```


```python
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```


```python
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

### 관찰
위의 히스토그램이 매우 양으로 치우쳐 있음을 알 수 있습니다.
첫 번째 열은 확률이 0.0과 0.1 사이인 관측치가 약 15,000개임을 나타냅니다.
확률이 0.5보다 작은 관측치가 있습니다.
그래서 이 소수의 관측치들은 내일 비가 올 것이라고 예측하고 있습니다.
내일은 비가 오지 않을 것이라는 관측이 대다수입니다.

#### 임계값을 낮춥니다


```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

Output:


```python
With 0.1 threshold the Confusion Matrix is  

 [[12726  9341]
 [  547  5825]] 

 with 18551 correct predictions,  

 9341 Type I errors( False Positives),  

 547 Type II errors( False Negatives),  

 Accuracy score:  0.6523084496641935 

 Sensitivity:  0.9141556811048337 

 Specificity:  0.5766982371867494 

 ==================================================== 


With 0.2 threshold the Confusion Matrix is  

 [[17066  5001]
 [ 1234  5138]] 

 with 22204 correct predictions,  

 5001 Type I errors( False Positives),  

 1234 Type II errors( False Negatives),  

 Accuracy score:  0.7807588171173389 

 Sensitivity:  0.8063402385436284 

 Specificity:  0.7733720034440568 

 ==================================================== 


With 0.3 threshold the Confusion Matrix is  

 [[19080  2987]
 [ 1872  4500]] 

 with 23580 correct predictions,  

 2987 Type I errors( False Positives),  

 1872 Type II errors( False Negatives),  

 Accuracy score:  0.8291430781673055 

 Sensitivity:  0.7062146892655368 

 Specificity:  0.8646395069560883 

 ==================================================== 


With 0.4 threshold the Confusion Matrix is  

 [[20191  1876]
 [ 2517  3855]] 

 with 24046 correct predictions,  

 1876 Type I errors( False Positives),  

 2517 Type II errors( False Negatives),  

 Accuracy score:  0.845529027040332 

 Sensitivity:  0.6049905838041432 

 Specificity:  0.9149861784565188 

 ==================================================== 
```

이항 문제에서는 예측 확률을 클래스 예측으로 변환하는 데 임계값 0.5가 기본적으로 사용됩니다.

임계값을 조정하여 감도 또는 특수성을 높일 수 있습니다.

민감도와 특수성은 역관계가 있습니다. 하나를 늘리면 다른 하나는 항상 감소하고 그 반대도 마찬가지입니다.

임계값 레벨을 높이면 정확도가 높아진다는 것을 알 수 있습니다.

임계값 레벨 조정은 모델 작성 프로세스에서 수행하는 마지막 단계 중 하나여야 합니다.

### ROC - AUC 

ROC 곡선
분류 모델 성능을 시각적으로 측정하는 또 다른 도구는 ROC 곡선입니다. ROC 곡선은 수신기 작동 특성 곡선을 나타냅니다. ROC 곡선은 다양한 분류 임계값 수준에서 분류 모델의 성능을 보여주는 그림입니다.

ROC 곡선은 다양한 임계값 레벨에서 FPR(False Positive Rate)에 대한 True Positive Rate(TPR)를 표시합니다.

실제 양의 비율(TPR)은 리콜이라고도 합니다. TP 대 (TP + FN)의 비율로 정의됩니다.

FPR(False Positive Rate)은 FP 대 (FP + TN)의 비율로 정의됩니다.

ROC 곡선에서는 단일 지점의 TPR(True Positive Rate)과 FPR(False Positive Rate)에 초점을 맞출 것입니다. 이를 통해 다양한 임계값 레벨에서 TPR과 FPR로 구성된 ROC 곡선의 일반적인 성능을 얻을 수 있습니다. 따라서 ROC 곡선은 여러 분류 임계값 수준에서 TPR 대 FPR을 표시합니다. 임계값 레벨을 낮추면 더 많은 항목이 포지티브로 분류될 수 있습니다. 그러면 True Positives(TP)와 False Positives(FP)가 모두 증가합니다.


```python
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')
```

ROC 곡선은 특정 컨텍스트에 대한 민감도와 특수성의 균형을 맞추는 임계값 레벨을 선택하는 데 도움이 됩니다.

### ROC-AUC
ROCAUC는 수신기 작동 특성 - 곡선 아래 영역을 나타냅니다. 분류기 성능을 비교하는 기술입니다. 이 기술에서 우리는 곡선 아래의 면적을 측정합니다. 완벽한 분류기는 ROC AUC가 1인 반면, 순수한 무작위 분류기는 ROC AUC가 0.5입니다.

즉, ROCAUC는 곡선 아래에 있는 ROC 그림의 백분율입니다.


```python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

ROC AUC는 분류기 성능의 단일 숫자 요약입니다. 값이 높을수록 분류기가 더 좋습니다.

우리 모델의 ROCAUC는 1에 접근합니다. 그래서, 분류기가 내일 비가 올지 안 올지 예측하는 것을 잘한다는 결론을 내릴 수 있습니다.


```python
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC)
```

### k-Fold Cross Validation


```python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

평균을 계산하여 교차 검증 정확도를 요약할 수 있습니다.


```python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

원래 모델 점수는 0.8476입니다. 교차 검증 평균 점수는 0.8474입니다. 따라서 교차 검증을 통해 성능이 향상되지 않는다는 결론을 내릴 수 있습니다.

### Hyperparameter Optimization using GridSearch CV


```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```


```python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```


```python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

Output:

GridSearch CV score on test set: 0.8507

원래 모델 테스트 정확도는 0.8501인 반면 그리드 검색 CV 정확도는 0.8507입니다.\
그리드 검색 CV가 이 특정 모델의 성능을 향상시킨다는 것을 알 수 있습니다.

### Results and conclusion

로지스틱 회귀 모형 정확도 점수는 0.8501입니다. 그래서, 이 모델은 호주에 내일 비가 올지 안 올지 예측하는 데 매우 좋은 역할을 합니다.

내일 비가 올 것이라는 관측은 소수입니다. 내일은 비가 오지 않을 것이라는 관측이 대다수입니다.

모형에 과적합 징후가 없습니다.

C 값을 늘리면 테스트 세트 정확도가 높아지고 교육 세트 정확도가 약간 높아집니다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘해야 한다는 결론을 내릴 수 있습니다.

임계값 레벨을 높이면 정확도가 높아집니다.

우리 모델의 ROCAUC는 1에 접근합니다. 그래서, 우리는 우리의 분류기가 내일 비가 올지 안 올지 예측하는 것을 잘한다는 결론을 내릴 수 있습니다.

우리의 원래 모델 정확도 점수는 0.8501인 반면 RFECV 이후 정확도 점수는 0.8500입니다. 따라서 기능 집합을 줄이면 거의 유사한 정확도를 얻을 수 있습니다.

원래 모델에서는 FP = 1175인 반면 FP1 = 1174입니다. 그래서 우리는 대략 같은 수의 오검출을 얻습니다. 또한 FN = 3087인 반면 FN1 = 3091입니다. 그래서 우리는 약간 더 높은 거짓 음성을 얻습니다.

우리의 원래 모델 점수는 0.8476입니다. 교차 검증 평균 점수는 0.8474입니다. 따라서 교차 검증을 통해 성능이 향상되지 않는다는 결론을 내릴 수 있습니다.

우리의 원래 모델 테스트 정확도는 0.8501인 반면 그리드 검색 CV 정확도는 0.8507입니다. 그리드 검색 CV가 이 특정 모델의 성능을 향상시킨다는 것을 알 수 있습니다.
