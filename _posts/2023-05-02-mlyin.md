import pandas, matplotlib및 numpy이전 단원에서 했던 것처럼 ufo 스프레드시트를 가져옵니다. 샘플 데이터 세트를 살펴볼 수 있습니다.


```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('ufos.csv')
ufos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>



ufo 데이터를 새 제목이 있는 작은 데이터 프레임으로 변환합니다. 필드 의 고유 값을 확인하십시오 Country.


```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```




    array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)



이제 null 값을 삭제하고 1-60초 사이의 관찰만 가져옴으로써 처리해야 하는 데이터의 양을 줄일 수 있습니다.


```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 25863 entries, 2 to 80330
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Seconds    25863 non-null  float64
     1   Country    25863 non-null  object 
     2   Latitude   25863 non-null  float64
     3   Longitude  25863 non-null  float64
    dtypes: float64(3), object(1)
    memory usage: 1010.3+ KB
    

Scikit-learn의 라이브러리를 가져와 LabelEncoder국가의 텍스트 값을 숫자로 변환합니다.

LabelEncoder는 데이터를 사전순으로 인코딩합니다.


```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>



이제 데이터를 교육 그룹과 테스트 그룹으로 나누어 모델을 교육할 준비를 할 수 있습니다.

련하려는 세 가지 기능을 X 벡터로 선택하면 y 벡터가 Country. 를 입력하고 Seconds반환 할 국가 ID를 얻을 수 있기를 원합니다 .LatitudeLongitude


```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

로지스틱 회귀를 사용하여 모델을 교육합니다.


```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs',max_iter=10000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        41
               1       0.85      0.46      0.60       250
               2       1.00      1.00      1.00         8
               3       1.00      1.00      1.00       131
               4       0.97      1.00      0.98      4743
    
        accuracy                           0.97      5173
       macro avg       0.96      0.89      0.92      5173
    weighted avg       0.97      0.97      0.97      5173
    
    Predicted labels:  [4 4 4 ... 3 4 4]
    Accuracy:  0.9698434177459888
    

정확도 는 나쁘지 않습니다 ( 약 95%) .CountryLatitude/Longitude

생성한 모델은 and Country에서 추론할 수 있어야 하므로 그다지 혁신적이지는 않지만 정리하고 내보낸 원시 데이터에서 교육을 시도한 다음 웹 앱에서 이 모델을 사용하는 것은 좋은 연습입니다.LatitudeLongitude

#### 연습 - 모델 '피클'

이제 모델을 피클 할 시간입니다! 몇 줄의 코드로 그렇게 할 수 있습니다. 피클링 되면 피클링된 모델을 로드하고 초, 위도 및 경도 값이 포함된 샘플 데이터 배열에 대해 테스트합니다.


```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
```

#### Flask 앱 빌드

이제 Flask 앱을 ​​빌드하여 모델을 호출하고 유사한 결과를 반환하지만 더 시각적으로 만족스러운 방식으로 반환할 수 있습니다.

ufo-model.pkl 파일이 있는 notebook.ipynb 파일 옆에 web-app 이라는 폴더를 생성하여 시작합니다 .

해당 폴더에 css 폴더가 있는 static 폴더 와 templates 폴더를 세 개 더 만듭니다 . 이제 다음 파일과 디렉터리가 있어야 합니다.

web-app/\
  static/\
    css/\
  templates/\
notebook.ipynb\
ufo-model.pkl

web-app 폴더 에 가장 먼저 생성할 파일은 requirements.txt 파일입니다. JavaScript 앱의 package.json 과 마찬가지로 이 파일은 앱에 필요한 종속성을 나열합니다. requirements.txt 에 다음 줄을 추가합니다.

scikit-learn\
pandas\
numpy\
flask

이제 web-app 로 이동하여 이 파일을 실행합니다 .

cd web-app

터미널 유형에서 requirements.txtpip install 에 나열된 라이브러리를 설치하려면 다음을 입력하십시오 .

pip install -r requirements.txt

이제 앱을 완료하기 위해 세 개의 파일을 더 만들 준비가 되었습니다.

1. 루트에 app.py를 만듭니다 .
2. 템플릿 디렉토리 에 index.html을 생성합니다 .
3. static/css 디렉토리 에 styles.css를 생성합니다 .

몇 가지 스타일을 사용하여 styles.css 파일을 빌드합니다 .


```python
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```

다음으로 index.html 파일을 빌드합니다 .


```python
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>🛸 UFO Appearance Prediction! 👽</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
  </head>

  <body>
    <div class="grid">

      <div class="box">

        <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>

        <form action="{{ url_for('predict')}}" method="post">
          <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
          <input type="text" name="latitude" placeholder="Latitude" required="required" />
          <input type="text" name="longitude" placeholder="Longitude" required="required" />
          <button type="submit" class="btn">Predict country where the UFO is seen</button>
        </form>

        <p>{{ prediction_text }}</p>

      </div>

    </div>

  </body>
</html>
```

이 파일의 템플릿을 살펴보십시오. 예측 텍스트와 같이 앱에서 제공할 변수 주변의 '콧수염' 구문에 주목하세요 {{}}. 경로에 예측을 게시하는 양식도 있습니다 /predict.

마지막으로 모델 소비 및 예측 표시를 구동하는 Python 파일을 빌드할 준비가 되었습니다.

추가 app.py


```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```

#### 이제 모든 준비는 끝났습니다. 웹 페이지를 실행하여 UFO가 목격된 위치에 대한 질문에 대한 답을 얻을 수 있습니다!


```python
!pip install IPython
from IPython.display import Image
```

    Requirement already satisfied: IPython in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (8.12.0)
    Requirement already satisfied: backcall in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (0.2.0)
    Requirement already satisfied: decorator in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (0.18.2)
    Requirement already satisfied: matplotlib-inline in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (0.1.6)
    Requirement already satisfied: pickleshare in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (3.0.38)
    Requirement already satisfied: pygments>=2.4.0 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (2.14.0)
    Requirement already satisfied: stack-data in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (0.6.2)
    Requirement already satisfied: traitlets>=5 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (5.9.0)
    Requirement already satisfied: colorama in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from IPython) (0.4.6)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from jedi>=0.16->IPython) (0.8.3)
    Requirement already satisfied: wcwidth in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->IPython) (0.2.6)
    Requirement already satisfied: executing>=1.2.0 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from stack-data->IPython) (1.2.0)
    Requirement already satisfied: asttokens>=2.1.0 in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from stack-data->IPython) (2.2.1)
    Requirement already satisfied: pure-eval in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from stack-data->IPython) (0.2.2)
    Requirement already satisfied: six in c:\users\xkrdu\appdata\local\programs\python\python311\lib\site-packages (from asttokens>=2.1.0->stack-data->IPython) (1.16.0)
    


```python
Image("w01.png")
```




    
![png](output_36_0.png)
    



웹 페이지를 열면 이런 식으로 나옵니다. 


```python
Image("w02.png")
```




    
![png](output_38_0.png)
    



초, 위도, 경도값을 넣어줍니다. 


```python
Image("w03.png")
```




    
![png](output_40_0.png)
    



UK가 나옵니다. 
