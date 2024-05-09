# Transformer Encoder Model predict for new stock


### Data Explain
신규상장주 데이터를 이용했으며, 상장후 4개월 간의 데이터를 사용하였다. 4개월 간의 데이터를 window size를 4~12 사이즈를 설정하여 잘라주었고 이 데이터가 미래의 12개의 데이터를 예측하는 과정을 진행하였다.
데이터의 특성의 구성으로는 현재가,현재가,시가,고가,저가,거래량,거래대금 이 기본적으로 포함되며, 공모가대비가격(현재가,시가,저가,고가) 특성을 축가해 주었으며 이전에 리뷰했던 time series clustering을 통해
각 시계열에 clustering 특성을 추가해 주었다. 결론적으로 데이터 형태는 (12 seq len x 11 feature)가 된다.


### Transformer Encoder Block
모델은 Transformer의 Encoder Block을 사용하였다. 
