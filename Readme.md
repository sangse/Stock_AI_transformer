# Transformer Encoder Model for New Stock


### Data Explain

신규 상장주 데이터를 활용하여 상장 후 4개월 동안의 데이터를 사용했습니다. 데이터는 윈도우 크기를 4~12 사이로 설정하여 분할하였고, 이 데이터를 활용하여 향후 12개 기간의 데이터를 예측했습니다.
데이터 구성에는 현재가, 시가, 고가, 저가, 거래량, 거래대금 등의 기본 요소가 포함되며, 공모가 대비 가격(현재가, 시가, 저가, 고가) 특성을 추가했습니다. 

또한 이전에 리뷰한 time series clustering(https://github.com/sangse/Stock-AI)을 통해 각 시계열에 클러스터링 특성을 추가했습니다. 결과적으로 데이터 형태는 (12 seq len x 11 feature)이 됩니다.

### Transformer Encoder Model
모델은 Transformer의 인코더 블록을 활용하여 학습을 진행했습니다. 이전 리뷰에서는 LTSF(Long Term Time Series Forecasting) 예측에 사용되는 DLinear 모델을 사용해 예측을 시도했습니다. 그러나 보유한 데이터의 최대 길이가 12에 불과하여 트렌드와 계절성 등의 정보를 충분히 포착하기 어려웠습니다. 이에 따라 다양한 분야에서 널리 사용되고 있는 Transformer를 사용하였습니다. 

#### 1)멀티-헤드 어텐션: Transformer의 핵심 특징 중 하나는 멀티-헤드 어텐션(Multi-head Attention)입니다. 이 기법을 통해 입력 데이터의 다양한 특성에 대해 동시에 주의를 기울일 수 있어, 시계열 데이터의 복잡한 패턴을 보다 잘 파악할 수 있습니다. 특히, 입력 시퀀스의 각 타임스텝 사이의 상관관계를 효과적으로 모델링할 수 있습니다.
#### 2)위치 임베딩: Transformer는 입력 데이터의 순서를 고려하지 않기 때문에 위치 임베딩(Position Embedding)을 사용하여 입력 데이터의 순차적인 정보 및 시계열 데이터의 시간 순서를 보존합니다. 이는 시계열 데이터에서 시간 흐름을 고려한 예측을 가능하게 합니다.

<p align='center'>
  <img src = "https://github.com/sangse/Stock_AI_transformer/assets/145996429/8009b09a-e885-4f5b-8f45-80de487d3f06">
</p>

Transformer 모델의 구조는 Encoder Decoder를 기본으로 합니다. 여기서 저는 Transformer의 Encoder block 만을 사용했습니다. Decoder와 같이 사용하게 되면 비용이 늘어나게 되고, 학습 데이터 길이 자체가 길지 않기 때문에 유의미하게 학습을 하지 못하고 다음과 같은 문제가 생길수 있습니다. 

#### [1] 과적합: 많은 수의 파라미터를 학습하면 모델이 훈련 데이터에 과적합될 수 있습니다. 과적합된 모델은 학습 데이터에 너무 최적화되어 일반화 능력이 저하될 수 있습니다. 이로 인해 테스트 데이터나 새로운 데이터에 대한 예측 성능이 저하되고, 결과값이 일정한 값으로 수렴할 수 있습니다.

#### [2] Loss 함수 문제: 너무 많은 파라미터를 가진 모델의 경우, 학습 과정에서 loss 함수가 극단적으로 낮은 값을 가지거나 오히려 일정한 값에 수렴할 수 있습니다. 이 경우 모델이 데이터를 잘 설명하지 못하고 예측 결과가 일정한 값에 수렴할 수 있습니다.

이러한 문제로 Encoder Block만을 사용하기로 결정하였습니다.

### Transformer Model Define
``` python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1),:]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, future_seq=12):
        super(TimeSeriesTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.linear_output = nn.Linear(d_model, 1)



    def forward(self, x, y, mode):
        eval_x = copy.deepcopy(x)

        x = self.linear(x)

        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x) # seq/batch/d_model

        x = x.permute(1, 0, 2)

        output = self.linear_output(x)

        return output

```
### Parameter Setting & DataSet
```python
# Hyperparameters
input_dim = 11  # Number of features
seq_length = 12  # Sequence length
d_model = 128  # Dimension of the model
nhead = 4  # Number of attention heads
num_layers = 2  # Number of Transformer layers
future_seq = 12  # Number of output classes


# Initialize the model
model = TimeSeriesTransformer(input_dim, d_model, nhead, num_layers, future_seq).to(device)
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Loader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
```
### Model Learning

```python
num_epochs = 300  # 학습 에폭 수
num_samples = len(train_loader.dataset)  # 전체 훈련 데이터 수
num_val_samples = len(val_loader.dataset)  # 전체 검증 데이터 수

best_val_loss = float('inf')  # 가장 좋은 검증 손실을 저장하기 위한 변수 초기화

for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0

    # 훈련 루프
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # 그래디언트 초기화
        outputs = model(inputs,targets,mode='eval')  # 모델을 통한 순전파
        loss = criterion(outputs.squeeze(), targets[:,:,0])  # 손실 계산
        loss.backward()  # 역전파 실행
        optimizer.step()  # 매개변수 업데이트

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / num_samples

    # 검증 루프
    model.eval()  # 모델을 평가 모드로 설정
    running_val_loss = 0.0
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for inputs, targets in val_loader:
            outputs = model(inputs,targets,mode='eval')
            loss = criterion(outputs.squeeze(), targets[:,:,0])
            running_val_loss += loss.item() * inputs.size(0)

    val_loss = running_val_loss / num_val_samples
    print(f'Epoch {epoch+1}  - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}')

    # 검증 손실이 현재까지의 최소값보다 낮을 경우, 모델 상태 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'/content/drive/MyDrive/best_model.pth')
        print(f'New best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}')
```
### Prediction Result
아래 그림은 모델의 실험 결과를 보여줍니다. DLinear 모델과 비교했을 때, Transformer 기반 모델은 상승 및 하락 추세를 더 적극적으로 예측하는 모습을 보입니다. 또한 대체로 추세의 방향을 예측하는 능력이 있지만, 예측 방향이 실제와 일치하지 않는 경우도 있습니다. 이러한 예측이 실제 주가 흐름과 다르게 진행될 경우 큰 손실로 이어질 수 있습니다.

모델의 예측 결과 중 일부를 조사해본 결과, 보호예수 기간과 관련이 있음을 발견했습니다. 모델이 바닥을 다지고 상승할 것으로 예상했지만, 실제로는 그렇지 않은 사례들이 있었습니다. 보호예수는 상장 후 1개월이 지나면 상장 예정 주식 수의 15.07%가 시장에 유통되는 제도입니다. 보호예수 해제 시 유동성이 커지면서 주가가 변동될 수 있습니다.

이러한 보호예수 관련 변수를 파악하여 모델에 적절히 추가하는 것이 다음 목표입니다. 또한 시계열 데이터를 LLM(언어 모델)에 파인 튜닝 방식으로 학습시키는 최신 모델이 주목을 받고 있습니다. 보유한 데이터를 활용하여 이러한 실험을 진행할 계획입니다.
<p align='center'>
  <img src = 'https://github.com/sangse/Stock_AI_transformer/assets/145996429/e42c13d3-bbac-48bb-b649-8a21fd4131c2'>
</p>
