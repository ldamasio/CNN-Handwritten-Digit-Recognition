

# Propósito

Esse código treina uma CNN para reconhecer dígitos manuscritos do MNIST e salva o modelo treinado. 

### Como usar

Instale as dependências: 
`
pip install tensorflow matplotlib numpy
`
Rode o script principal: 
`python main.py`

### Princiais Funcionalidades

- Utiliza a base de dados MNIST com 70.000 imagens (60.000 para treino e 10.000 para teste).
- Implementação de uma CNN com múltiplas camadas convolucionais e de pooling.
- Treinamento do modelo com 10 épocas e otimização usando Adam.
- Visualização dos resultados, incluindo acurácia e exemplos de predições.

### Arquitetura da CNN
- 3 camadas convolucionais com ativação ReLU
- 2 camadas de pooling (MaxPooling 2x2)
- Camada totalmente conectada (64 neurônios, ReLU)
- Camada de saída (10 neurônios, softmax)

