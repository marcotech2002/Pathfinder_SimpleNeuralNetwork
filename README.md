# Rede Neural Simples para Reconhecimento de Dígitos (MNIST)

Este projeto implementa uma rede neural artificial para reconhecimento de dígitos manuscritos utilizando o dataset MNIST. A rede neural possui três camadas e é treinada por meio de aprendizado supervisionado.

## Ferramentas utilizadas

![Logo do VS Code](https://code.visualstudio.com/assets/apple-touch-icon.png) ![Logo do Python](https://static.wixstatic.com/media/4bef97_3fca4225935f490783ac9ecb3f27a8b1~mv2.png/v1/fill/w_256,h_256,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/python_logo.png)

- Visual Studio Code: ambiente de desenvolvimento
- Python: Linguagem de programação

## Estrutura do Código

### Definição da Rede Neural (neuralNetwork)

A classe `neuralNetwork` define a arquitetura da rede e seus principais métodos:

- Camadas:
  - Entrada: 784 neurônios (28×28 pixels das imagens).
  - Oculta: 100 neurônios.
  - Saída: 10 neurônios (representando os dígitos de 0 a 9).
- Função de ativação: Sigmoide (scipy.special.expit).
- Pesos iniciais: Valores aleatórios normalizados.
- Taxa de aprendizado: Definida pelo usuário (padrão: 0.1).
- Métodos:
  - `train(inputs_list, targets_list)` – Treina a rede ajustando os pesos com base no erro.
  - `query(inputs_list)` – Faz previsões a partir de novos dados de entrada.
 
### Treinamento da rede

- Os dados são carregados do arquivo `mnist_train_100.csv`.
- Os valores das imagens (0-255) são normalizados para a faixa [0.01, 1.0].
- As saídas esperadas são definidas em one-hot encoding.
- A rede é treinada por 5 épocas, ajustando os pesos a cada iteração.

### Teste da Rede

- Os dados de teste são lidos do arquivo `mnist_test_10.csv`.
- A rede processa as imagens e gera previsões.
- O resultado é comparado com os valores reais e armazenado na lista scorecard.

### Cálculo da Acurácia

- O percentual de acertos é calculado pela fórmula:

    `acurracy = scorecard.count(1) / len(scorecard) * 100.00`

- A acurácia final é exibida no console.

### Como Executar

1. Clone este repositório
2. Instale as dependências necessárias (caso ainda não tenha):

    `pip install numpy scipy matplotlib`

3. Execute o script principal:

    `python network.py`

### Observações

- Certifique-se de que os arquivos `mnist_train_100.csv` e `mnist_test_10.csv` estão na mesma pasta do script.
- O modelo pode ser aprimorado aumentando o número de neurônios, ajustando a taxa de aprendizado ou treinando com um dataset maior.
