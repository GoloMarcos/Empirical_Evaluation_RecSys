# Avaliação Empírica de Técnicas de Recomendação para Previsão de Notas de Usuários para Filmes

- Marcos Paulo Silva Gôlo: marcosgolo@usp.br

- Competição: https://www.kaggle.com/competitions/scc5966

## Métodos explorados:
- ItemkNN
- UserkNN
- Baseline
- SVD-AL
- SVD-GD
- FBCkNN baseado em:
  - Gênero dos filmes
  - Reviews representadas pela bag-of-words
  - Reviews representadas pelo BERT
- Filtragem Híbrida monolitica (ItemkNN + FBC-KNN - Gênero)

## Resultado dos métodos
![Results](/imagem/results.png)
