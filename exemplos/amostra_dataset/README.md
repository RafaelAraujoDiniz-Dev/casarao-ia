# Amostra ilustrativa do dataset CASARÃO IA

Esta pasta contém uma amostra representativa das três sessões de coleta utilizadas no projeto CASARÃO IA.

## Composição

A amostra possui 15 imagens:

- 5 imagens da sessão 1 diurna;
- 5 imagens da sessão 2 diurna;
- 5 imagens da sessão 3 noturna;
- uma imagem por classe em cada sessão.

## Classes

- Museu Casa de Nhozinho
- Palácio Cristo Rei
- Palácio de La Ravardière
- Palácio dos Leões
- Teatro Arthur Azevedo

## Relação com as partições experimentais

- Sessão 1 diurna: origem do conjunto de treinamento;
- Sessão 2 diurna: origem da validação;
- Sessão 3 noturna: origem do teste externo.

Esta pasta possui finalidade exclusivamente ilustrativa. As imagens não formam um novo conjunto de teste e não devem ser utilizadas para recalcular as métricas apresentadas no trabalho.

O dataset balanceado utilizado nos experimentos possui 725 imagens:

- 300 imagens de treinamento;
- 175 imagens de validação diurna;
- 250 imagens de teste externo noturno.

O conjunto noturno permaneceu isolado durante a seleção do modelo e foi utilizado uma única vez após a escolha final do experimento E2.
