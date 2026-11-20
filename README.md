\# CASARÃO IA



Sistema de visão computacional para reconhecimento de edificações históricas de São Luís, desenvolvido como Trabalho de Conclusão de Curso do Bacharelado Interdisciplinar em Ciência e Tecnologia.



\## Edificações reconhecidas



\- Museu Casa de Nhozinho

\- Palácio Cristo Rei

\- Palácio de La Ravardière

\- Palácio dos Leões

\- Teatro Arthur Azevedo



\## Estrutura do projeto



\- `aplicativo\_android/`: projeto Android desenvolvido em Kotlin e Jetpack Compose;

\- `treinamento/scripts/`: scripts de preparação, treinamento, avaliação e conversão;

\- `modelos/`: modelo TensorFlow Lite Float16 e arquivo de rótulos;

\- `resultados/`: métricas, gráficos e relatórios dos experimentos;

\- `exemplos/amostra\_dataset/`: amostra ilustrativa das três sessões;

\- `documentacao/`: materiais metodológicos e instruções adicionais.



\## Modelo de classificação



O modelo final utiliza a arquitetura MobileNetV3Small, inicialmente carregada com pesos ImageNet e posteriormente submetida a ajuste fino nas camadas finais.



\### Resultados principais



\- acurácia na validação diurna: 96,00%;

\- F1-score macro na validação: 0,96;

\- acurácia Top-2 na validação: 100%;

\- acurácia no teste externo noturno: 71,60%;

\- tamanho do modelo TensorFlow Lite Float16: 1,8229 MB;

\- concordância Top-1 entre Keras e TensorFlow Lite: 100%.



\## Dataset experimental



O dataset balanceado contém 725 imagens:



\- 300 imagens de treinamento;

\- 175 imagens de validação diurna;

\- 250 imagens de teste externo noturno.



A pasta `exemplos/amostra\_dataset/` contém uma amostra ilustrativa com uma imagem de cada classe em cada uma das três sessões, totalizando 15 imagens.



O dataset completo não está incluído neste repositório devido ao volume dos arquivos e à finalidade acadêmica das imagens.



\## Aplicativo Android



O protótipo foi desenvolvido com:



\- Kotlin;

\- Jetpack Compose;

\- CameraX 1.6.2;

\- LiteRT 1.4.2;

\- ExifInterface 1.4.2;

\- minSdk 26, correspondente ao Android 8.0 ou superior.



O aplicativo permite:



1\. fotografar uma fachada com a câmera;

2\. selecionar uma imagem da galeria;

3\. executar a classificação localmente;

4\. apresentar o nome da edificação;

5\. exibir confiança e tempo de inferência;

6\. apresentar informações históricas;

7\. rejeitar imagens sem correspondência segura.



\## Regras de confiança



\- 95% ou mais: alta confiança;

\- de 90% a 94,99%: possível correspondência;

\- abaixo de 90%: imagem não reconhecida.



\## Como executar o aplicativo



1\. Abra a pasta `aplicativo\_android/CasaraoIA` no Android Studio.

2\. Aguarde a sincronização do Gradle.

3\. Execute o projeto em um aparelho ou emulador com Android 8.0 ou superior.

4\. Autorize o uso da câmera.

5\. Fotografe uma fachada ou selecione uma imagem da galeria.



O modelo e os rótulos utilizados pelo aplicativo estão em:



```text

aplicativo\_android/CasaraoIA/app/src/main/assets/


## Ambiente de treinamento

Crie um ambiente Python e instale as dependências:

```bash
python -m pip install -r requirements-treinamento.txt
```

Os scripts estão disponíveis em:

```text
treinamento/scripts/
```

## Ambiente de conversão TensorFlow Lite

A conversão final foi realizada em um ambiente separado com TensorFlow 2.20.0.

Para instalar as dependências:

```bash
python -m pip install -r requirements-conversao.txt
```

## Fluxo metodológico

O desenvolvimento do CASARÃO IA seguiu as etapas abaixo:

```text
Coleta das imagens
→ Extração de quadros
→ Conversão HEIC/JPEG
→ Verificação de similaridade
→ Remoção de redundâncias
→ Criação do dataset balanceado
→ Treinamento do experimento E1
→ Ajuste fino no experimento E2
→ Avaliação externa noturna
→ Conversão para TensorFlow Lite
→ Integração ao aplicativo Android
→ Testes funcionais
```

## Observações metodológicas

- A sessão 1 diurna originou o conjunto de treinamento.
- A sessão 2 diurna originou o conjunto de validação.
- A sessão 3 noturna originou o teste externo.
- O conjunto noturno permaneceu isolado durante a seleção do modelo.
- O teste externo noturno foi realizado uma única vez após a escolha final do experimento E2.
- A amostra publicada possui finalidade exclusivamente ilustrativa.
- Modelos Keras, ambientes virtuais, dataset completo e arquivos de compilação não são versionados.

## Autor

Rafael Araujo Diniz

## Finalidade

Projeto acadêmico voltado ao reconhecimento e à valorização do patrimônio histórico de São Luís, Maranhão.

