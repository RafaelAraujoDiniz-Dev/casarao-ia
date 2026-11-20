from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

RAIZ_PROJETO = Path("/mnt/f/casarao_ia")
MODELO_FINAL = (
    RAIZ_PROJETO / "08_modelos" / "mobilenetv3small_ajuste_fino_v1" / "melhor_modelo.keras"
)
PASTA_TESTE = RAIZ_PROJETO / "04_dataset_final" / "teste_noturno"
PASTA_RESULTADOS = (
    RAIZ_PROJETO / "09_resultados" / "mobilenetv3small_ajuste_fino_v1" / "teste_noturno_final"
)

SEMENTE = 20261118
TAMANHO_IMAGEM = (224, 224)
BATCH_SIZE = 16


def configurar_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("GPUs:", gpus)


def main() -> int:
    if not MODELO_FINAL.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODELO_FINAL}")
    if not PASTA_TESTE.exists():
        raise FileNotFoundError(f"Teste noturno não encontrado: {PASTA_TESTE}")

    configurar_gpu()
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    dataset = tf.keras.utils.image_dataset_from_directory(
        PASTA_TESTE,
        labels="inferred",
        label_mode="int",
        image_size=TAMANHO_IMAGEM,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEMENTE,
    )
    classes = dataset.class_names
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

    print("Classes:", classes)
    print("Carregando modelo final E2:", MODELO_FINAL)
    modelo = tf.keras.models.load_model(MODELO_FINAL)

    inicio = time.perf_counter()
    avaliacao = modelo.evaluate(dataset, verbose=1, return_dict=True)
    probabilidades = modelo.predict(dataset, verbose=1)
    tempo = time.perf_counter() - inicio

    verdadeiros = np.concatenate([rotulos.numpy() for _, rotulos in dataset])
    previstos = np.argmax(probabilidades, axis=1)
    confiancas = np.max(probabilidades, axis=1)
    matriz = confusion_matrix(verdadeiros, previstos)

    relatorio_texto = classification_report(
        verdadeiros, previstos, target_names=classes, zero_division=0
    )
    relatorio_dict = classification_report(
        verdadeiros, previstos, target_names=classes, output_dict=True, zero_division=0
    )

    (PASTA_RESULTADOS / "relatorio_teste_noturno.txt").write_text(
        relatorio_texto, encoding="utf-8"
    )

    with (PASTA_RESULTADOS / "metricas_teste_noturno.json").open("w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "modelo": str(MODELO_FINAL),
                "avaliacao_keras": {k: float(v) for k, v in avaliacao.items()},
                "classification_report": relatorio_dict,
                "quantidade_imagens": int(len(verdadeiros)),
                "tempo_total_segundos": float(tempo),
                "tempo_medio_por_imagem_ms": float(tempo / len(verdadeiros) * 1000),
                "confianca_media": float(np.mean(confiancas)),
                "confianca_mediana": float(np.median(confiancas)),
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    with (PASTA_RESULTADOS / "matriz_confusao_teste_noturno.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["classe_real/classe_prevista", *classes])
        for classe, linha in zip(classes, matriz):
            escritor.writerow([classe, *linha.tolist()])

    with (PASTA_RESULTADOS / "previsoes_teste_noturno.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as arquivo:
        campos = ["indice", "classe_real", "classe_prevista", "confianca", "correta"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        for indice, (real, previsto, confianca) in enumerate(
            zip(verdadeiros, previstos, confiancas), start=1
        ):
            escritor.writerow(
                {
                    "indice": indice,
                    "classe_real": classes[int(real)],
                    "classe_prevista": classes[int(previsto)],
                    "confianca": f"{float(confianca):.8f}",
                    "correta": int(real == previsto),
                }
            )

    figura, eixo = plt.subplots(figsize=(8, 7))
    imagem = eixo.imshow(matriz, cmap="Blues")
    figura.colorbar(imagem, ax=eixo)
    eixo.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Classe prevista",
        ylabel="Classe real",
        title="Matriz de confusão - teste noturno final",
    )
    plt.setp(eixo.get_xticklabels(), rotation=45, ha="right")
    limite = matriz.max() / 2
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            eixo.text(
                j, i, str(matriz[i, j]), ha="center", va="center",
                color="white" if matriz[i, j] > limite else "black",
            )
    figura.tight_layout()
    figura.savefig(
        PASTA_RESULTADOS / "matriz_confusao_teste_noturno.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figura)

    print("\n" + "=" * 78)
    print("AVALIAÇÃO NOTURNA FINALIZADA")
    print("=" * 78)
    print(relatorio_texto)
    print("Métricas Keras:", avaliacao)
    print(f"Tempo total: {tempo:.2f} s")
    print(f"Tempo médio por imagem: {tempo / len(verdadeiros) * 1000:.2f} ms")
    print(f"Confiança média: {np.mean(confiancas):.4f}")
    print("Resultados:", PASTA_RESULTADOS)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as erro:
        print(f"ERRO: {type(erro).__name__}: {erro}")
        raise
