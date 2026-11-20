from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURACOES DO EXPERIMENTO
# ============================================================

RAIZ_PROJETO = Path("/mnt/f/casarao_ia")
DATASET_RAIZ = RAIZ_PROJETO / "04_dataset_final"
PASTA_TREINO = DATASET_RAIZ / "treino"
PASTA_VALIDACAO = DATASET_RAIZ / "validacao_diurna"
PASTA_TESTE_NOTURNO = DATASET_RAIZ / "teste_noturno"

NOME_EXPERIMENTO = "mobilenetv3small_base_congelada_v1"
PASTA_MODELOS = RAIZ_PROJETO / "08_modelos" / NOME_EXPERIMENTO
PASTA_RESULTADOS = RAIZ_PROJETO / "09_resultados" / NOME_EXPERIMENTO
PASTA_LOGS = PASTA_RESULTADOS / "tensorboard"

SEMENTE = 20261118
ALTURA = 224
LARGURA = 224
TAMANHO_IMAGEM = (ALTURA, LARGURA)
BATCH_SIZE = 32
EPOCAS_MAXIMAS = 30
TAXA_APRENDIZADO = 1e-3
PATIENCE_EARLY_STOPPING = 6
PATIENCE_REDUCAO_LR = 3
FATOR_REDUCAO_LR = 0.2
MIN_LR = 1e-6
DROPOUT = 0.25

# O teste noturno permanece isolado neste primeiro script.
# Altere para True somente apos escolher definitivamente o modelo.
AVALIAR_TESTE_NOTURNO = False

# RTX 3050: mixed_float16 reduz memoria e pode acelerar o treinamento.
# A camada de saida permanece em float32 para estabilidade numerica.
USAR_PRECISAO_MISTA = True

# Se ocorrer ResourceExhaustedError, reduza BATCH_SIZE para 16.


# ============================================================
# REPRODUTIBILIDADE E AMBIENTE
# ============================================================

def configurar_reprodutibilidade() -> None:
    """Define sementes e tenta ativar operacoes deterministicas."""
    os.environ["PYTHONHASHSEED"] = str(SEMENTE)
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    tf.keras.utils.set_random_seed(SEMENTE)

    try:
        tf.config.experimental.enable_op_determinism()
        print("Operacoes deterministicas: ativadas")
    except Exception as erro:
        print(f"Aviso: nao foi possivel ativar determinismo: {erro}")


def configurar_gpu() -> list[tf.config.PhysicalDevice]:
    """Configura crescimento gradual de memoria da GPU."""
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as erro:
            print(f"Aviso ao configurar memory growth: {erro}")

    if gpus:
        print(f"GPU(s) detectada(s): {gpus}")
    else:
        print("AVISO: nenhuma GPU detectada; o treinamento usara CPU.")

    return gpus


def validar_pastas() -> None:
    """Valida a existencia dos subconjuntos do dataset."""
    obrigatorias = [PASTA_TREINO, PASTA_VALIDACAO, PASTA_TESTE_NOTURNO]
    ausentes = [str(pasta) for pasta in obrigatorias if not pasta.exists()]

    if ausentes:
        raise FileNotFoundError(
            "Pastas obrigatorias ausentes:\n- " + "\n- ".join(ausentes)
        )


def contar_imagens(pasta: Path) -> int:
    extensoes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1
        for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    )


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def criar_datasets():
    """Cria datasets de treino, validacao e teste, preservando as classes."""
    argumentos_comuns = dict(
        labels="inferred",
        label_mode="int",
        image_size=TAMANHO_IMAGEM,
        batch_size=BATCH_SIZE,
        seed=SEMENTE,
    )

    treino = tf.keras.utils.image_dataset_from_directory(
        PASTA_TREINO,
        shuffle=True,
        **argumentos_comuns,
    )

    validacao = tf.keras.utils.image_dataset_from_directory(
        PASTA_VALIDACAO,
        shuffle=False,
        **argumentos_comuns,
    )

    teste_noturno = tf.keras.utils.image_dataset_from_directory(
        PASTA_TESTE_NOTURNO,
        shuffle=False,
        **argumentos_comuns,
    )

    classes = treino.class_names

    if classes != validacao.class_names or classes != teste_noturno.class_names:
        raise ValueError(
            "A ordem ou os nomes das classes diferem entre os subconjuntos."
        )

    autotune = tf.data.AUTOTUNE
    treino = treino.prefetch(autotune)
    validacao = validacao.cache().prefetch(autotune)
    teste_noturno = teste_noturno.cache().prefetch(autotune)

    return treino, validacao, teste_noturno, classes


# ============================================================
# MODELO
# ============================================================

def criar_modelo(numero_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Cria a MobileNetV3Small com base congelada e pesos ImageNet."""
    aumento_dados = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEMENTE),
            tf.keras.layers.RandomRotation(0.05, seed=SEMENTE + 1),
            tf.keras.layers.RandomZoom(0.10, seed=SEMENTE + 2),
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                seed=SEMENTE + 3,
            ),
            tf.keras.layers.RandomContrast(0.10, seed=SEMENTE + 4),
        ],
        name="aumento_dados",
    )

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(ALTURA, LARGURA, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
        pooling=None,
    )
    base.trainable = False

    entradas = tf.keras.Input(
        shape=(ALTURA, LARGURA, 3),
        name="imagem_entrada",
    )
    x = aumento_dados(entradas)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout")(x)
    saidas = tf.keras.layers.Dense(
        numero_classes,
        activation="softmax",
        dtype="float32",
        name="classificador",
    )(x)

    modelo = tf.keras.Model(entradas, saidas, name=NOME_EXPERIMENTO)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=TAXA_APRENDIZADO
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acuracia"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=2,
                name="acuracia_top2",
            ),
        ],
    )

    return modelo, base


# ============================================================
# CALLBACKS
# ============================================================

def criar_callbacks() -> list[tf.keras.callbacks.Callback]:
    melhor_modelo = PASTA_MODELOS / "melhor_modelo.keras"
    historico_csv = PASTA_RESULTADOS / "historico_treinamento.csv"

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=melhor_modelo,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=FATOR_REDUCAO_LR,
            patience=PATIENCE_REDUCAO_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            historico_csv,
            separator=";",
            append=False,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=PASTA_LOGS,
            histogram_freq=0,
            write_graph=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


# ============================================================
# REGISTRO E VISUALIZACOES
# ============================================================

def salvar_configuracao(
    classes: list[str],
    gpus: list[tf.config.PhysicalDevice],
    base: tf.keras.Model,
) -> None:
    detalhes_gpu = []
    for gpu in gpus:
        try:
            detalhes_gpu.append(
                tf.config.experimental.get_device_details(gpu)
            )
        except Exception:
            detalhes_gpu.append({"device": str(gpu)})

    configuracao = {
        "experimento": NOME_EXPERIMENTO,
        "data_hora_inicio": datetime.now().isoformat(),
        "sistema": platform.platform(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "gpu_detectada": bool(gpus),
        "detalhes_gpu": detalhes_gpu,
        "precisao_mista": USAR_PRECISAO_MISTA,
        "semente": SEMENTE,
        "tamanho_imagem": [ALTURA, LARGURA],
        "batch_size": BATCH_SIZE,
        "epocas_maximas": EPOCAS_MAXIMAS,
        "taxa_aprendizado": TAXA_APRENDIZADO,
        "patience_early_stopping": PATIENCE_EARLY_STOPPING,
        "patience_reducao_lr": PATIENCE_REDUCAO_LR,
        "dropout": DROPOUT,
        "arquitetura": "MobileNetV3Small",
        "pesos_iniciais": "ImageNet",
        "base_congelada": True,
        "parametros_base": int(base.count_params()),
        "classes": classes,
        "quantidade_classes": len(classes),
        "imagens_treino": contar_imagens(PASTA_TREINO),
        "imagens_validacao": contar_imagens(PASTA_VALIDACAO),
        "imagens_teste_noturno": contar_imagens(PASTA_TESTE_NOTURNO),
        "avaliar_teste_noturno": AVALIAR_TESTE_NOTURNO,
    }

    with (PASTA_RESULTADOS / "configuracao_experimento.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(configuracao, arquivo, ensure_ascii=False, indent=2)

    with (PASTA_MODELOS / "classes.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(classes, arquivo, ensure_ascii=False, indent=2)


def plotar_historico(historico: tf.keras.callbacks.History) -> None:
    dados = historico.history
    epocas = range(1, len(dados["loss"]) + 1)

    figura, eixos = plt.subplots(1, 2, figsize=(13, 5))

    eixos[0].plot(epocas, dados["loss"], label="Treinamento")
    eixos[0].plot(epocas, dados["val_loss"], label="Validação")
    eixos[0].set_title("Perda por época")
    eixos[0].set_xlabel("Época")
    eixos[0].set_ylabel("Perda")
    eixos[0].legend()
    eixos[0].grid(alpha=0.3)

    eixos[1].plot(epocas, dados["acuracia"], label="Treinamento")
    eixos[1].plot(epocas, dados["val_acuracia"], label="Validação")
    eixos[1].set_title("Acurácia por época")
    eixos[1].set_xlabel("Época")
    eixos[1].set_ylabel("Acurácia")
    eixos[1].set_ylim(0, 1.05)
    eixos[1].legend()
    eixos[1].grid(alpha=0.3)

    figura.tight_layout()
    figura.savefig(
        PASTA_RESULTADOS / "curvas_treinamento.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figura)


def avaliar_dataset(
    modelo: tf.keras.Model,
    dataset: tf.data.Dataset,
    classes: list[str],
    nome: str,
) -> dict:
    """Calcula métricas, relatório e matriz de confusão."""
    inicio = time.perf_counter()
    avaliacao = modelo.evaluate(dataset, verbose=1, return_dict=True)
    probabilidades = modelo.predict(dataset, verbose=1)
    tempo = time.perf_counter() - inicio

    verdadeiros = np.concatenate(
        [rotulos.numpy() for _, rotulos in dataset], axis=0
    )
    previstos = np.argmax(probabilidades, axis=1)

    relatorio_dict = classification_report(
        verdadeiros,
        previstos,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    relatorio_texto = classification_report(
        verdadeiros,
        previstos,
        target_names=classes,
        zero_division=0,
    )
    matriz = confusion_matrix(verdadeiros, previstos)

    with (PASTA_RESULTADOS / f"relatorio_{nome}.txt").open(
        "w", encoding="utf-8"
    ) as arquivo:
        arquivo.write(relatorio_texto)

    with (PASTA_RESULTADOS / f"metricas_{nome}.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            {
                "avaliacao_keras": {
                    chave: float(valor) for chave, valor in avaliacao.items()
                },
                "classification_report": relatorio_dict,
                "tempo_total_segundos": tempo,
                "quantidade_imagens": int(len(verdadeiros)),
                "tempo_medio_por_imagem_ms": float(
                    tempo / len(verdadeiros) * 1000
                ),
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    with (PASTA_RESULTADOS / f"matriz_confusao_{nome}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["classe_real/classe_prevista", *classes])
        for classe, linha in zip(classes, matriz):
            escritor.writerow([classe, *linha.tolist()])

    figura, eixo = plt.subplots(figsize=(8, 7))
    imagem = eixo.imshow(matriz, interpolation="nearest", cmap="Blues")
    figura.colorbar(imagem, ax=eixo)
    eixo.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Classe real",
        xlabel="Classe prevista",
        title=f"Matriz de confusão - {nome}",
    )
    plt.setp(eixo.get_xticklabels(), rotation=45, ha="right")

    limite = matriz.max() / 2 if matriz.size else 0
    for linha in range(matriz.shape[0]):
        for coluna in range(matriz.shape[1]):
            eixo.text(
                coluna,
                linha,
                str(matriz[linha, coluna]),
                ha="center",
                va="center",
                color="white" if matriz[linha, coluna] > limite else "black",
            )

    figura.tight_layout()
    figura.savefig(
        PASTA_RESULTADOS / f"matriz_confusao_{nome}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figura)

    return {
        "avaliacao": avaliacao,
        "tempo_segundos": tempo,
        "quantidade": len(verdadeiros),
    }


# ============================================================
# EXECUCAO PRINCIPAL
# ============================================================

def main() -> int:
    inicio_total = time.perf_counter()

    print("=" * 78)
    print("CASARÃO IA - TREINAMENTO MOBILENETV3SMALL")
    print("=" * 78)

    configurar_reprodutibilidade()
    validar_pastas()
    gpus = configurar_gpu()

    if USAR_PRECISAO_MISTA and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Precisão mista: ativada (mixed_float16)")
    else:
        print("Precisão mista: desativada")

    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)

    print(f"Treino:           {PASTA_TREINO}")
    print(f"Validação diurna: {PASTA_VALIDACAO}")
    print(f"Teste noturno:    {PASTA_TESTE_NOTURNO}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Imagem:           {ALTURA}x{LARGURA}")
    print()

    treino, validacao, teste_noturno, classes = criar_datasets()
    print(f"Classes ({len(classes)}): {classes}")

    modelo, base = criar_modelo(len(classes))
    salvar_configuracao(classes, gpus, base)

    modelo.summary()

    callbacks = criar_callbacks()

    print("\nIniciando treinamento da base congelada...")
    historico = modelo.fit(
        treino,
        validation_data=validacao,
        epochs=EPOCAS_MAXIMAS,
        callbacks=callbacks,
        verbose=1,
    )

    modelo.save(PASTA_MODELOS / "modelo_final_epoca.keras")
    plotar_historico(historico)

    melhor_modelo = tf.keras.models.load_model(
        PASTA_MODELOS / "melhor_modelo.keras"
    )

    print("\nAvaliação no conjunto de validação diurna...")
    resultado_validacao = avaliar_dataset(
        melhor_modelo,
        validacao,
        classes,
        "validacao_diurna",
    )

    if AVALIAR_TESTE_NOTURNO:
        print("\nAvaliação no teste noturno...")
        avaliar_dataset(
            melhor_modelo,
            teste_noturno,
            classes,
            "teste_noturno",
        )
    else:
        print(
            "\nTeste noturno preservado. "
            "AVALIAR_TESTE_NOTURNO permanece False."
        )

    tempo_total = time.perf_counter() - inicio_total

    resumo = {
        "experimento": NOME_EXPERIMENTO,
        "epocas_executadas": len(historico.history["loss"]),
        "melhor_val_loss": float(min(historico.history["val_loss"])),
        "melhor_val_acuracia": float(max(historico.history["val_acuracia"])),
        "avaliacao_validacao": {
            chave: float(valor)
            for chave, valor in resultado_validacao["avaliacao"].items()
        },
        "tempo_total_segundos": tempo_total,
        "teste_noturno_avaliado": AVALIAR_TESTE_NOTURNO,
    }

    with (PASTA_RESULTADOS / "resumo_treinamento.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("TREINAMENTO FINALIZADO")
    print("=" * 78)
    print(f"Épocas executadas:       {resumo['epocas_executadas']}")
    print(f"Melhor val_loss:         {resumo['melhor_val_loss']:.4f}")
    print(f"Melhor val_acuracia:     {resumo['melhor_val_acuracia']:.4f}")
    print(f"Tempo total:             {tempo_total / 60:.2f} minutos")
    print(f"Melhor modelo:           {PASTA_MODELOS / 'melhor_modelo.keras'}")
    print(f"Resultados:              {PASTA_RESULTADOS}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
        sys.exit(130)
    except Exception as erro:
        print(f"\nERRO: {type(erro).__name__}: {erro}")
        raise
