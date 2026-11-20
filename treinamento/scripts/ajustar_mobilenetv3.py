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
# CONFIGURACOES DO EXPERIMENTO E2
# ============================================================

RAIZ_PROJETO = Path("/mnt/f/casarao_ia")
DATASET_RAIZ = RAIZ_PROJETO / "04_dataset_final"
PASTA_TREINO = DATASET_RAIZ / "treino"
PASTA_VALIDACAO = DATASET_RAIZ / "validacao_diurna"
PASTA_TESTE_NOTURNO = DATASET_RAIZ / "teste_noturno"

MODELO_E1 = (
    RAIZ_PROJETO
    / "08_modelos"
    / "mobilenetv3small_base_congelada_v1"
    / "melhor_modelo.keras"
)

NOME_EXPERIMENTO = "mobilenetv3small_ajuste_fino_v1"
PASTA_MODELOS = RAIZ_PROJETO / "08_modelos" / NOME_EXPERIMENTO
PASTA_RESULTADOS = RAIZ_PROJETO / "09_resultados" / NOME_EXPERIMENTO
PASTA_LOGS = PASTA_RESULTADOS / "tensorboard"

SEMENTE = 20261118
ALTURA = 224
LARGURA = 224
TAMANHO_IMAGEM = (ALTURA, LARGURA)
BATCH_SIZE = 16
EPOCAS_MAXIMAS = 20
TAXA_APRENDIZADO = 1e-5
ULTIMAS_CAMADAS_DESCONGELADAS = 30
PATIENCE_EARLY_STOPPING = 5
PATIENCE_REDUCAO_LR = 3
FATOR_REDUCAO_LR = 0.2
MIN_LR = 1e-7
USAR_PRECISAO_MISTA = True

# O conjunto noturno continua isolado. Nao altere antes de escolher
# definitivamente entre E1 e E2.
AVALIAR_TESTE_NOTURNO = False


# ============================================================
# AMBIENTE E REPRODUTIBILIDADE
# ============================================================

def configurar_reprodutibilidade() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEMENTE)
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    tf.keras.utils.set_random_seed(SEMENTE)
    try:
        tf.config.experimental.enable_op_determinism()
        print("Operacoes deterministicas: ativadas")
    except Exception as erro:
        print(f"Aviso ao ativar determinismo: {erro}")


def configurar_gpu() -> list[tf.config.PhysicalDevice]:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as erro:
            print(f"Aviso ao configurar memory growth: {erro}")
    print(f"GPU(s) detectada(s): {gpus}" if gpus else "AVISO: treinamento em CPU")
    return gpus


def validar_entradas() -> None:
    caminhos = [PASTA_TREINO, PASTA_VALIDACAO, PASTA_TESTE_NOTURNO, MODELO_E1]
    ausentes = [str(caminho) for caminho in caminhos if not caminho.exists()]
    if ausentes:
        raise FileNotFoundError("Caminhos ausentes:\n- " + "\n- ".join(ausentes))


def contar_imagens(pasta: Path) -> int:
    extensoes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1 for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    )


# ============================================================
# DATASETS
# ============================================================

def criar_datasets():
    argumentos = dict(
        labels="inferred",
        label_mode="int",
        image_size=TAMANHO_IMAGEM,
        batch_size=BATCH_SIZE,
        seed=SEMENTE,
    )

    treino = tf.keras.utils.image_dataset_from_directory(
        PASTA_TREINO, shuffle=True, **argumentos
    )
    validacao = tf.keras.utils.image_dataset_from_directory(
        PASTA_VALIDACAO, shuffle=False, **argumentos
    )
    teste_noturno = tf.keras.utils.image_dataset_from_directory(
        PASTA_TESTE_NOTURNO, shuffle=False, **argumentos
    )

    classes = treino.class_names
    if classes != validacao.class_names or classes != teste_noturno.class_names:
        raise ValueError("As classes ou sua ordem diferem entre os subconjuntos.")

    autotune = tf.data.AUTOTUNE
    treino = treino.prefetch(autotune)
    validacao = validacao.cache().prefetch(autotune)
    teste_noturno = teste_noturno.cache().prefetch(autotune)
    return treino, validacao, teste_noturno, classes


# ============================================================
# AJUSTE FINO
# ============================================================

def localizar_base(modelo: tf.keras.Model) -> tf.keras.Model:
    for camada in modelo.layers:
        if isinstance(camada, tf.keras.Model) and "mobilenetv3small" in camada.name.lower():
            return camada
    raise ValueError("A base MobileNetV3Small nao foi localizada no modelo E1.")


def preparar_modelo() -> tuple[tf.keras.Model, tf.keras.Model, list[str]]:
    print(f"Carregando modelo E1: {MODELO_E1}")
    modelo = tf.keras.models.load_model(MODELO_E1)
    base = localizar_base(modelo)

    # Primeiro, congela integralmente a base.
    base.trainable = True
    for camada in base.layers:
        camada.trainable = False

    # Depois, libera somente as ultimas N camadas, mantendo BatchNormalization
    # congelada para evitar instabilidade com apenas 300 imagens de treino.
    candidatas = base.layers[-ULTIMAS_CAMADAS_DESCONGELADAS:]
    nomes_treinaveis = []

    for camada in candidatas:
        if isinstance(camada, tf.keras.layers.BatchNormalization):
            camada.trainable = False
        else:
            camada.trainable = True
            nomes_treinaveis.append(camada.name)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acuracia"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="acuracia_top2"),
        ],
    )

    return modelo, base, nomes_treinaveis


def criar_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            PASTA_MODELOS / "melhor_modelo.keras",
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
            PASTA_RESULTADOS / "historico_treinamento.csv",
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
# RESULTADOS
# ============================================================

def salvar_configuracao(classes, gpus, modelo, base, nomes_treinaveis) -> None:
    dados = {
        "experimento": NOME_EXPERIMENTO,
        "inicio": datetime.now().isoformat(),
        "modelo_origem_e1": str(MODELO_E1),
        "sistema": platform.platform(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "gpu_detectada": bool(gpus),
        "precisao_mista": USAR_PRECISAO_MISTA,
        "semente": SEMENTE,
        "tamanho_imagem": [ALTURA, LARGURA],
        "batch_size": BATCH_SIZE,
        "epocas_maximas": EPOCAS_MAXIMAS,
        "taxa_aprendizado": TAXA_APRENDIZADO,
        "ultimas_camadas_consideradas": ULTIMAS_CAMADAS_DESCONGELADAS,
        "camadas_base_treinaveis": nomes_treinaveis,
        "quantidade_camadas_base_treinaveis": len(nomes_treinaveis),
        "batch_normalization_congelada": True,
        "parametros_totais": int(modelo.count_params()),
        "parametros_treinaveis": int(sum(np.prod(v.shape) for v in modelo.trainable_weights)),
        "parametros_nao_treinaveis": int(sum(np.prod(v.shape) for v in modelo.non_trainable_weights)),
        "classes": classes,
        "imagens_treino": contar_imagens(PASTA_TREINO),
        "imagens_validacao": contar_imagens(PASTA_VALIDACAO),
        "imagens_teste_noturno": contar_imagens(PASTA_TESTE_NOTURNO),
        "teste_noturno_avaliado": AVALIAR_TESTE_NOTURNO,
    }
    with (PASTA_RESULTADOS / "configuracao_experimento.json").open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)
    with (PASTA_MODELOS / "classes.json").open("w", encoding="utf-8") as arquivo:
        json.dump(classes, arquivo, ensure_ascii=False, indent=2)


def plotar_historico(historico: tf.keras.callbacks.History) -> None:
    dados = historico.history
    epocas = range(1, len(dados["loss"]) + 1)
    figura, eixos = plt.subplots(1, 2, figsize=(13, 5))

    eixos[0].plot(epocas, dados["loss"], label="Treinamento")
    eixos[0].plot(epocas, dados["val_loss"], label="Validacao")
    eixos[0].set(title="Perda por epoca", xlabel="Epoca", ylabel="Perda")
    eixos[0].legend()
    eixos[0].grid(alpha=0.3)

    eixos[1].plot(epocas, dados["acuracia"], label="Treinamento")
    eixos[1].plot(epocas, dados["val_acuracia"], label="Validacao")
    eixos[1].set(title="Acuracia por epoca", xlabel="Epoca", ylabel="Acuracia", ylim=(0, 1.05))
    eixos[1].legend()
    eixos[1].grid(alpha=0.3)

    figura.tight_layout()
    figura.savefig(PASTA_RESULTADOS / "curvas_treinamento.png", dpi=200, bbox_inches="tight")
    plt.close(figura)


def avaliar_dataset(modelo, dataset, classes, nome: str) -> dict:
    inicio = time.perf_counter()
    avaliacao = modelo.evaluate(dataset, verbose=1, return_dict=True)
    probabilidades = modelo.predict(dataset, verbose=1)
    tempo = time.perf_counter() - inicio

    verdadeiros = np.concatenate([rotulos.numpy() for _, rotulos in dataset])
    previstos = np.argmax(probabilidades, axis=1)
    matriz = confusion_matrix(verdadeiros, previstos)

    relatorio_dict = classification_report(
        verdadeiros, previstos, target_names=classes, output_dict=True, zero_division=0
    )
    relatorio_texto = classification_report(
        verdadeiros, previstos, target_names=classes, zero_division=0
    )

    with (PASTA_RESULTADOS / f"relatorio_{nome}.txt").open("w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio_texto)

    with (PASTA_RESULTADOS / f"metricas_{nome}.json").open("w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "avaliacao_keras": {k: float(v) for k, v in avaliacao.items()},
                "classification_report": relatorio_dict,
                "tempo_total_segundos": tempo,
                "quantidade_imagens": int(len(verdadeiros)),
                "tempo_medio_por_imagem_ms": float(tempo / len(verdadeiros) * 1000),
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
    imagem = eixo.imshow(matriz, cmap="Blues")
    figura.colorbar(imagem, ax=eixo)
    eixo.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Classe prevista",
        ylabel="Classe real",
        title=f"Matriz de confusao - {nome}",
    )
    plt.setp(eixo.get_xticklabels(), rotation=45, ha="right")
    limite = matriz.max() / 2
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            eixo.text(j, i, str(matriz[i, j]), ha="center", va="center",
                      color="white" if matriz[i, j] > limite else "black")
    figura.tight_layout()
    figura.savefig(PASTA_RESULTADOS / f"matriz_confusao_{nome}.png", dpi=200, bbox_inches="tight")
    plt.close(figura)

    return {"avaliacao": avaliacao, "relatorio": relatorio_dict, "matriz": matriz.tolist()}


# ============================================================
# EXECUCAO
# ============================================================

def main() -> int:
    inicio_total = time.perf_counter()
    print("=" * 78)
    print("CASARAO IA - E2 AJUSTE FINO MOBILENETV3SMALL")
    print("=" * 78)

    configurar_reprodutibilidade()
    validar_entradas()
    gpus = configurar_gpu()

    if USAR_PRECISAO_MISTA and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Precisao mista: ativada (mixed_float16)")

    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)

    treino, validacao, teste_noturno, classes = criar_datasets()
    modelo, base, nomes_treinaveis = preparar_modelo()
    salvar_configuracao(classes, gpus, modelo, base, nomes_treinaveis)

    print(f"Classes: {classes}")
    print(f"Camadas finais consideradas: {ULTIMAS_CAMADAS_DESCONGELADAS}")
    print(f"Camadas efetivamente treinaveis na base: {len(nomes_treinaveis)}")
    print(f"Taxa de aprendizado: {TAXA_APRENDIZADO}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Parametros treinaveis: {sum(np.prod(v.shape) for v in modelo.trainable_weights):,}")
    print("Camadas treinaveis da base:")
    for nome in nomes_treinaveis:
        print(f"- {nome}")

    modelo.summary()

    historico = modelo.fit(
        treino,
        validation_data=validacao,
        epochs=EPOCAS_MAXIMAS,
        callbacks=criar_callbacks(),
        verbose=1,
    )

    modelo.save(PASTA_MODELOS / "modelo_final_epoca.keras")
    plotar_historico(historico)

    melhor_modelo = tf.keras.models.load_model(PASTA_MODELOS / "melhor_modelo.keras")

    print("\nAvaliando E2 na validacao diurna...")
    resultado_validacao = avaliar_dataset(
        melhor_modelo, validacao, classes, "validacao_diurna"
    )

    if AVALIAR_TESTE_NOTURNO:
        print("\nAvaliando teste noturno...")
        avaliar_dataset(melhor_modelo, teste_noturno, classes, "teste_noturno")
    else:
        print("\nTeste noturno preservado. AVALIAR_TESTE_NOTURNO=False.")

    tempo_total = time.perf_counter() - inicio_total
    resumo = {
        "experimento": NOME_EXPERIMENTO,
        "epocas_executadas": len(historico.history["loss"]),
        "melhor_val_loss": float(min(historico.history["val_loss"])),
        "melhor_val_acuracia": float(max(historico.history["val_acuracia"])),
        "avaliacao_validacao": {k: float(v) for k, v in resultado_validacao["avaliacao"].items()},
        "tempo_total_segundos": tempo_total,
        "teste_noturno_avaliado": AVALIAR_TESTE_NOTURNO,
    }
    with (PASTA_RESULTADOS / "resumo_treinamento.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("E2 FINALIZADO")
    print("=" * 78)
    print(f"Epocas executadas:   {resumo['epocas_executadas']}")
    print(f"Melhor val_loss:     {resumo['melhor_val_loss']:.4f}")
    print(f"Melhor val_acuracia: {resumo['melhor_val_acuracia']:.4f}")
    print(f"Tempo total:         {tempo_total / 60:.2f} minutos")
    print(f"Melhor modelo:       {PASTA_MODELOS / 'melhor_modelo.keras'}")
    print(f"Resultados:          {PASTA_RESULTADOS}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAjuste fino interrompido pelo usuario.")
        sys.exit(130)
    except Exception as erro:
        print(f"\nERRO: {type(erro).__name__}: {erro}")
        raise
