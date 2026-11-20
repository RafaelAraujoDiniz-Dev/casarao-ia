from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ_PROJETO = Path("/mnt/f/casarao_ia")

MODELO_KERAS = (
    RAIZ_PROJETO
    / "08_modelos"
    / "mobilenetv3small_ajuste_fino_v1"
    / "melhor_modelo.keras"
)

PASTA_MODELO = (
    RAIZ_PROJETO
    / "08_modelos"
    / "mobilenetv3small_ajuste_fino_v1"
)

PASTA_SAIDA = PASTA_MODELO / "tflite"
PASTA_SAVED_MODEL = PASTA_SAIDA / "saved_model_inferencia"

PASTA_VALIDACAO = (
    RAIZ_PROJETO
    / "04_dataset_final"
    / "validacao_diurna"
)

ARQUIVO_CLASSES = PASTA_MODELO / "classes.json"

TAMANHO_IMAGEM = (224, 224)
BATCH_SIZE = 1
SEMENTE = 20261118
LIMITE_IMAGENS_VALIDACAO = 175

GERAR_FLOAT32 = True
GERAR_FLOAT16 = True

# Mantem apenas operadores nativos do TFLite/LiteRT.
# Nao habilitar SELECT_TF_OPS neste primeiro pacote Android.
OPERADORES_SUPORTADOS = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]


# ============================================================
# VALIDACAO E UTILITARIOS
# ============================================================

def validar_entradas() -> None:
    caminhos = [
        MODELO_KERAS,
        PASTA_VALIDACAO,
        ARQUIVO_CLASSES,
    ]

    ausentes = [
        str(caminho)
        for caminho in caminhos
        if not caminho.exists()
    ]

    if ausentes:
        raise FileNotFoundError(
            "Caminhos obrigatorios ausentes:\n- "
            + "\n- ".join(ausentes)
        )


def carregar_classes() -> list[str]:
    with ARQUIVO_CLASSES.open(
        "r",
        encoding="utf-8",
    ) as arquivo:
        classes = json.load(arquivo)

    if not isinstance(classes, list) or not classes:
        raise ValueError(
            "classes.json nao contem uma lista valida."
        )

    return classes


def salvar_bytes(
    caminho: Path,
    dados: bytes,
) -> None:
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    caminho.write_bytes(dados)

    if (
        not caminho.exists()
        or caminho.stat().st_size == 0
    ):
        raise IOError(
            f"Falha ao salvar o modelo: {caminho}"
        )


def salvar_labels(classes: list[str]) -> Path:
    destino = PASTA_SAIDA / "labels.txt"

    destino.write_text(
        "\n".join(classes) + "\n",
        encoding="utf-8",
    )

    return destino


# ============================================================
# MODELO EXCLUSIVO DE INFERENCIA
# ============================================================

def localizar_camada(
    modelo: tf.keras.Model,
    nome: str,
) -> tf.keras.layers.Layer:
    try:
        return modelo.get_layer(nome)
    except ValueError as erro:
        nomes = [camada.name for camada in modelo.layers]

        raise ValueError(
            f"Camada '{nome}' nao encontrada. "
            f"Camadas disponiveis: {nomes}"
        ) from erro


def criar_modelo_inferencia(
    modelo_treinado: tf.keras.Model,
) -> tf.keras.Model:
    """
    Cria um modelo exclusivo para inferencia.

    O bloco de aumento de dados nao e incluido. O modelo preserva:
    - entrada float32 224x224x3;
    - pre-processamento interno da MobileNetV3Small;
    - base ajustada no experimento E2;
    - GlobalAveragePooling2D;
    - classificador Softmax.

    A camada Dropout e omitida porque permanece inativa durante a
    inferencia e nao altera o resultado deterministico.
    """

    entrada = tf.keras.Input(
        shape=(224, 224, 3),
        batch_size=1,
        dtype=tf.float32,
        name="imagem_entrada",
    )

    base = localizar_camada(
        modelo_treinado,
        "MobileNetV3Small",
    )

    global_pooling = localizar_camada(
        modelo_treinado,
        "global_average_pooling",
    )

    classificador = localizar_camada(
        modelo_treinado,
        "classificador",
    )

    base.trainable = False

    x = base(
        entrada,
        training=False,
    )

    x = global_pooling(x)

    saida = classificador(x)

    # A camada classificadora ja foi criada com dtype float32.
    # O cast explicito preserva uma saida float32 para o Android.
    saida = tf.keras.layers.Activation(
        "linear",
        dtype="float32",
        name="probabilidades",
    )(saida)

    modelo_inferencia = tf.keras.Model(
        inputs=entrada,
        outputs=saida,
        name="casarao_ia_inferencia",
    )

    return modelo_inferencia


# ============================================================
# EXPORTACAO SAVEDMODEL
# ============================================================

def exportar_saved_model(
    modelo: tf.keras.Model,
) -> Path:
    if PASTA_SAVED_MODEL.exists():
        shutil.rmtree(PASTA_SAVED_MODEL)

    PASTA_SAVED_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, 224, 224, 3],
                dtype=tf.float32,
                name="imagem_entrada",
            )
        ]
    )
    def servir(imagem):
        probabilidades = modelo(
            imagem,
            training=False,
        )

        return {
            "probabilidades": tf.cast(
                probabilidades,
                tf.float32,
            )
        }

    tf.saved_model.save(
        modelo,
        str(PASTA_SAVED_MODEL),
        signatures={
            "serving_default": servir,
        },
    )

    arquivo_pb = PASTA_SAVED_MODEL / "saved_model.pb"

    if not arquivo_pb.exists():
        raise IOError(
            "O SavedModel nao foi exportado corretamente."
        )

    return PASTA_SAVED_MODEL


# ============================================================
# CONVERSAO TFLITE
# ============================================================

def criar_conversor_saved_model(
    pasta_saved_model: Path,
) -> tf.lite.TFLiteConverter:
    conversor = (
        tf.lite.TFLiteConverter.from_saved_model(
            str(pasta_saved_model)
        )
    )

    conversor.target_spec.supported_ops = (
        OPERADORES_SUPORTADOS
    )

    conversor.experimental_new_converter = True

    return conversor


def converter_float32(
    pasta_saved_model: Path,
) -> Path:
    conversor = criar_conversor_saved_model(
        pasta_saved_model
    )

    modelo_tflite = conversor.convert()

    destino = (
        PASTA_SAIDA
        / "casarao_ia_mobilenetv3_float32.tflite"
    )

    salvar_bytes(
        destino,
        modelo_tflite,
    )

    return destino


def converter_float16(
    pasta_saved_model: Path,
) -> Path:
    conversor = criar_conversor_saved_model(
        pasta_saved_model
    )

    conversor.optimizations = [
        tf.lite.Optimize.DEFAULT,
    ]

    conversor.target_spec.supported_types = [
        tf.float16,
    ]

    modelo_tflite = conversor.convert()

    destino = (
        PASTA_SAIDA
        / "casarao_ia_mobilenetv3_float16.tflite"
    )

    salvar_bytes(
        destino,
        modelo_tflite,
    )

    return destino


# ============================================================
# DATASET DE VALIDACAO
# ============================================================

def criar_dataset_validacao():
    dataset = (
        tf.keras.utils.image_dataset_from_directory(
            PASTA_VALIDACAO,
            labels="inferred",
            label_mode="int",
            image_size=TAMANHO_IMAGEM,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=SEMENTE,
        )
    )

    classes = dataset.class_names

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset, classes


# ============================================================
# INSPECAO E VALIDACAO TFLITE
# ============================================================

def criar_interpretador(
    caminho_modelo: Path,
) -> tf.lite.Interpreter:
    interpretador = tf.lite.Interpreter(
        model_path=str(caminho_modelo),
        num_threads=4,
    )

    interpretador.allocate_tensors()

    return interpretador


def detalhes_interpretador(
    caminho_modelo: Path,
) -> dict:
    interpretador = criar_interpretador(
        caminho_modelo
    )

    entrada = interpretador.get_input_details()[0]
    saida = interpretador.get_output_details()[0]

    return {
        "entrada_nome": entrada["name"],
        "entrada_shape": entrada["shape"].tolist(),
        "entrada_dtype": np.dtype(
            entrada["dtype"]
        ).name,
        "entrada_quantizacao": [
            float(entrada["quantization"][0]),
            int(entrada["quantization"][1]),
        ],
        "saida_nome": saida["name"],
        "saida_shape": saida["shape"].tolist(),
        "saida_dtype": np.dtype(
            saida["dtype"]
        ).name,
        "saida_quantizacao": [
            float(saida["quantization"][0]),
            int(saida["quantization"][1]),
        ],
    }


def executar_tflite(
    interpretador: tf.lite.Interpreter,
    imagem: np.ndarray,
) -> np.ndarray:
    entrada = interpretador.get_input_details()[0]
    saida = interpretador.get_output_details()[0]

    tensor = np.expand_dims(
        imagem,
        axis=0,
    ).astype(entrada["dtype"])

    interpretador.set_tensor(
        entrada["index"],
        tensor,
    )

    interpretador.invoke()

    probabilidades = interpretador.get_tensor(
        saida["index"]
    )[0]

    return probabilidades.astype(np.float32)


def avaliar_tflite(
    caminho_modelo: Path,
    dataset,
    classes: list[str],
    limite: int,
) -> dict:
    interpretador = criar_interpretador(
        caminho_modelo
    )

    verdadeiros = []
    previstos = []
    confiancas = []
    tempos_ms = []

    for imagens, rotulos in dataset:
        for imagem, rotulo in zip(
            imagens.numpy(),
            rotulos.numpy(),
        ):
            if len(verdadeiros) >= limite:
                break

            inicio = time.perf_counter()

            probabilidades = executar_tflite(
                interpretador,
                imagem,
            )

            tempos_ms.append(
                (time.perf_counter() - inicio)
                * 1000
            )

            previsto = int(
                np.argmax(probabilidades)
            )

            verdadeiros.append(
                int(rotulo)
            )

            previstos.append(
                previsto
            )

            confiancas.append(
                float(np.max(probabilidades))
            )

        if len(verdadeiros) >= limite:
            break

    verdadeiros_array = np.asarray(
        verdadeiros
    )

    previstos_array = np.asarray(
        previstos
    )

    acuracia = float(
        np.mean(
            verdadeiros_array
            == previstos_array
        )
    )

    por_classe = {}

    for indice, classe in enumerate(classes):
        mascara = verdadeiros_array == indice
        quantidade = int(np.sum(mascara))

        acertos = int(
            np.sum(
                previstos_array[mascara]
                == indice
            )
        )

        por_classe[classe] = {
            "quantidade": quantidade,
            "acertos": acertos,
            "acuracia": (
                float(acertos / quantidade)
                if quantidade
                else None
            ),
        }

    return {
        "modelo": caminho_modelo.name,
        "tamanho_bytes": caminho_modelo.stat().st_size,
        "quantidade_imagens": len(verdadeiros),
        "acuracia": acuracia,
        "confianca_media": float(
            np.mean(confiancas)
        ),
        "latencia_media_ms": float(
            np.mean(tempos_ms)
        ),
        "latencia_mediana_ms": float(
            np.median(tempos_ms)
        ),
        "latencia_p95_ms": float(
            np.percentile(tempos_ms, 95)
        ),
        "por_classe": por_classe,
    }


def comparar_keras_tflite(
    modelo_keras: tf.keras.Model,
    caminho_tflite: Path,
    dataset,
    limite: int,
) -> dict:
    interpretador = criar_interpretador(
        caminho_tflite
    )

    diferenca_maxima = 0.0
    soma_diferencas = 0.0
    quantidade = 0
    concordancias = 0

    for imagens, _ in dataset:
        for imagem in imagens.numpy():
            if quantidade >= limite:
                break

            lote = np.expand_dims(
                imagem,
                axis=0,
            ).astype(np.float32)

            predicao_keras = modelo_keras(
                lote,
                training=False,
            ).numpy()[0].astype(np.float32)

            predicao_tflite = executar_tflite(
                interpretador,
                imagem,
            )

            diferenca = np.abs(
                predicao_keras
                - predicao_tflite
            )

            diferenca_maxima = max(
                diferenca_maxima,
                float(np.max(diferenca)),
            )

            soma_diferencas += float(
                np.mean(diferenca)
            )

            concordancias += int(
                np.argmax(predicao_keras)
                == np.argmax(predicao_tflite)
            )

            quantidade += 1

        if quantidade >= limite:
            break

    return {
        "quantidade_comparada": quantidade,
        "concordancia_top1": float(
            concordancias / quantidade
        ),
        "diferenca_absoluta_media": float(
            soma_diferencas / quantidade
        ),
        "diferenca_absoluta_maxima": (
            diferenca_maxima
        ),
    }


# ============================================================
# RELATORIOS
# ============================================================

def salvar_resumo_csv(
    resultados: list[dict],
) -> Path:
    destino = (
        PASTA_SAIDA
        / "resumo_modelos_tflite.csv"
    )

    campos = [
        "modelo",
        "tamanho_bytes",
        "tamanho_mb",
        "quantidade_imagens",
        "acuracia",
        "confianca_media",
        "latencia_media_ms",
        "latencia_mediana_ms",
        "latencia_p95_ms",
        "concordancia_top1_keras",
        "diferenca_media_keras",
        "diferenca_maxima_keras",
    ]

    with destino.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos,
            delimiter=";",
        )

        escritor.writeheader()

        for resultado in resultados:
            comparacao = resultado[
                "comparacao_keras"
            ]

            escritor.writerow(
                {
                    "modelo": resultado["modelo"],
                    "tamanho_bytes": resultado[
                        "tamanho_bytes"
                    ],
                    "tamanho_mb": (
                        f"{resultado['tamanho_bytes'] / (1024 ** 2):.4f}"
                    ),
                    "quantidade_imagens": resultado[
                        "quantidade_imagens"
                    ],
                    "acuracia": (
                        f"{resultado['acuracia']:.8f}"
                    ),
                    "confianca_media": (
                        f"{resultado['confianca_media']:.8f}"
                    ),
                    "latencia_media_ms": (
                        f"{resultado['latencia_media_ms']:.4f}"
                    ),
                    "latencia_mediana_ms": (
                        f"{resultado['latencia_mediana_ms']:.4f}"
                    ),
                    "latencia_p95_ms": (
                        f"{resultado['latencia_p95_ms']:.4f}"
                    ),
                    "concordancia_top1_keras": (
                        f"{comparacao['concordancia_top1']:.8f}"
                    ),
                    "diferenca_media_keras": (
                        f"{comparacao['diferenca_absoluta_media']:.10f}"
                    ),
                    "diferenca_maxima_keras": (
                        f"{comparacao['diferenca_absoluta_maxima']:.10f}"
                    ),
                }
            )

    return destino


# ============================================================
# EXECUCAO
# ============================================================

def main() -> int:
    print("=" * 78)
    print(
        "CASARAO IA - CONVERSAO E VALIDACAO TFLITE"
    )
    print("=" * 78)

    validar_entradas()

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    classes = carregar_classes()

    labels = salvar_labels(classes)

    print(
        "Carregando modelo Keras:",
        MODELO_KERAS,
    )

    modelo_treinado = tf.keras.models.load_model(
        MODELO_KERAS,
        compile=False,
    )

    modelo_inferencia = criar_modelo_inferencia(
        modelo_treinado
    )

    print(
        "Modelo exclusivo de inferencia criado, "
        "sem aumento de dados."
    )

    modelo_inferencia.summary()

    pasta_saved_model = exportar_saved_model(
        modelo_inferencia
    )

    print(
        "SavedModel exportado:",
        pasta_saved_model,
    )

    modelos_gerados = []

    if GERAR_FLOAT32:
        print("Convertendo versao float32...")

        modelos_gerados.append(
            converter_float32(
                pasta_saved_model
            )
        )

    if GERAR_FLOAT16:
        print("Convertendo versao float16...")

        modelos_gerados.append(
            converter_float16(
                pasta_saved_model
            )
        )

    dataset, classes_dataset = (
        criar_dataset_validacao()
    )

    if classes_dataset != classes:
        raise ValueError(
            "Ordem de classes divergente. "
            f"classes.json={classes}; "
            f"dataset={classes_dataset}"
        )

    resultados = []

    for caminho in modelos_gerados:
        print(
            f"\nValidando {caminho.name}..."
        )

        resultado = avaliar_tflite(
            caminho,
            dataset,
            classes,
            LIMITE_IMAGENS_VALIDACAO,
        )

        resultado[
            "detalhes_interpretador"
        ] = detalhes_interpretador(
            caminho
        )

        resultado[
            "comparacao_keras"
        ] = comparar_keras_tflite(
            modelo_inferencia,
            caminho,
            dataset,
            LIMITE_IMAGENS_VALIDACAO,
        )

        resultados.append(resultado)

        print(
            "Tamanho: "
            f"{resultado['tamanho_bytes'] / (1024 ** 2):.2f} MB"
        )

        print(
            "Acuracia: "
            f"{resultado['acuracia']:.4f}"
        )

        print(
            "Latencia media no WSL: "
            f"{resultado['latencia_media_ms']:.2f} ms/imagem"
        )

        print(
            "Concordancia Top-1 com Keras: "
            f"{resultado['comparacao_keras']['concordancia_top1']:.4f}"
        )

    validacao_json = (
        PASTA_SAIDA
        / "validacao_tflite.json"
    )

    with validacao_json.open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            {
                "modelo_keras": str(MODELO_KERAS),
                "saved_model": str(PASTA_SAVED_MODEL),
                "classes": classes,
                "tamanho_imagem": list(
                    TAMANHO_IMAGEM
                ),
                "entrada_esperada": (
                    "float32 com valores de pixel entre 0 e 255; "
                    "o pre-processamento esta incorporado na "
                    "MobileNetV3Small"
                ),
                "resultado_softmax": True,
                "resultados": resultados,
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    resumo_csv = salvar_resumo_csv(
        resultados
    )

    print("\n" + "=" * 78)
    print("CONVERSAO TFLITE FINALIZADA")
    print("=" * 78)
    print("Pasta:", PASTA_SAIDA)
    print("Labels:", labels)
    print("Resumo:", resumo_csv)
    print("Validacao:", validacao_json)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nConversao interrompida pelo usuario."
        )
        sys.exit(130)
    except Exception as erro:
        print(
            f"ERRO: {type(erro).__name__}: {erro}"
        )
        raise
