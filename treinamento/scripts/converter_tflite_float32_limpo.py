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

RAIZ = Path("/mnt/f/casarao_ia")
MODELO_E2 = RAIZ / "08_modelos" / "mobilenetv3small_ajuste_fino_v1" / "melhor_modelo.keras"
PASTA_MODELO = RAIZ / "08_modelos" / "mobilenetv3small_ajuste_fino_v1"
PASTA_SAIDA = PASTA_MODELO / "tflite"
PASTA_VALIDACAO = RAIZ / "04_dataset_final" / "validacao_diurna"
ARQUIVO_CLASSES = PASTA_MODELO / "classes.json"

IMG_SIZE = (224, 224)
SEMENTE = 20261118
NUM_THREADS = 4

# ============================================================
# MODELO FLOAT32 LIMPO
# ============================================================

def criar_modelo_float32_limpo(modelo_treinado: tf.keras.Model) -> tf.keras.Model:
    """Reconstrói a MobileNetV3Small inteiramente em float32 e transfere pesos."""
    tf.keras.mixed_precision.set_global_policy("float32")

    base_origem = modelo_treinado.get_layer("MobileNetV3Small")
    classificador_origem = modelo_treinado.get_layer("classificador")

    base_nova = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
        include_preprocessing=True,
        pooling=None,
        name="MobileNetV3Small",
    )
    base_nova.set_weights(base_origem.get_weights())
    base_nova.trainable = False

    entrada = tf.keras.Input(
        shape=(224, 224, 3),
        batch_size=1,
        dtype=tf.float32,
        name="imagem_entrada",
    )
    x = base_nova(entrada, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)
    saida = tf.keras.layers.Dense(
        5,
        activation="softmax",
        dtype="float32",
        name="classificador",
    )(x)

    modelo = tf.keras.Model(entrada, saida, name="casarao_ia_inferencia_float32")
    modelo.get_layer("classificador").set_weights(classificador_origem.get_weights())
    return modelo


def validar_equivalencia(modelo_original, modelo_limpo) -> None:
    rng = np.random.default_rng(SEMENTE)
    lote = rng.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
    pred_original = modelo_original(lote, training=False).numpy().astype(np.float32)
    pred_limpo = modelo_limpo(lote, training=False).numpy().astype(np.float32)
    diff = np.abs(pred_original - pred_limpo)
    print("Diferença média Keras original x limpo:", float(diff.mean()))
    print("Diferença máxima Keras original x limpo:", float(diff.max()))
    print("Top-1 original:", int(np.argmax(pred_original)))
    print("Top-1 limpo:", int(np.argmax(pred_limpo)))
    if np.argmax(pred_original) != np.argmax(pred_limpo):
        raise RuntimeError("Modelo float32 limpo alterou a classe Top-1 no teste de equivalência.")

# ============================================================
# CONVERSAO
# ============================================================

def converter(modelo: tf.keras.Model, float16: bool) -> tuple[Path, bytes]:
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    nome = "casarao_ia_mobilenetv3_float32.tflite"
    if float16:
        conversor.optimizations = [tf.lite.Optimize.DEFAULT]
        conversor.target_spec.supported_types = [tf.float16]
        nome = "casarao_ia_mobilenetv3_float16.tflite"

    dados = conversor.convert()
    caminho = PASTA_SAIDA / nome
    caminho.write_bytes(dados)
    if caminho.stat().st_size == 0:
        raise IOError(f"Arquivo vazio: {caminho}")
    return caminho, dados

# ============================================================
# VALIDACAO
# ============================================================

def criar_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(
        PASTA_VALIDACAO,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=False,
        seed=SEMENTE,
    )
    return ds, ds.class_names


def avaliar_tflite(caminho: Path, dataset) -> dict:
    interp = tf.lite.Interpreter(model_path=str(caminho), num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    verdadeiros, previstos, confs, tempos = [], [], [], []
    for imagens, rotulos in dataset:
        imagem = imagens.numpy().astype(inp["dtype"])
        inicio = time.perf_counter()
        interp.set_tensor(inp["index"], imagem)
        interp.invoke()
        probs = interp.get_tensor(out["index"])[0].astype(np.float32)
        tempos.append((time.perf_counter() - inicio) * 1000)
        verdadeiros.append(int(rotulos.numpy()[0]))
        previstos.append(int(np.argmax(probs)))
        confs.append(float(np.max(probs)))

    v = np.asarray(verdadeiros)
    p = np.asarray(previstos)
    return {
        "modelo": caminho.name,
        "tamanho_bytes": caminho.stat().st_size,
        "tamanho_mb": caminho.stat().st_size / (1024 ** 2),
        "acuracia": float(np.mean(v == p)),
        "confianca_media": float(np.mean(confs)),
        "latencia_media_ms": float(np.mean(tempos)),
        "latencia_mediana_ms": float(np.median(tempos)),
        "latencia_p95_ms": float(np.percentile(tempos, 95)),
        "entrada_shape": inp["shape"].tolist(),
        "entrada_dtype": np.dtype(inp["dtype"]).name,
        "saida_shape": out["shape"].tolist(),
        "saida_dtype": np.dtype(out["dtype"]).name,
    }


def comparar_keras_tflite(modelo, caminho, dataset) -> dict:
    interp = tf.lite.Interpreter(model_path=str(caminho), num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    concordancias = 0
    diffs_medias = []
    diff_max = 0.0
    total = 0

    for imagens, _ in dataset:
        lote = imagens.numpy().astype(np.float32)
        keras_probs = modelo(lote, training=False).numpy()[0].astype(np.float32)
        interp.set_tensor(inp["index"], lote.astype(inp["dtype"]))
        interp.invoke()
        lite_probs = interp.get_tensor(out["index"])[0].astype(np.float32)
        diff = np.abs(keras_probs - lite_probs)
        diffs_medias.append(float(diff.mean()))
        diff_max = max(diff_max, float(diff.max()))
        concordancias += int(np.argmax(keras_probs) == np.argmax(lite_probs))
        total += 1

    return {
        "concordancia_top1": concordancias / total,
        "diferenca_media": float(np.mean(diffs_medias)),
        "diferenca_maxima": diff_max,
    }

# ============================================================
# EXECUCAO
# ============================================================

def main() -> int:
    print("=" * 78)
    print("CASARAO IA - CONVERSAO TFLITE A PARTIR DE MODELO FLOAT32 LIMPO")
    print("=" * 78)
    print("TensorFlow:", tf.__version__)
    print("Python:", sys.executable)

    if not MODELO_E2.exists():
        raise FileNotFoundError(MODELO_E2)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    tf.keras.mixed_precision.set_global_policy("float32")
    modelo_original = tf.keras.models.load_model(MODELO_E2, compile=False)
    modelo_limpo = criar_modelo_float32_limpo(modelo_original)
    modelo_limpo.summary()
    validar_equivalencia(modelo_original, modelo_limpo)

    # Salva uma versão Keras limpa para rastreabilidade.
    caminho_keras_limpo = PASTA_SAIDA / "casarao_ia_inferencia_float32.keras"
    modelo_limpo.save(caminho_keras_limpo)

    caminho_f32, _ = converter(modelo_limpo, float16=False)
    print("Float32 criado:", caminho_f32, f"({caminho_f32.stat().st_size/(1024**2):.2f} MB)")

    caminho_f16, _ = converter(modelo_limpo, float16=True)
    print("Float16 criado:", caminho_f16, f"({caminho_f16.stat().st_size/(1024**2):.2f} MB)")

    dataset, classes_dataset = criar_dataset()
    classes = json.loads(ARQUIVO_CLASSES.read_text(encoding="utf-8"))
    if classes_dataset != classes:
        raise ValueError(f"Classes divergentes: {classes_dataset} != {classes}")

    (PASTA_SAIDA / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    resultados = []
    for caminho in [caminho_f32, caminho_f16]:
        print("Validando:", caminho.name)
        resultado = avaliar_tflite(caminho, dataset)
        resultado["comparacao_keras"] = comparar_keras_tflite(modelo_limpo, caminho, dataset)
        resultados.append(resultado)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))

    (PASTA_SAIDA / "validacao_tflite.json").write_text(
        json.dumps(
            {
                "tensorflow_conversao": tf.__version__,
                "python": sys.executable,
                "modelo_origem": str(MODELO_E2),
                "modelo_keras_limpo": str(caminho_keras_limpo),
                "classes": classes,
                "entrada": "float32 [1,224,224,3], pixels 0..255",
                "resultados": resultados,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    campos = [
        "modelo", "tamanho_mb", "acuracia", "confianca_media",
        "latencia_media_ms", "latencia_mediana_ms", "latencia_p95_ms",
        "concordancia_top1_keras", "diferenca_media_keras", "diferenca_maxima_keras",
    ]
    with (PASTA_SAIDA / "resumo_modelos_tflite.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        for r in resultados:
            c = r["comparacao_keras"]
            escritor.writerow({
                "modelo": r["modelo"],
                "tamanho_mb": f"{r['tamanho_mb']:.4f}",
                "acuracia": f"{r['acuracia']:.8f}",
                "confianca_media": f"{r['confianca_media']:.8f}",
                "latencia_media_ms": f"{r['latencia_media_ms']:.4f}",
                "latencia_mediana_ms": f"{r['latencia_mediana_ms']:.4f}",
                "latencia_p95_ms": f"{r['latencia_p95_ms']:.4f}",
                "concordancia_top1_keras": f"{c['concordancia_top1']:.8f}",
                "diferenca_media_keras": f"{c['diferenca_media']:.10f}",
                "diferenca_maxima_keras": f"{c['diferenca_maxima']:.10f}",
            })

    print("=" * 78)
    print("CONVERSAO E VALIDACAO FINALIZADAS")
    print("Saida:", PASTA_SAIDA)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as erro:
        print(f"ERRO: {type(erro).__name__}: {erro}")
        raise
