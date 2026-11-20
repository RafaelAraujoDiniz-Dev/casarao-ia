from pathlib import Path
import tensorflow as tf

RAIZ = Path("/mnt/f/casarao_ia")

PASTA_TFLITE = (
    RAIZ
    / "08_modelos"
    / "mobilenetv3small_ajuste_fino_v1"
    / "tflite"
)

SAVED_MODEL = (
    PASTA_TFLITE
    / "saved_model_inferencia"
)

MODELO_FLOAT32 = (
    PASTA_TFLITE
    / "casarao_ia_mobilenetv3_float32.tflite"
)

MODELO_FLOAT16 = (
    PASTA_TFLITE
    / "casarao_ia_mobilenetv3_float16.tflite"
)


def salvar(caminho, dados):
    caminho.write_bytes(dados)

    if not caminho.exists() or caminho.stat().st_size == 0:
        raise IOError(f"Falha ao salvar: {caminho}")

    print()
    print("Modelo criado:", caminho)
    print(
        "Tamanho:",
        f"{caminho.stat().st_size / (1024 ** 2):.2f} MB",
    )


def converter_float32():
    print("Convertendo float32 com conversor legado...")

    conversor = (
        tf.lite.TFLiteConverter.from_saved_model(
            str(SAVED_MODEL)
        )
    )

    conversor.experimental_new_converter = False

    conversor.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    modelo = conversor.convert()

    salvar(
        MODELO_FLOAT32,
        modelo,
    )


def converter_float16():
    print("Convertendo float16 com conversor legado...")

    conversor = (
        tf.lite.TFLiteConverter.from_saved_model(
            str(SAVED_MODEL)
        )
    )

    conversor.experimental_new_converter = False

    conversor.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    conversor.target_spec.supported_types = [
        tf.float16
    ]

    conversor.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    modelo = conversor.convert()

    salvar(
        MODELO_FLOAT16,
        modelo,
    )


def validar(caminho):
    interpretador = tf.lite.Interpreter(
        model_path=str(caminho),
        num_threads=4,
    )

    interpretador.allocate_tensors()

    entrada = interpretador.get_input_details()[0]
    saida = interpretador.get_output_details()[0]

    print()
    print("Modelo validado:", caminho.name)
    print("Entrada:", entrada["shape"], entrada["dtype"])
    print("Saida:", saida["shape"], saida["dtype"])


def main():
    if not SAVED_MODEL.exists():
        raise FileNotFoundError(
            f"SavedModel não encontrado: {SAVED_MODEL}"
        )

    PASTA_TFLITE.mkdir(
        parents=True,
        exist_ok=True,
    )

    converter_float32()
    converter_float16()

    validar(MODELO_FLOAT32)
    validar(MODELO_FLOAT16)

    print()
    print("=" * 72)
    print("CONVERSAO LEGADA FINALIZADA")
    print("=" * 72)


if __name__ == "__main__":
    main()
