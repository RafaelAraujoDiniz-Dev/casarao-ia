from pathlib import Path
from datetime import datetime
import csv

from PIL import Image, ImageOps
import imagehash


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(r"F:\casarao_ia")

PASTA_ENTRADA = RAIZ_PROJETO / "03_imagens_convertidas"
PASTA_RELATORIOS = RAIZ_PROJETO / "06_metadados"

EXTENSOES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# Quanto menor o limite, mais parecidas precisam ser as imagens.
#
# 0 = imagens praticamente idênticas
# 1 a 3 = muito semelhantes
# 4 a 6 = semelhantes, exige revisão visual
LIMITE_DISTANCIA = 5

# Compara somente imagens da mesma edificação.
COMPARAR_APENAS_MESMA_CLASSE = True


# ============================================================
# FUNÇÕES
# ============================================================

def obter_classe_e_sessao(arquivo):
    relativo = arquivo.relative_to(PASTA_ENTRADA)
    partes = relativo.parts

    classe = partes[0] if len(partes) >= 1 else ""
    sessao = partes[1] if len(partes) >= 2 else ""

    return classe, sessao


def calcular_hash_visual(arquivo):
    """
    Calcula o hash perceptual da imagem.

    Esse hash permite identificar imagens visualmente semelhantes,
    mesmo quando possuem nomes, tamanhos ou compressões diferentes.
    """

    with Image.open(arquivo) as imagem:
        imagem = ImageOps.exif_transpose(imagem)
        imagem = imagem.convert("RGB")

        return imagehash.phash(imagem)


def carregar_imagens():
    imagens = []

    for arquivo in sorted(PASTA_ENTRADA.rglob("*")):
        if not arquivo.is_file():
            continue

        if arquivo.suffix.lower() not in EXTENSOES:
            continue

        classe, sessao = obter_classe_e_sessao(arquivo)

        try:
            hash_visual = calcular_hash_visual(arquivo)

            imagens.append(
                {
                    "arquivo": arquivo,
                    "classe": classe,
                    "sessao": sessao,
                    "hash": hash_visual,
                }
            )

        except Exception as erro:
            print(f"[ERRO] {arquivo}")
            print(f"       {type(erro).__name__}: {erro}")

    return imagens


def encontrar_pares(imagens):
    pares = []

    total = len(imagens)

    for indice_a in range(total):
        imagem_a = imagens[indice_a]

        for indice_b in range(indice_a + 1, total):
            imagem_b = imagens[indice_b]

            if (
                COMPARAR_APENAS_MESMA_CLASSE
                and imagem_a["classe"] != imagem_b["classe"]
            ):
                continue

            distancia = imagem_a["hash"] - imagem_b["hash"]

            if distancia <= LIMITE_DISTANCIA:
                pares.append(
                    {
                        "classe": imagem_a["classe"],
                        "sessao_a": imagem_a["sessao"],
                        "arquivo_a": str(imagem_a["arquivo"]),
                        "sessao_b": imagem_b["sessao"],
                        "arquivo_b": str(imagem_b["arquivo"]),
                        "distancia_hash": distancia,
                        "classificacao": classificar_distancia(
                            distancia
                        ),
                        "decisao": "",
                        "observacao": "",
                    }
                )

    return pares


def classificar_distancia(distancia):
    if distancia == 0:
        return "praticamente_identicas"

    if distancia <= 3:
        return "muito_semelhantes"

    return "semelhantes_revisar"


def salvar_relatorio(pares):
    PASTA_RELATORIOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    relatorio = (
        PASTA_RELATORIOS
        / f"relatorio_similaridade_{timestamp}.csv"
    )

    campos = [
        "classe",
        "sessao_a",
        "arquivo_a",
        "sessao_b",
        "arquivo_b",
        "distancia_hash",
        "classificacao",
        "decisao",
        "observacao",
    ]

    with relatorio.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as arquivo_csv:

        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=campos,
            delimiter=";",
        )

        escritor.writeheader()
        escritor.writerows(pares)

    return relatorio


def main():
    print("=" * 72)
    print("CASARÃO IA - VERIFICAÇÃO DE SIMILARIDADE VISUAL")
    print("=" * 72)

    if not PASTA_ENTRADA.exists():
        print(f"Pasta não encontrada: {PASTA_ENTRADA}")
        return

    imagens = carregar_imagens()

    print(f"Imagens válidas: {len(imagens)}")
    print(f"Limite utilizado: {LIMITE_DISTANCIA}")
    print()

    pares = encontrar_pares(imagens)

    relatorio = salvar_relatorio(pares)

    identicas = sum(
        par["distancia_hash"] == 0
        for par in pares
    )

    muito_semelhantes = sum(
        1 <= par["distancia_hash"] <= 3
        for par in pares
    )

    semelhantes = sum(
        4 <= par["distancia_hash"] <= LIMITE_DISTANCIA
        for par in pares
    )

    print("=" * 72)
    print("ANÁLISE FINALIZADA")
    print("=" * 72)
    print(f"Pares encontrados:       {len(pares)}")
    print(f"Praticamente idênticas:  {identicas}")
    print(f"Muito semelhantes:       {muito_semelhantes}")
    print(f"Semelhantes para revisar:{semelhantes}")
    print(f"Relatório:                {relatorio}")
    print()
    print("Nenhum arquivo foi excluído ou alterado.")


if __name__ == "__main__":
    main()