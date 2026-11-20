from pathlib import Path
from datetime import datetime
import csv
import hashlib
import shutil

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(r"F:\casarao_ia")

PASTA_ENTRADA = RAIZ_PROJETO / "03_imagens_selecionadas"
PASTA_SAIDA = RAIZ_PROJETO / "03_imagens_convertidas"
PASTA_RELATORIOS = RAIZ_PROJETO / "06_metadados"

QUALIDADE_JPEG = 95

# False preserva arquivos JPG já existentes na pasta de saída.
SOBRESCREVER = False

EXTENSOES_HEIC = {".heic", ".heif"}

# Imagens já compatíveis serão copiadas para a pasta convertida.
EXTENSOES_COMPATIVEIS = {".jpg", ".jpeg", ".png", ".webp"}

register_heif_opener()


# ============================================================
# FUNÇÕES
# ============================================================

def gerar_identificador(caminho: Path) -> str:
    """Gera identificador curto para prevenir colisões de nomes."""

    relativo = caminho.relative_to(PASTA_ENTRADA)
    hash_curto = hashlib.sha256(
        relativo.as_posix().lower().encode("utf-8")
    ).hexdigest()[:10]

    return hash_curto


def criar_destino(arquivo_origem: Path, extensao: str) -> Path:
    """
    Reproduz a estrutura relativa da entrada na pasta de saída.

    Exemplo:
    entrada/palacio_dos_leoes/sessao_01/IMG_0001.HEIC
    saída/palacio_dos_leoes/sessao_01/IMG_0001.jpg
    """

    caminho_relativo = arquivo_origem.relative_to(PASTA_ENTRADA)
    pasta_relativa = caminho_relativo.parent

    pasta_destino = PASTA_SAIDA / pasta_relativa
    pasta_destino.mkdir(parents=True, exist_ok=True)

    nome_destino = f"{arquivo_origem.stem}{extensao}"
    destino = pasta_destino / nome_destino

    # Evita colisões entre IMG_0001.HEIC e IMG_0001.HEIF.
    if destino.exists() and not SOBRESCREVER:
        identificador = gerar_identificador(arquivo_origem)
        nome_destino = (
            f"{arquivo_origem.stem}_{identificador}{extensao}"
        )
        destino = pasta_destino / nome_destino

    return destino


def converter_heic(arquivo_origem: Path, arquivo_destino: Path) -> dict:
    """Converte uma imagem HEIC ou HEIF para JPG."""

    with Image.open(arquivo_origem) as imagem:
        # Corrige a orientação registrada nos metadados EXIF.
        imagem = ImageOps.exif_transpose(imagem)

        largura, altura = imagem.size

        # JPG não suporta transparência.
        if imagem.mode in ("RGBA", "LA"):
            fundo = Image.new("RGB", imagem.size, "white")
            canal_alpha = imagem.getchannel("A")
            fundo.paste(imagem, mask=canal_alpha)
            imagem = fundo

        elif imagem.mode != "RGB":
            imagem = imagem.convert("RGB")

        imagem.save(
            arquivo_destino,
            format="JPEG",
            quality=QUALIDADE_JPEG,
            optimize=True,
            subsampling=0,
        )

    if not arquivo_destino.exists():
        raise IOError("O arquivo JPG não foi encontrado após a conversão.")

    if arquivo_destino.stat().st_size == 0:
        raise IOError("O arquivo JPG foi criado com tamanho zero.")

    return {
        "largura": largura,
        "altura": altura,
    }


def copiar_imagem_compativel(
    arquivo_origem: Path,
    arquivo_destino: Path,
) -> dict:
    """Copia imagens que já estão em formato compatível."""

    with Image.open(arquivo_origem) as imagem:
        imagem.verify()

    with Image.open(arquivo_origem) as imagem:
        largura, altura = imagem.size

    shutil.copy2(arquivo_origem, arquivo_destino)

    return {
        "largura": largura,
        "altura": altura,
    }


def obter_classe_e_sessao(arquivo: Path) -> tuple[str, str]:
    """
    Identifica classe e sessão na estrutura:
    classe/sessao/arquivo.
    """

    caminho_relativo = arquivo.relative_to(PASTA_ENTRADA)
    partes = caminho_relativo.parts

    classe = partes[0] if len(partes) >= 1 else ""
    sessao = partes[1] if len(partes) >= 2 else ""

    return classe, sessao


def salvar_relatorio(resultados: list[dict]) -> Path:
    """Salva o relatório da conversão em CSV."""

    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    caminho_relatorio = (
        PASTA_RELATORIOS
        / f"relatorio_conversao_heic_{timestamp}.csv"
    )

    campos = [
        "arquivo_origem",
        "arquivo_destino",
        "classe",
        "sessao",
        "formato_origem",
        "acao",
        "status",
        "largura",
        "altura",
        "tamanho_origem_bytes",
        "tamanho_destino_bytes",
        "observacao",
    ]

    with caminho_relatorio.open(
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
        escritor.writerows(resultados)

    return caminho_relatorio


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("=" * 72)
    print("CASARÃO IA - CONVERSÃO HEIC/HEIF PARA JPG")
    print("=" * 72)

    if not PASTA_ENTRADA.exists():
        print(f"Pasta de entrada não encontrada: {PASTA_ENTRADA}")
        return

    arquivos = sorted(
        arquivo
        for arquivo in PASTA_ENTRADA.rglob("*")
        if arquivo.is_file()
        and arquivo.suffix.lower()
        in (EXTENSOES_HEIC | EXTENSOES_COMPATIVEIS)
    )

    if not arquivos:
        print("Nenhuma imagem compatível foi encontrada.")
        return

    print(f"Imagens encontradas: {len(arquivos)}")
    print()

    resultados = []

    convertidos = 0
    copiados = 0
    ignorados = 0
    erros = 0

    for numero, arquivo_origem in enumerate(arquivos, start=1):
        classe, sessao = obter_classe_e_sessao(arquivo_origem)
        extensao = arquivo_origem.suffix.lower()

        resultado = {
            "arquivo_origem": str(arquivo_origem),
            "arquivo_destino": "",
            "classe": classe,
            "sessao": sessao,
            "formato_origem": extensao,
            "acao": "",
            "status": "",
            "largura": 0,
            "altura": 0,
            "tamanho_origem_bytes": arquivo_origem.stat().st_size,
            "tamanho_destino_bytes": 0,
            "observacao": "",
        }

        print(f"[{numero}/{len(arquivos)}] {arquivo_origem.name}")

        try:
            if extensao in EXTENSOES_HEIC:
                arquivo_destino = criar_destino(
                    arquivo_origem,
                    ".jpg",
                )
                acao = "convertido_heic_para_jpg"

            else:
                arquivo_destino = criar_destino(
                    arquivo_origem,
                    extensao,
                )
                acao = "copiado_sem_conversao"

            resultado["arquivo_destino"] = str(arquivo_destino)
            resultado["acao"] = acao

            if arquivo_destino.exists() and not SOBRESCREVER:
                resultado["status"] = "arquivo_existente"
                resultado["observacao"] = (
                    "O arquivo de destino já existia."
                )
                resultado["tamanho_destino_bytes"] = (
                    arquivo_destino.stat().st_size
                )

                ignorados += 1
                resultados.append(resultado)

                print("    Arquivo já existente.")
                continue

            if extensao in EXTENSOES_HEIC:
                dimensoes = converter_heic(
                    arquivo_origem,
                    arquivo_destino,
                )
                convertidos += 1

            else:
                dimensoes = copiar_imagem_compativel(
                    arquivo_origem,
                    arquivo_destino,
                )
                copiados += 1

            resultado["largura"] = dimensoes["largura"]
            resultado["altura"] = dimensoes["altura"]
            resultado["tamanho_destino_bytes"] = (
                arquivo_destino.stat().st_size
            )
            resultado["status"] = "concluido"

            print(
                f"    Concluído: "
                f"{dimensoes['largura']}x{dimensoes['altura']}"
            )

        except Exception as erro:
            resultado["status"] = "erro"
            resultado["observacao"] = (
                f"{type(erro).__name__}: {erro}"
            )
            erros += 1

            print(f"    ERRO: {erro}")

        resultados.append(resultado)

    relatorio = salvar_relatorio(resultados)

    print()
    print("=" * 72)
    print("CONVERSÃO FINALIZADA")
    print("=" * 72)
    print(f"HEIC/HEIF convertidos: {convertidos}")
    print(f"Imagens copiadas:      {copiados}")
    print(f"Arquivos existentes:   {ignorados}")
    print(f"Erros:                  {erros}")
    print(f"Relatório:              {relatorio}")


if __name__ == "__main__":
    main()