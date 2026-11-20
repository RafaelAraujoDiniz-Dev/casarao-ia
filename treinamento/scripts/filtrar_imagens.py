from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import re
import shutil
import sys


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ_PROJETO = Path(r"F:\casarao_ia")

PASTA_ENTRADA = RAIZ_PROJETO / "03_imagens_convertidas"
PASTA_SAIDA = RAIZ_PROJETO / "03_imagens_limpas"
PASTA_RELATORIOS = RAIZ_PROJETO / "06_metadados"

RELATORIO_DECISOES = (
    PASTA_RELATORIOS / "relatorio_similaridade_20261109_225343.csv"
)

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}

# False: realiza apenas validacao e simulacao, sem copiar arquivos.
# True: copia as imagens aprovadas para 03_imagens_limpas.
EXECUTAR_COPIA = True

# False: interrompe se 03_imagens_limpas ja contiver arquivos.
# True: permite substituir arquivos existentes na pasta de saida.
SOBRESCREVER = False

# Quantidades esperadas com base na revisao atual.
TOTAL_ENTRADA_ESPERADO = 478
TOTAL_DESCARTES_UNICOS_ESPERADO = 10
TOTAL_SAIDA_ESPERADO = 468


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def normalizar_caminho_windows(texto: str) -> Path:
    """Converte o caminho do CSV em Path e remove espacos externos."""
    return Path(texto.strip())


def obter_imagens(pasta: Path) -> list[Path]:
    """Lista imagens validas de forma recursiva."""
    return sorted(
        arquivo
        for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    )


def extrair_decisao_descarte(decisao: str) -> tuple[str, str] | None:
    """
    Interpreta decisoes no formato:
      descartar_arquivo_a_nome.jpg
      descartar_arquivo_b_nome.jpg

    Retorna (lado, nome_informado) ou None para a decisao 'manter'.
    """
    decisao = decisao.strip()

    if decisao == "manter":
        return None

    correspondencia = re.fullmatch(
        r"descartar_arquivo_([ab])_(.+)",
        decisao,
        flags=re.IGNORECASE,
    )

    if not correspondencia:
        raise ValueError(f"Decisao invalida: {decisao!r}")

    return correspondencia.group(1).lower(), correspondencia.group(2)


def ler_relatorio_decisoes(caminho_csv: Path) -> tuple[set[Path], list[str]]:
    """Le o CSV, valida as decisoes e retorna caminhos unicos de descarte."""
    descartes: set[Path] = set()
    erros: list[str] = []

    with caminho_csv.open("r", encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo, delimiter=";")

        colunas_necessarias = {"arquivo_a", "arquivo_b", "decisao"}
        colunas_encontradas = set(leitor.fieldnames or [])
        faltantes = colunas_necessarias - colunas_encontradas

        if faltantes:
            raise ValueError(
                "Colunas ausentes no CSV: " + ", ".join(sorted(faltantes))
            )

        for numero_linha, linha in enumerate(leitor, start=2):
            decisao = (linha.get("decisao") or "").strip()

            if not decisao:
                erros.append(f"Linha {numero_linha}: decisao vazia.")
                continue

            try:
                resultado = extrair_decisao_descarte(decisao)
            except ValueError as erro:
                erros.append(f"Linha {numero_linha}: {erro}")
                continue

            if resultado is None:
                continue

            lado, nome_informado = resultado
            coluna = f"arquivo_{lado}"
            texto_caminho = (linha.get(coluna) or "").strip()

            if not texto_caminho:
                erros.append(
                    f"Linha {numero_linha}: caminho ausente em {coluna}."
                )
                continue

            caminho = normalizar_caminho_windows(texto_caminho)

            if caminho.name.casefold() != nome_informado.casefold():
                erros.append(
                    f"Linha {numero_linha}: o nome informado na decisao "
                    f"({nome_informado}) nao corresponde a {coluna} "
                    f"({caminho.name})."
                )
                continue

            try:
                caminho.relative_to(PASTA_ENTRADA)
            except ValueError:
                erros.append(
                    f"Linha {numero_linha}: arquivo fora da pasta de entrada: "
                    f"{caminho}"
                )
                continue

            if not caminho.exists():
                erros.append(
                    f"Linha {numero_linha}: arquivo de descarte nao existe: "
                    f"{caminho}"
                )
                continue

            descartes.add(caminho)

    return descartes, erros


def validar_pasta_saida() -> None:
    """Impede mistura acidental com uma execucao anterior."""
    if not PASTA_SAIDA.exists():
        return

    arquivos_existentes = [p for p in PASTA_SAIDA.rglob("*") if p.is_file()]

    if arquivos_existentes and not SOBRESCREVER:
        raise RuntimeError(
            f"A pasta de saida ja contem {len(arquivos_existentes)} arquivo(s): "
            f"{PASTA_SAIDA}\n"
            "Esvazie a pasta ou defina SOBRESCREVER = True conscientemente."
        )


def copiar_imagens(
    imagens_entrada: list[Path],
    descartes: set[Path],
) -> tuple[list[dict], Counter]:
    """Copia imagens aprovadas, preservando classe e sessao."""
    registros: list[dict] = []
    contadores: Counter = Counter()

    for origem in imagens_entrada:
        relativo = origem.relative_to(PASTA_ENTRADA)
        partes = relativo.parts
        classe = partes[0] if len(partes) >= 1 else ""
        sessao = partes[1] if len(partes) >= 2 else ""

        if origem in descartes:
            registros.append(
                {
                    "arquivo_origem": str(origem),
                    "arquivo_destino": "",
                    "classe": classe,
                    "sessao": sessao,
                    "acao": "descartado_por_similaridade",
                    "status": "nao_copiado",
                    "observacao": "Arquivo listado no relatorio revisado.",
                }
            )
            contadores["descartados"] += 1
            continue

        destino = PASTA_SAIDA / relativo

        if EXECUTAR_COPIA:
            destino.parent.mkdir(parents=True, exist_ok=True)

            if destino.exists() and not SOBRESCREVER:
                status = "ja_existia"
                contadores["ja_existiam"] += 1
            else:
                shutil.copy2(origem, destino)
                status = "copiado"
                contadores["copiados"] += 1
        else:
            status = "simulado"
            contadores["simulados"] += 1

        registros.append(
            {
                "arquivo_origem": str(origem),
                "arquivo_destino": str(destino),
                "classe": classe,
                "sessao": sessao,
                "acao": "manter",
                "status": status,
                "observacao": "",
            }
        )

    return registros, contadores


def salvar_relatorio(registros: list[dict], modo: str) -> Path:
    """Registra todas as imagens mantidas e descartadas."""
    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = PASTA_RELATORIOS / f"relatorio_filtragem_{modo}_{timestamp}.csv"

    campos = [
        "arquivo_origem",
        "arquivo_destino",
        "classe",
        "sessao",
        "acao",
        "status",
        "observacao",
    ]

    with destino.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        escritor.writerows(registros)

    return destino


def contar_por_classe(registros: list[dict]) -> Counter:
    """Conta imagens aprovadas por classe."""
    return Counter(
        registro["classe"]
        for registro in registros
        if registro["acao"] == "manter"
    )


# ============================================================
# EXECUCAO
# ============================================================

def main() -> int:
    print("=" * 76)
    print("CASARAO IA - FILTRAGEM DE IMAGENS POR RELATORIO DE SIMILARIDADE")
    print("=" * 76)
    print(f"Entrada:    {PASTA_ENTRADA}")
    print(f"Saida:      {PASTA_SAIDA}")
    print(f"Relatorio:  {RELATORIO_DECISOES}")
    print(f"Modo:       {'COPIA' if EXECUTAR_COPIA else 'SIMULACAO'}")
    print()

    if not PASTA_ENTRADA.exists():
        print(f"ERRO: pasta de entrada inexistente: {PASTA_ENTRADA}")
        return 1

    if not RELATORIO_DECISOES.exists():
        print(f"ERRO: relatorio inexistente: {RELATORIO_DECISOES}")
        return 1

    imagens_entrada = obter_imagens(PASTA_ENTRADA)
    descartes, erros = ler_relatorio_decisoes(RELATORIO_DECISOES)

    if erros:
        print("ERROS ENCONTRADOS NO RELATORIO:")
        for erro in erros:
            print(f"- {erro}")
        print("\nNenhum arquivo foi copiado.")
        return 1

    print(f"Imagens encontradas:          {len(imagens_entrada)}")
    print(f"Descartes unicos validados:   {len(descartes)}")
    print(f"Imagens previstas na saida:   {len(imagens_entrada) - len(descartes)}")
    print()

    avisos = []
    if len(imagens_entrada) != TOTAL_ENTRADA_ESPERADO:
        avisos.append(
            f"Total de entrada diferente do esperado: "
            f"{len(imagens_entrada)} != {TOTAL_ENTRADA_ESPERADO}."
        )

    if len(descartes) != TOTAL_DESCARTES_UNICOS_ESPERADO:
        avisos.append(
            f"Descartes unicos diferentes do esperado: "
            f"{len(descartes)} != {TOTAL_DESCARTES_UNICOS_ESPERADO}."
        )

    previsto = len(imagens_entrada) - len(descartes)
    if previsto != TOTAL_SAIDA_ESPERADO:
        avisos.append(
            f"Total previsto de saida diferente do esperado: "
            f"{previsto} != {TOTAL_SAIDA_ESPERADO}."
        )

    if avisos:
        print("AVISOS:")
        for aviso in avisos:
            print(f"- {aviso}")
        print()

    if EXECUTAR_COPIA:
        validar_pasta_saida()

    registros, contadores = copiar_imagens(imagens_entrada, descartes)
    modo = "copia" if EXECUTAR_COPIA else "simulacao"
    relatorio_saida = salvar_relatorio(registros, modo)
    por_classe = contar_por_classe(registros)

    print("Distribuicao aprovada por classe:")
    for classe, quantidade in sorted(por_classe.items()):
        print(f"- {classe}: {quantidade}")

    print()
    print("=" * 76)
    print("FILTRAGEM FINALIZADA")
    print("=" * 76)
    print(f"Descartados:       {contadores['descartados']}")
    print(f"Copiados:          {contadores['copiados']}")
    print(f"Simulados:         {contadores['simulados']}")
    print(f"Ja existentes:     {contadores['ja_existiam']}")
    print(f"Relatorio gerado:  {relatorio_saida}")

    if not EXECUTAR_COPIA:
        print()
        print("SIMULACAO CONCLUIDA: nenhum arquivo foi copiado ou alterado.")
        print("Revise os totais e depois altere EXECUTAR_COPIA = True.")
    else:
        total_saida = len(obter_imagens(PASTA_SAIDA))
        print(f"Arquivos na saida: {total_saida}")

        if total_saida != previsto:
            print(
                "ERRO: o total da pasta de saida nao corresponde ao previsto."
            )
            return 1

        print("Copia validada com sucesso.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
