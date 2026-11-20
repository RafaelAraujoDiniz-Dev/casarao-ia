from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import random
import shutil
import sys


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ_PROJETO = Path(r"F:\casarao_ia")

PASTA_ENTRADA = RAIZ_PROJETO / "03_imagens_convertidas"
PASTA_SAIDA = RAIZ_PROJETO / "04_dataset_final"
PASTA_RELATORIOS = RAIZ_PROJETO / "06_metadados"

CLASSES = [
    "museu_casa_de_nhozinho",
    "palacio_cristo_rei",
    "palacio_de_la_ravardiere",
    "palacio_dos_leoes",
    "teatro_arthur_azevedo",
]

# Cada conjunto utiliza uma sessao independente.
PLANO = {
    "treino": {
        "sessao": "sessao_01",
        "quantidade_por_classe": 60,
    },
    "validacao_diurna": {
        "sessao": "sessao_02",
        "quantidade_por_classe": 35,
    },
    "teste_noturno": {
        "sessao": "sessao_03",
        "quantidade_por_classe": 50,
    },
}

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}

# Semente fixa para tornar a selecao reproduzivel.
SEMENTE = 20261118

# False: apenas valida e simula.
# True: cria efetivamente o dataset balanceado.
EXECUTAR_COPIA = True


# False: interrompe caso 04_dataset_final ja tenha arquivos.
# True: apaga e recria 04_dataset_final antes da copia.
RECRIAR_SAIDA = False

# Calcula SHA-256 para verificar se o mesmo conteudo apareceu
# em mais de um subconjunto. Recomendado manter True.
VERIFICAR_HASH = True


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def sha256_arquivo(caminho: Path) -> str:
    """Calcula o SHA-256 de um arquivo."""
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def listar_imagens(pasta: Path) -> list[Path]:
    """Lista imagens validas diretamente dentro de uma pasta."""
    if not pasta.exists():
        return []

    return sorted(
        arquivo
        for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    )


def contar_imagens_recursivamente(pasta: Path) -> int:
    """Conta imagens validas em uma arvore de diretorios."""
    if not pasta.exists():
        return 0

    return sum(
        1
        for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    )


def validar_estrutura() -> list[str]:
    """Valida pastas, classes, sessoes e quantidades minimas."""
    erros: list[str] = []

    if not PASTA_ENTRADA.exists():
        return [f"Pasta de entrada inexistente: {PASTA_ENTRADA}"]

    for conjunto, configuracao in PLANO.items():
        sessao = configuracao["sessao"]
        quantidade = configuracao["quantidade_por_classe"]

        for classe in CLASSES:
            pasta = PASTA_ENTRADA / classe / sessao
            disponiveis = len(listar_imagens(pasta))

            if not pasta.exists():
                erros.append(
                    f"Pasta ausente para {conjunto}: {pasta}"
                )
            elif disponiveis < quantidade:
                erros.append(
                    f"Quantidade insuficiente em {classe}/{sessao}: "
                    f"disponiveis={disponiveis}, necessarias={quantidade}."
                )

    return erros


def validar_saida() -> None:
    """Protege o dataset final contra mistura com execucoes anteriores."""
    total_existente = contar_imagens_recursivamente(PASTA_SAIDA)

    if total_existente == 0:
        return

    if not RECRIAR_SAIDA:
        raise RuntimeError(
            f"A pasta de saida ja contem {total_existente} imagem(ns): "
            f"{PASTA_SAIDA}\n"
            "Defina RECRIAR_SAIDA = True somente se desejar apaga-la "
            "e reconstrui-la integralmente."
        )

    shutil.rmtree(PASTA_SAIDA)


def selecionar_imagens() -> list[dict]:
    """
    Seleciona aleatoriamente a quantidade definida para cada classe,
    usando uma semente fixa e mantendo as sessoes separadas.
    """
    registros: list[dict] = []

    for indice_conjunto, (conjunto, configuracao) in enumerate(PLANO.items()):
        sessao = configuracao["sessao"]
        quantidade = configuracao["quantidade_por_classe"]

        for indice_classe, classe in enumerate(CLASSES):
            pasta_origem = PASTA_ENTRADA / classe / sessao
            disponiveis = listar_imagens(pasta_origem)

            # Gerador independente por conjunto e classe.
            semente_local = SEMENTE + indice_conjunto * 1000 + indice_classe
            gerador = random.Random(semente_local)
            selecionadas = gerador.sample(disponiveis, quantidade)

            for ordem, origem in enumerate(sorted(selecionadas), start=1):
                destino = PASTA_SAIDA / conjunto / classe / origem.name

                registros.append(
                    {
                        "conjunto": conjunto,
                        "classe": classe,
                        "sessao": sessao,
                        "ordem": ordem,
                        "arquivo_origem": str(origem),
                        "arquivo_destino": str(destino),
                        "nome_arquivo": origem.name,
                        "semente_local": semente_local,
                        "sha256": "",
                        "status": "planejado",
                        "observacao": "",
                    }
                )

    return registros


def verificar_colisoes_de_destino(registros: list[dict]) -> list[str]:
    """Verifica se mais de uma origem produziria o mesmo destino."""
    destinos: defaultdict[str, list[str]] = defaultdict(list)

    for registro in registros:
        destinos[registro["arquivo_destino"].casefold()].append(
            registro["arquivo_origem"]
        )

    erros = []
    for destino, origens in destinos.items():
        if len(origens) > 1:
            erros.append(
                f"Colisao de destino: {destino} <- {origens}"
            )

    return erros


def verificar_sobreposicao_de_caminhos(registros: list[dict]) -> list[str]:
    """Garante que uma mesma origem nao esteja em dois conjuntos."""
    origens: defaultdict[str, set[str]] = defaultdict(set)

    for registro in registros:
        origens[registro["arquivo_origem"].casefold()].add(
            registro["conjunto"]
        )

    return [
        f"Arquivo em mais de um conjunto: {origem} -> {sorted(conjuntos)}"
        for origem, conjuntos in origens.items()
        if len(conjuntos) > 1
    ]


def calcular_e_validar_hashes(registros: list[dict]) -> list[str]:
    """
    Calcula SHA-256 e verifica se o mesmo conteudo aparece em conjuntos
    diferentes. Conteudo duplicado dentro do mesmo conjunto tambem e avisado.
    """
    por_hash: defaultdict[str, list[dict]] = defaultdict(list)

    for registro in registros:
        caminho = Path(registro["arquivo_origem"])
        hash_arquivo = sha256_arquivo(caminho)
        registro["sha256"] = hash_arquivo
        por_hash[hash_arquivo].append(registro)

    erros = []
    for hash_arquivo, itens in por_hash.items():
        if len(itens) <= 1:
            continue

        conjuntos = sorted({item["conjunto"] for item in itens})
        caminhos = [item["arquivo_origem"] for item in itens]

        if len(conjuntos) > 1:
            erros.append(
                "Conteudo identico em conjuntos diferentes: "
                f"hash={hash_arquivo}, conjuntos={conjuntos}, arquivos={caminhos}"
            )
        else:
            erros.append(
                "Conteudo identico repetido no mesmo conjunto: "
                f"hash={hash_arquivo}, conjunto={conjuntos[0]}, arquivos={caminhos}"
            )

    return erros


def copiar_dataset(registros: list[dict]) -> None:
    """Copia as imagens selecionadas para 04_dataset_final."""
    for indice, registro in enumerate(registros, start=1):
        origem = Path(registro["arquivo_origem"])
        destino = Path(registro["arquivo_destino"])

        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, destino)

        if not destino.exists() or destino.stat().st_size == 0:
            registro["status"] = "erro_copia"
            registro["observacao"] = "Arquivo de destino ausente ou vazio."
            raise IOError(f"Falha ao copiar: {origem} -> {destino}")

        registro["status"] = "copiado"

        if indice % 100 == 0 or indice == len(registros):
            print(f"Copiados: {indice}/{len(registros)}")


def salvar_manifesto(registros: list[dict], modo: str) -> Path:
    """Salva um inventario reproduzivel da selecao."""
    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = PASTA_RELATORIOS / f"manifesto_dataset_{modo}_{timestamp}.csv"

    campos = [
        "conjunto",
        "classe",
        "sessao",
        "ordem",
        "arquivo_origem",
        "arquivo_destino",
        "nome_arquivo",
        "semente_local",
        "sha256",
        "status",
        "observacao",
    ]

    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        escritor.writerows(registros)

    return caminho


def salvar_resumo(registros: list[dict], modo: str) -> Path:
    """Salva os totais por conjunto, classe e sessao."""
    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = PASTA_RELATORIOS / f"resumo_dataset_{modo}_{timestamp}.csv"

    contagem = Counter(
        (registro["conjunto"], registro["classe"], registro["sessao"])
        for registro in registros
    )

    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        campos = ["conjunto", "classe", "sessao", "quantidade"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()

        for (conjunto, classe, sessao), quantidade in sorted(contagem.items()):
            escritor.writerow(
                {
                    "conjunto": conjunto,
                    "classe": classe,
                    "sessao": sessao,
                    "quantidade": quantidade,
                }
            )

    return caminho


def validar_dataset_criado(registros: list[dict]) -> list[str]:
    """Confere os totais fisicos apos a copia."""
    erros = []

    for conjunto, configuracao in PLANO.items():
        esperado_por_classe = configuracao["quantidade_por_classe"]

        for classe in CLASSES:
            pasta = PASTA_SAIDA / conjunto / classe
            encontrado = len(listar_imagens(pasta))

            if encontrado != esperado_por_classe:
                erros.append(
                    f"Total incorreto em {conjunto}/{classe}: "
                    f"encontrado={encontrado}, esperado={esperado_por_classe}."
                )

    total_esperado = sum(
        configuracao["quantidade_por_classe"] * len(CLASSES)
        for configuracao in PLANO.values()
    )
    total_encontrado = contar_imagens_recursivamente(PASTA_SAIDA)

    if total_encontrado != total_esperado:
        erros.append(
            f"Total geral incorreto: encontrado={total_encontrado}, "
            f"esperado={total_esperado}."
        )

    return erros


def imprimir_resumo(registros: list[dict]) -> None:
    """Exibe a distribuicao planejada."""
    contagem = Counter(
        (registro["conjunto"], registro["classe"])
        for registro in registros
    )

    print("Distribuicao planejada:")
    for conjunto in PLANO:
        print(f"\n{conjunto}:")
        for classe in CLASSES:
            print(f"  {classe}: {contagem[(conjunto, classe)]}")

    print(f"\nTotal planejado: {len(registros)}")


# ============================================================
# EXECUCAO
# ============================================================

def main() -> int:
    print("=" * 78)
    print("CASARAO IA - CONSTRUCAO DO DATASET BALANCEADO")
    print("=" * 78)
    print(f"Entrada: {PASTA_ENTRADA}")
    print(f"Saida:   {PASTA_SAIDA}")
    print(f"Semente: {SEMENTE}")
    print(f"Modo:    {'COPIA' if EXECUTAR_COPIA else 'SIMULACAO'}")
    print()

    erros_estrutura = validar_estrutura()
    if erros_estrutura:
        print("ERROS NA ESTRUTURA OU NAS QUANTIDADES:")
        for erro in erros_estrutura:
            print(f"- {erro}")
        print("\nNenhum arquivo foi copiado.")
        return 1

    try:
        registros = selecionar_imagens()
    except Exception as erro:
        print(f"ERRO NA SELECAO: {type(erro).__name__}: {erro}")
        return 1

    erros = []
    erros.extend(verificar_colisoes_de_destino(registros))
    erros.extend(verificar_sobreposicao_de_caminhos(registros))

    if VERIFICAR_HASH:
        print("Calculando hashes SHA-256 e verificando sobreposicoes...")
        erros.extend(calcular_e_validar_hashes(registros))

    if erros:
        print("\nERROS DE INTEGRIDADE:")
        for erro in erros:
            print(f"- {erro}")
        manifesto = salvar_manifesto(registros, "erro")
        print(f"\nManifesto de diagnostico: {manifesto}")
        print("Nenhum arquivo foi copiado.")
        return 1

    imprimir_resumo(registros)

    if not EXECUTAR_COPIA:
        manifesto = salvar_manifesto(registros, "simulacao")
        resumo = salvar_resumo(registros, "simulacao")
        print("\nSIMULACAO CONCLUIDA: nenhum arquivo foi copiado.")
        print(f"Manifesto: {manifesto}")
        print(f"Resumo:    {resumo}")
        print("Altere EXECUTAR_COPIA = True para criar o dataset.")
        return 0

    try:
        validar_saida()
        copiar_dataset(registros)
    except Exception as erro:
        manifesto = salvar_manifesto(registros, "falha_copia")
        print(f"\nERRO NA COPIA: {type(erro).__name__}: {erro}")
        print(f"Manifesto parcial: {manifesto}")
        return 1

    erros_finais = validar_dataset_criado(registros)
    manifesto = salvar_manifesto(registros, "copia")
    resumo = salvar_resumo(registros, "copia")

    print("\n" + "=" * 78)
    print("DATASET FINALIZADO")
    print("=" * 78)
    print(f"Imagens copiadas: {len(registros)}")
    print(f"Manifesto:        {manifesto}")
    print(f"Resumo:           {resumo}")

    if erros_finais:
        print("\nERROS NA VALIDACAO FINAL:")
        for erro in erros_finais:
            print(f"- {erro}")
        return 1

    print("Validacao concluida: dataset balanceado e integro.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
