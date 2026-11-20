package br.com.casaraoia.data

object BuildingRepository {
    private val buildings = mapOf(
        "museu_casa_de_nhozinho" to BuildingInfo(
            "museu_casa_de_nhozinho", "Museu Casa de Nhozinho",
            "Espaço de valorização da cultura popular maranhense",
            "O museu preserva e divulga manifestações da cultura popular do Maranhão, incluindo artesanato, modos de vida e saberes tradicionais.",
            "A instituição homenageia Antônio Bruno Pinto Nogueira, conhecido como Nhozinho, artista popular reconhecido por trabalhos com a fibra do buriti.",
            "Casarão do conjunto urbano da Praia Grande, com fachada vertical, múltiplos pavimentos e vãos regulares.",
            "Rua Portugal, Praia Grande, Centro Histórico de São Luís – MA.",
            "Secretaria de Estado da Cultura do Maranhão"
        ),
        "palacio_cristo_rei" to BuildingInfo(
            "palacio_cristo_rei", "Palácio Cristo Rei",
            "Memorial e edifício histórico da UFMA",
            "Edifício histórico vinculado à Universidade Federal do Maranhão, com atividades administrativas e de preservação da memória institucional.",
            "Foi residência, sede de instituições de ensino, espaço do Arcebispado e sede da Reitoria. Após incêndio em 1991, foi restaurado e reinaugurado em 1992.",
            "Sobrado de três pavimentos ligado à tradição luso-maranhense, com jardim, balcões, portões e elementos artísticos.",
            "Praça Gonçalves Dias, Centro de São Luís – MA.",
            "Universidade Federal do Maranhão"
        ),
        "palacio_de_la_ravardiere" to BuildingInfo(
            "palacio_de_la_ravardiere", "Palácio de La Ravardière",
            "Sede da administração municipal de São Luís",
            "Edifício representativo do núcleo administrativo e histórico da cidade, utilizado como sede da Prefeitura Municipal.",
            "Construído por volta de 1689, abrigou a antiga Casa de Câmara e Cadeia e passou por sucessivas reformas.",
            "Fachada simétrica de dois pavimentos, balcão balustrado, cantaria, estuques e vãos em arco abatido.",
            "Praça Dom Pedro II, Centro Histórico de São Luís – MA.",
            "Prefeitura Municipal de São Luís"
        ),
        "palacio_dos_leoes" to BuildingInfo(
            "palacio_dos_leoes", "Palácio dos Leões",
            "Sede do Poder Executivo do Estado do Maranhão",
            "Uma das edificações mais representativas do conjunto histórico de São Luís e sede do Poder Executivo estadual.",
            "Ocupa uma área associada ao núcleo fundacional da cidade e passou por transformações até adquirir a configuração palaciana atual.",
            "Composição monumental, fachada simétrica, grandes vãos e salões com acervos de mobiliário, pinturas e objetos artísticos.",
            "Avenida Dom Pedro II, Centro Histórico de São Luís – MA.",
            "Acervo Digital do Palácio dos Leões"
        ),
        "teatro_arthur_azevedo" to BuildingInfo(
            "teatro_arthur_azevedo", "Teatro Arthur Azevedo",
            "Um dos principais espaços de artes cênicas do Maranhão",
            "Espaço cultural destinado a espetáculos de teatro, música, dança e outras manifestações artísticas.",
            "Construído entre 1816 e 1817, foi inaugurado como Teatro União, depois chamado Teatro São Luiz e recebeu o nome atual em homenagem ao dramaturgo maranhense.",
            "Arquitetura neoclássica, plateia em ferradura, camarotes, galerias, salão nobre e foyer.",
            "Rua do Sol, Centro Histórico de São Luís – MA.",
            "Secretaria de Estado da Cultura do Maranhão"
        )
    )
    fun findByLabel(label: String) = buildings[label]
}
