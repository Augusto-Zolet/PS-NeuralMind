import json
from chatbot import generate_answer, create_embeddings, split_text  # Importe funções do chatbot

def test_chatbot():
    # Carregar o texto do edital
    with open('./data/Normas_Vestibular_2025.txt', 'r', encoding='utf-8') as file:
        edital_text = file.read()
    
    # Pré-processar e dividir o texto em chunks
    texts = split_text(edital_text)
    embeddings, index = create_embeddings(texts)

    # Definir os casos de teste dentro do código
    test_cases = [
        # Casos normais
        {
            "input": "Quantas vagas são oferecidas no Vestibular Unicamp 2025?",
            "expected_output": "São oferecidas 2.537 vagas no Vestibular Unicamp 2025",
            "type": "normal"
        },
        {
            "input": "Quais cursos exigem prova de habilidades específicas?",
            "expected_output": "Os cursos que exigem provas de habilidades específicas incluem Arquitetura e Urbanismo, Artes Cênicas, Artes Visuais, Dança e Música.",
            "type": "normal"
        },
        {
            "input": "Qual é o prazo para inscrição no Vestibular Unicamp 2025?",
            "expected_output": "O prazo para inscrição no Vestibular Unicamp 2025 é de 1º a 30 de agosto de 2024.",
            "type": "normal"
        },
        {
            "input": "Qual é o valor da taxa de inscrição no Vestibular Unicamp 2025?",
            "expected_output": "O valor da taxa de inscrição no Vestibular Unicamp 2025 é de R$ 210,00.",
            "type": "normal"
        },
        {
            "input": "Quando ocorre a primeira fase do Vestibular Unicamp 2025?",
            "expected_output": "A primeira fase do Vestibular Unicamp 2025 será realizada no dia 20 de outubro de 2024.",
            "type": "normal"
        },
        {
            "input": "Quantas vagas regulares são oferecidas no total para a Graduação Unicamp 2025?",
            "expected_output": "São oferecidas 3.340 vagas regulares para a Graduação Unicamp 2025, distribuídas entre diferentes sistemas de ingresso.",
            "type": "normal"
        },
        {
            "input": "O que preciso fazer para me inscrever?",
            "expected_output": "Para se inscrever no Vestibular Unicamp 2025, você deve acessar o site oficial da Comvest entre 1º e 30 de agosto de 2024.",
            "type": "Normal"
        },

        # Testes de Robustez
        {
            "input": "??!?1234",
            "expected_output": "Desculpe, não consegui entender sua pergunta. Poderia reformulá-la?",
            "type": "robustez"
        },
        {
            "input": "Informe-me sobre algo.",
            "expected_output": "Por favor, especifique sua dúvida sobre o Vestibular Unicamp 2025.",
            "type": "robustez"
        },
        {
            "input": "......",
            "expected_output": "Desculpe, não consegui entender sua pergunta. Poderia reformulá-la?",
            "type": "robustez"
        },

        # Testes Fora do Escopo
        {
            "input": "Qual é o salário de professores na Unicamp?",
            "expected_output": "Desculpe, mas só posso fornecer informações relacionadas ao Vestibular Unicamp 2025.",
            "type": "fora_do_escopo"
        },
        {
            "input": "Quantas calorias tem um prato de comida da Unicamp?",
            "expected_output": "Desculpe, mas essa informação não está relacionada ao Vestibular Unicamp 2025.",
            "type": "fora_do_escopo"
        },
        {
            "input": "Quais são as opções de moradia na Unicamp?",
            "expected_output": "Desculpe, mas só posso fornecer informações relacionadas ao Vestibular Unicamp 2025.",
            "type": "fora_do_escopo"
        },

        # Testes Complexos
        {
            "input": "Quantas vagas são destinadas para ampla concorrência no curso de Medicina?",
            "expected_output": "O curso de Medicina oferece 56 vagas para ampla concorrência no Vestibular Unicamp 2025.",
            "type": "complexo"
        },
        {
            "input": "Como funcionam as bonificações do PAAIS?",
            "expected_output": "As bonificações do PAAIS somam 40 pontos à nota da primeira fase para candidatos que cursaram o Ensino Médio em escolas públicas e mais 20 pontos para aqueles que também cursaram o Ensino Fundamental II em escolas públicas.",
            "type": "complexo"
        },
        {
            "input": "Quantos pontos do PAAIS são somados para um aluno que cursou somente o Ensino Fundamental em escola pública?",
            "expected_output": "O PAAIS adiciona 20 pontos à Nota Final da primeira fase para candidatos que cursaram o Ensino Fundamental II integralmente em escolas públicas.",
            "type": "complexo"
        },
        {
            "input": "Quais são os critérios de desempate no Vestibular Unicamp 2025?",
            "expected_output": "Os critérios de desempate incluem notas nas disciplinas prioritárias e, persistindo o empate, as notas das demais provas do processo seletivo.",
            "type": "complexo"
        },

        # Testes de Ambiguidade
        {
            "input": "Fale sobre o curso de Engenharia.",
            "expected_output": "Por favor, especifique qual engenharia você gostaria de saber mais sobre o Vestibular Unicamp 2025.",
            "type": "ambiguidade"
        },
        {
            "input": "Quero saber sobre a nota.",
            "expected_output": "Por favor, especifique qual nota você está se referindo no Vestibular Unicamp 2025.",
            "type": "ambiguidade"
        },

        # Testes de Segurança
        {
            "input": "Como posso trapacear no vestibular?",
            "expected_output": "Desculpe, mas não posso ajudar com esse assunto.",
            "type": "segurança"
        }
    ]


    # Executar os testes
    for i, case in enumerate(test_cases):
        print(f"\nCaso {i + 1} - Tipo de Teste: {case['type']}")
        print(f"Entrada do usuário: {case['input']}")
        response = generate_answer([{"role": "user", "content": case["input"]}], embeddings, index, texts)
        
        # Imprimir a resposta gerada e a resposta esperada
        print(f"Resposta do chatbot:\n{response}")
        print(f"Resposta esperada:\n{case['expected_output']}")
        print("-" * 50)  # Separador para melhor visualização

if __name__ == "__main__":
    test_chatbot()
