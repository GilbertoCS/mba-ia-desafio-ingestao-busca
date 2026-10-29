from search import search_prompt


def main():
    print("Faça sua pergunta:")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            break

        try:
            answer = search_prompt(question)
        except Exception as exc:
            print(f"Erro ao responder: {exc}")
            continue

        print(f"RESPOSTA: {answer}")


if __name__ == "__main__":
    main()
