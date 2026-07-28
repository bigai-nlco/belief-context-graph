def load_instruction(path: str) -> str:
    with open(path, "r") as file:
        return file.read().strip()


def load_protocol(path: str) -> str:
    with open(path, "r") as file:
        return file.read().strip()
