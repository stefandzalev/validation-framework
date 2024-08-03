def snake_case_to_camel_case_with_space(text: str) -> str:
    return " ".join(w.capitalize() for w in text.split("_"))
