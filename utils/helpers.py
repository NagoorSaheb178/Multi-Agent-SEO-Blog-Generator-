def save_to_file(filename, content):
    with open(filename, "w") as file:
        file.write(content)
    print(f"Content saved to {filename}")