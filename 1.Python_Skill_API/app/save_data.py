import json

def save_to_json(data, filename):
    """
    Menyimpan data ke dalam file JSON.

    :param data: Data yang akan disimpan.
    :param filename: Nama file JSON.
    """
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
