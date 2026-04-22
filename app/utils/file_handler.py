import os

import uuid

from fastapi import UploadFile

from app.config.settings import UPLOAD_DIR

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}

MAX_FILE_SIZE = 50 * 1024 * 1024


def is_valid_file_extension(filename: str) -> bool:

    ext = os.path.splitext(filename)[-1].lower()

    return ext in ALLOWED_EXTENSIONS


def save_temp_file(file: UploadFile) -> str:

    if not file.filename:

        raise ValueError("No filename provided")

    if not is_valid_file_extension(file.filename):

        raise ValueError(
            f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename)[-1]

    file_name = f"file_{uuid.uuid4().hex}{ext}"

    file_path = os.path.join(UPLOAD_DIR, file_name)

    try:

        with open(file_path, "wb") as f:

            content = file.file.read()

            if len(content) > MAX_FILE_SIZE:

                raise ValueError("File too large (max 50MB)")

            f.write(content)

        return file_path

    except Exception as e:

        if os.path.exists(file_path):

            os.remove(file_path)

        raise Exception(f"Error saving file: {str(e)}")
