from pathlib import Path
import shutil


SOURCE_DIR = Path("terorden_arananlar")
DEST_ROOT = Path("public/gallery/persons")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sanitize_folder_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in name).strip()
    return cleaned or "isimsiz"


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Kaynak klasor bulunamadi: {SOURCE_DIR}")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in SOURCE_DIR.iterdir():
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        person_name = sanitize_folder_name(image_path.stem)
        person_dir = DEST_ROOT / person_name
        person_dir.mkdir(parents=True, exist_ok=True)

        destination_file = person_dir / f"photo{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination_file)
        copied += 1
        print(f"Kopyalandi: {image_path.name} -> {destination_file}")

    print(f"Tamamlandi. Toplam {copied} fotograf kopyalandi.")


if __name__ == "__main__":
    main()
