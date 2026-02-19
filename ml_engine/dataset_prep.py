import os, shutil, argparse
from PIL import Image

def extract_indian_faces(source_dir, dest_dir):
    """Extract Indian faces from UTKFace dataset (race code = 3)"""
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    errors = 0

    print(f"Scanning {source_dir} for Indian faces...")

    for fname in os.listdir(source_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            # UTKFace format: age_gender_race_timestamp.jpg
            # race codes: 0=White 1=Black 2=Asian 3=Indian 4=Other
            parts = fname.split("_")
            if len(parts) >= 3 and parts[2] == "3":
                src_path = os.path.join(source_dir, fname)
                dst_path = os.path.join(dest_dir, f"utkface_indian_{count:05d}.jpg")

                # Verify image opens correctly
                img = Image.open(src_path)
                img = img.resize((256, 256))
                img.save(dst_path, quality=95)
                count += 1

                if count % 100 == 0:
                    print(f"Extracted {count} Indian faces...")
        except Exception as e:
            errors += 1
            continue

    print(f"Done. Extracted {count} Indian faces. Errors: {errors}")
    return count

def check_dataset(real_dir="dataset/real", fake_dir="dataset/fake"):
    """Check dataset counts and image validity"""
    real_count = len([f for f in os.listdir(real_dir)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
    fake_count = len([f for f in os.listdir(fake_dir)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])

    print(f"Real faces: {real_count}")
    print(f"Fake faces: {fake_count}")
    print(f"Total: {real_count + fake_count}")

    if real_count < 5000:
        print("WARNING: Real faces count is low. Recommend 10000+")
    if fake_count < 5000:
        print("WARNING: Fake faces count is low. Recommend 10000+")
    if real_count >= 8000 and fake_count >= 8000:
        print("Dataset looks good. Ready to train.")

def resize_all(directory, size=(256, 256)):
    """Resize all images in directory to target size"""
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(('.jpg','.jpeg','.png'))]

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(size, Image.LANCZOS)
            img.save(path, quality=95)
        except:
            os.remove(path)

        if i % 500 == 0:
            print(f"Resized {i}/{len(files)}")

    print(f"Resized {len(files)} images in {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Source directory for UTKFace")
    parser.add_argument("--dest", help="Destination directory")
    parser.add_argument("--check", action="store_true", help="Check dataset counts")
    parser.add_argument("--resize", help="Resize all images in directory")
    args = parser.parse_args()

    if args.check:
        check_dataset()
    elif args.source and args.dest:
        extract_indian_faces(args.source, args.dest)
    elif args.resize:
        resize_all(args.resize)
