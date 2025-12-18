import io
import json
import random
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from PIL import Image
from google.cloud import storage
from google.api_core.exceptions import NotFound


class PieceClsDatasetConfig:
    """
    Simple mutable config container (no dataclass).
    """

    def __init__(
        self,
        credentials_json: str = "ai-2025-cv-project-53890dfc6d30.json",
        bucket_name: str = "ai-cv-2025-photos",
        source_prefix: str = "single-piece/",            # images: single-piece/<piece_id>/<img_index>.jpg
        labels_prefix: str = "single-piece-labels/",     # labels: single-piece-labels/<piece_id>/<img_index>.txt
                                                        
        out_dir: Path = Path("single_piece_dataset"),
        train_ratio: float = 0.80,
        random_seed: int = 123,
        expand_ratio: float = 0.05,
        pad_to_square: bool = True,
        square_bg: Tuple[int, int, int] = (114, 114, 114),
    ):
        self.credentials_json = credentials_json
        self.bucket_name = bucket_name
        self.source_prefix = source_prefix
        self.labels_prefix = labels_prefix
        self.out_dir = out_dir
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.expand_ratio = expand_ratio
        self.pad_to_square = pad_to_square
        self.square_bg = square_bg


class PieceClsDatasetBuilder:
    def __init__(self, cfg: PieceClsDatasetConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)

    # ---------------- AUTH ----------------
    def make_storage_client(self) -> storage.Client:
        p = Path(self.cfg.credentials_json)
        if not p.exists():
            raise FileNotFoundError(f"Missing credentials file: {p.resolve()}")

        data = json.loads(p.read_text(encoding="utf-8"))

        # Service account JSON
        if data.get("type") == "service_account":
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(str(p))
            return storage.Client(credentials=creds)

        # OAuth client_secret.json
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        token_path = p.with_name("gcs_oauth_token.json")
        scopes = ["https://www.googleapis.com/auth/devstorage.read_only"]

        creds: Optional[Credentials] = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes=scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(p), scopes=scopes)
                try:
                    creds = flow.run_local_server(port=0)
                except Exception:
                    creds = flow.run_console()

            token_path.write_text(creds.to_json(), encoding="utf-8")
            print(f"[AUTH] Saved token to {token_path}")

        return storage.Client(credentials=creds)

    # ---------------- UTILS ----------------
    @staticmethod
    def norm_stem(stem: str) -> str:
        """Normalize numeric stems: '009' -> '9'."""
        return str(int(stem)) if stem.isdigit() else stem

    def list_piece_ids(self, bucket: storage.Bucket) -> List[str]:
        it = bucket.list_blobs(prefix=self.cfg.source_prefix, delimiter="/")
        _ = list(it)  # force fetch
        prefixes = list(getattr(it, "prefixes", []))

        piece_ids: List[str] = []
        for pref in prefixes:
            s = pref[len(self.cfg.source_prefix) :].strip("/")
            if s:
                piece_ids.append(s)

        piece_ids.sort(key=lambda x: int(x) if x.isdigit() else x)
        return piece_ids

    @staticmethod
    def parse_yolo_txt(txt_bytes: bytes) -> Tuple[float, float, float, float]:
        line = txt_bytes.decode("utf-8").strip().split()
        if len(line) < 5:
            raise ValueError(f"Bad label line: {line}")
        xc = float(line[1])
        yc = float(line[2])
        w = float(line[3])
        h = float(line[4])
        return xc, yc, w, h

    @staticmethod
    def yolo_to_xyxy(
        xc: float,
        yc: float,
        w: float,
        h: float,
        W: int,
        H: int,
        expand_ratio: float = 0.0,
    ) -> Tuple[int, int, int, int]:
        bw = w * W
        bh = h * H
        cx = xc * W
        cy = yc * H

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        ex = bw * expand_ratio
        ey = bh * expand_ratio
        x1 -= ex
        x2 += ex
        y1 -= ey
        y2 += ey

        x1 = max(0, min(W - 1, int(round(x1))))
        y1 = max(0, min(H - 1, int(round(y1))))
        x2 = max(0, min(W - 1, int(round(x2))))
        y2 = max(0, min(H - 1, int(round(y2))))

        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)

        return x1, y1, x2, y2

    @staticmethod
    def pad_to_square_rgb(im: Image.Image, bg: Tuple[int, int, int] = (114, 114, 114)) -> Image.Image:
        w, h = im.size
        s = max(w, h)
        out = Image.new("RGB", (s, s), bg)
        out.paste(im, ((s - w) // 2, (s - h) // 2))
        return out

    @staticmethod
    def save_jpg(im: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format="JPEG", quality=95)

    def build_label_map(self, bucket: storage.Bucket) -> Dict[Tuple[str, str], storage.Blob]:
        """
        Build lookup: (piece_id, img_index_stem_norm) -> label blob.
        Handles both:
          single-piece-labels/12/9.txt
          single-piece-labels/12\\9.txt   (literal backslash in object name)
        """
        label_map: Dict[Tuple[str, str], storage.Blob] = {}
        blobs = bucket.list_blobs(prefix=self.cfg.labels_prefix)

        count = 0
        for b in blobs:
            name = b.name
            if not name.lower().endswith(".txt"):
                continue

            rel = name[len(self.cfg.labels_prefix) :]
            rel_norm = rel.replace("\\", "/")
            parts = rel_norm.split("/", 1)
            if len(parts) != 2:
                continue

            pid = parts[0].strip()
            fname = parts[1].strip()
            stem = Path(fname).stem
            key = (pid, self.norm_stem(stem))

            label_map[key] = b
            count += 1

        print(f"[INFO] Indexed {count} label files into map")
        return label_map

    # ---------------- MAIN ----------------
    def run(self) -> None:
        client = self.make_storage_client()
        bucket = client.bucket(self.cfg.bucket_name)

        piece_ids = self.list_piece_ids(bucket)
        if not piece_ids:
            raise SystemExit("No piece folders found. Check BUCKET_NAME/SOURCE_PREFIX.")

        print(f"[INFO] Found {len(piece_ids)} piece folders")

        label_map = self.build_label_map(bucket)

        train_root = self.cfg.out_dir / "train"
        val_root = self.cfg.out_dir / "val"
        train_root.mkdir(parents=True, exist_ok=True)
        val_root.mkdir(parents=True, exist_ok=True)

        classes_txt = self.cfg.out_dir / "classes.txt"
        classes_txt.write_text("\n".join([f"piece_{pid}" for pid in piece_ids]) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {classes_txt}")

        total_written = 0

        for pid in piece_ids:
            img_prefix = f"{self.cfg.source_prefix}{pid}/"
            blobs = list(bucket.list_blobs(prefix=img_prefix))
            img_blobs = [b for b in blobs if b.name.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not img_blobs:
                print(f"[WARN] No images for piece {pid}")
                continue

            def img_key(b: storage.Blob):
                name = Path(b.name).name
                stem = Path(name).stem
                return int(stem) if stem.isdigit() else stem

            img_blobs.sort(key=img_key)

            idxs = list(range(len(img_blobs)))
            self.rng.shuffle(idxs)

            n_train = max(1, int(round(len(img_blobs) * self.cfg.train_ratio)))
            train_set = set(idxs[:n_train])

            piece_train_count = 0
            piece_val_count = 0

            for bi in idxs:
                b = img_blobs[bi]
                img_name = Path(b.name).name  # e.g. "9.jpg"
                stem = self.norm_stem(Path(img_name).stem)  # normalized

                label_blob = label_map.get((pid, stem))
                if label_blob is None:
                    print(f"[WARN] Missing label for piece {pid} image {img_name} (expected key {pid}/{stem}.txt). Skip.")
                    continue

                try:
                    img_bytes = b.download_as_bytes()
                    txt_bytes = label_blob.download_as_bytes()
                except NotFound:
                    print(f"[WARN] NotFound downloading: {b.name} or {label_blob.name} (skip)")
                    continue

                im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                W, H = im.size

                xc, yc, w, h = self.parse_yolo_txt(txt_bytes)
                x1, y1, x2, y2 = self.yolo_to_xyxy(
                    xc, yc, w, h, W, H, expand_ratio=self.cfg.expand_ratio
                )
                crop = im.crop((x1, y1, x2, y2))

                if self.cfg.pad_to_square:
                    crop = self.pad_to_square_rgb(crop, bg=self.cfg.square_bg)

                class_folder = f"piece_{pid}"
                if bi in train_set:
                    piece_train_count += 1
                    out_path = train_root / class_folder / f"img{piece_train_count:03d}.jpg"
                else:
                    piece_val_count += 1
                    out_path = val_root / class_folder / f"img{piece_val_count:03d}.jpg"

                self.save_jpg(crop, out_path)
                total_written += 1

            print(f"[OK] piece {pid}: train={piece_train_count} val={piece_val_count}")

        print(f"\n[DONE] Wrote {total_written} cropped images into: {self.cfg.out_dir.resolve()}")
        print("      Mapping file:", (self.cfg.out_dir / "classes.txt").resolve())