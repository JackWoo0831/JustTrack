import os 
class VideoIO(IOType):
    support_types = {'path': str, 'file': str}

    def __init__(self, value: str):
        super().__init__(value)
        if not Path(self.value).exists():
            self.value = self.value[:-4]  # remove the wrong suffix (.mp4 .avi etc.)
            if not Path(self.value).exists():
                raise FileNotFoundError(f"No such file: '{self.value}'")
            
        if '.mp4' in value or '.avi' in value: self.type = 'file'

        self.root_path, self.images, self.cap = None, None, None
        if self.type == 'path':
            self.root_path = Path(self.value)
            self.images = sorted(os.listdir(self.root_path))
        else:
            self.cap = cv2.VideoCapture(value)

        self.cnt = 0

    def to_path(self) -> str:
        return self.to('path')

    def to_pil(self) -> Image.Image:
        return self.to('pil')

    def to_array(self) -> np.ndarray:
        return self.to('array')

    def to_file(self) -> IOBase:
        if self.type == 'path':
            return open(self.value, 'rb')
        else:
            file = BytesIO()
            self.to_pil().save(file, 'PNG')
            file.seek(0)
            return file
        
    def next_image(self) -> Image:
        if self.type == 'path':
            ret = Image.open(os.path.join(self.root_path, self.images[self.cnt]))
        else:
            ret, frame = self.cap.read()
            if not ret: return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret = Image.fromarray(frame)

        self.cnt += 1
        return ret 
    
    def is_finish(self) -> bool:
        if self.type == 'path':
            return self.cnt == len(self.images) - 1
        else:
            return False

    @classmethod
    def from_file(cls, file: IOBase) -> 'ImageIO':
        from PIL import Image
        return cls(Image.open(file))

    @staticmethod
    def _path_to_pil(path: str) -> Image.Image:
        return Image.open(path)

    @staticmethod
    def _path_to_array(path: str) -> np.ndarray:
        return np.array(Image.open(path).convert('RGB'))

    @staticmethod
    def _pil_to_path(image: Image.Image) -> str:
        filename = temp_path('image', '.png')
        image.save(filename)
        return filename

    @staticmethod
    def _pil_to_array(image: Image.Image) -> np.ndarray:
        return np.array(image.convert('RGB'))

    @staticmethod
    def _array_to_pil(image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    @staticmethod
    def _array_to_path(image: np.ndarray) -> str:
        filename = temp_path('image', '.png')
        Image.fromarray(image).save(filename)
        return filename


CatgoryToIO = {
    'image': ImageIO,
    'text': str,
    'audio': AudioIO,
    'bool': bool,
    'int': int,
    'float': float,
    'file': File,
    'video': VideoIO, 
}

__all__ = ['ImageIO', 'AudioIO', 'CatgoryToIO', 'Info', 'Annotated', 'VideoIO']