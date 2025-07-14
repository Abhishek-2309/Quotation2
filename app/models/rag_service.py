from byaldi import RAGMultiModalModel
import base64, io
from PIL import Image
import uuid

RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", verbose=1)

def index_pdf(path: str):
    unique_name = f"uploaded_{uuid.uuid4().hex[:8]}"
    RAG.index(
        input_path=path,
        index_name=unique_name,
        store_collection_with_index=True,
        overwrite=True
    )

def search_image(question: str):
    results = RAG.search(question, k=1, index_name=index_name)
    image_bytes = base64.b64decode(results[0].base64)
    return Image.open(io.BytesIO(image_bytes))
