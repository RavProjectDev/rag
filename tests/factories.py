import random
import string
import uuid

from rag.app.models.data import SanityData
from rag.app.schemas.data import Chunk, VectorEmbedding


def random_string(length=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def random_url():
    return f"http://{random_string(10)}.com"


def make_sanity_data(randomize=False, **overrides):
    if randomize:
        data = {
            "id": str(uuid.uuid4()),
            "slug": random_string(12),
            "title": random_string(20),
            "transcriptURL": random_url(),
            "hash": random_string(16),
        }
    else:
        data = {
            "id": "default_id",
            "slug": "default_slug",
            "title": "default_title",
            "transcriptURL": "http://default.url",
            "hash": "default_hash",
        }
    data.update(overrides)
    return SanityData(**data)


def random_srt_time():
    h = f"{random.randint(0, 1):02}"
    m = f"{random.randint(0, 59):02}"
    s = f"{random.randint(0, 59):02}"
    ms = f"{random.randint(0, 999):03}"
    return f"{h}:{m}:{s},{ms}"


def make_chunk(randomize=False, include_times=False, **overrides):
    data = {
        "full_text": random_string(50) if randomize else "default text",
        "text_to_embed": random_string(50) if randomize else "default text",
        "chunk_size": random.randint(50, 200) if randomize else 100,
        "name_space": random_string(15) if randomize else "default namespace",
        "embed_size": 50,
    }

    if include_times:
        data["time_start"] = random_srt_time() if randomize else "00:00:00,000"
        data["time_end"] = random_srt_time() if randomize else "00:00:05,000"

    data.update(overrides)
    return Chunk(**data)


def make_vector_embedding(
    randomize_vector=False,
    randomize_metadata=False,
    randomize_sanity_data=False,
    **overrides,
):
    vector_length = 748
    data = {
        "vector": (
            [random.uniform(-1, 1) for _ in range(vector_length)]
            if randomize_vector
            else [0.1] * vector_length
        ),
        "dimension": vector_length,
        "metadata": make_chunk(randomize=randomize_metadata),
        "sanity_data": make_sanity_data(randomize=randomize_sanity_data),
    }
    data.update(overrides)
    return VectorEmbedding(**data)
