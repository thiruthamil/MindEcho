# lab/sanity_store.py
from app.storage import VectorStore
from app.db import DB
from app.settings import settings


def main() -> None:
    vs = VectorStore(settings.data_dir, settings.embeddings_model)
    db = DB(f"{settings.data_dir}/entries.sqlite")

    text = "felt nervous about viva, quick rehearsal helped"
    faiss_id, entry_id = vs.add(text)
    db.insert_entry(faiss_id, entry_id, text)

    ids, sims = vs.search("nervous before exam", k=3)
    rows = db.fetch_by_ids(ids)

    print(list(zip(sims, [r[3] for r in rows])))


if __name__ == "__main__":
    main()
