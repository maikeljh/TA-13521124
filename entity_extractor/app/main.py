import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

metadata = {
  "database": "sakila",
  "tables": [
    {
      "name": "actor",
      "description": "Table for storing actor information.",
      "columns": [
        {"key": "actor_id", "name": "actor_id", "type": "INT", "nullable": False, "attributes": ["PRIMARY KEY", "AUTO_INCREMENT"], "references": None, "description": "Unique identifier for each actor."},
        {"key": "first_name", "name": "first_name", "type": "VARCHAR(45)", "nullable": False, "attributes": [], "references": None, "description": "First name of the actor."},
        {"key": "last_name", "name": "last_name", "type": "VARCHAR(45)", "nullable": False, "attributes": [], "references": None, "description": "Last name of the actor."},
        {"key": "last_update", "name": "last_update", "type": "TIMESTAMP", "nullable": False, "attributes": ["DEFAULT CURRENT_TIMESTAMP"], "references": None, "description": "Timestamp for the last update of the record."}
      ],
      "relations": [
        {"foreign_table": "film_actor", "primary_table": "actor", "join": "actor.actor_id = film_actor.actor_id", "description": "Actor is related to films via the film_actor table, representing the films in which the actor appeared."}
      ],
      "unique_keys": [
        {"name": "actor_unique", "columns": ["actor_id"], "description": "The actor_id column is unique for each actor in the table."}
      ]
    },
    {
      "name": "film_actor",
      "description": "Join table for storing the relationship between films and actors.",
      "columns": [
        {"key": "actor_id", "name": "actor_id", "type": "INT", "nullable": False, "attributes": ["PRIMARY KEY"], "references": {"table": "actor", "column": "actor_id"}, "description": "Foreign key referencing actor_id in the actor table."},
        {"key": "film_id", "name": "film_id", "type": "INT", "nullable": False, "attributes": ["PRIMARY KEY"], "references": {"table": "film", "column": "film_id"}, "description": "Foreign key referencing film_id in the film table."}
      ],
      "relations": [
        {"foreign_table": "actor", "primary_table": "film_actor", "join": "film_actor.actor_id = actor.actor_id", "description": "Film_actor table references the actor table, linking each actor to the films they appeared in."},
        {"foreign_table": "film", "primary_table": "film_actor", "join": "film_actor.film_id = film.film_id", "description": "Film_actor table references the film table, linking each film to its actors."}
      ],
      "unique_keys": [
        {"name": "film_actor_unique", "columns": ["actor_id", "film_id"], "description": "The combination of actor_id and film_id is unique in the film_actor table."}
      ]
    },
    {
      "name": "film",
      "description": "Table for storing film details.",
      "columns": [
        {"key": "film_id", "name": "film_id", "type": "INT", "nullable": False, "attributes": ["PRIMARY KEY", "AUTO_INCREMENT"], "references": None, "description": "Unique identifier for each film."},
        {"key": "title", "name": "title", "type": "VARCHAR(255)", "nullable": False, "attributes": [], "references": None, "description": "Title of the film."},
        {"key": "release_year", "name": "release_year", "type": "YEAR", "nullable": True, "attributes": [], "references": None, "description": "The release year of the film."},
        {"key": "language_id", "name": "language_id", "type": "TINYINT", "nullable": False, "attributes": [], "references": {"table": "language", "column": "language_id"}, "description": "Foreign key referencing the language of the film."},
        {"key": "last_update", "name": "last_update", "type": "TIMESTAMP", "nullable": False, "attributes": ["DEFAULT CURRENT_TIMESTAMP"], "references": None, "description": "Timestamp for the last update of the record."}
      ],
      "relations": [
        {"foreign_table": "language", "primary_table": "film", "join": "film.language_id = language.language_id", "description": "Film table references the language table to store the language of the film."}
      ],
      "unique_keys": [
        {"name": "film_unique", "columns": ["film_id"], "description": "The film_id column is unique for each film."}
      ]
    }
  ]
}

def generate_metadata_text(metadata):
    metadata_texts = []
    for table in metadata["tables"]:
        table_text = f"Table: {table['name']} | Description: {table['description']}"
        metadata_texts.append(table_text)
        
        for column in table["columns"]:
            column_text = f"  Column: {column['name']} | Type: {column['type']} | Nullable: {column['nullable']} | Attributes: {', '.join(column['attributes'])} | References: {column['references']} | Description: {column['description']}"
            metadata_texts.append(column_text)
        
        for relation in table["relations"]:
            relation_text = f"  Relation: {relation['primary_table']} -> {relation['foreign_table']} | Join: {relation['join']} | Description: {relation['description']}"
            metadata_texts.append(relation_text)

        for unique_key in table["unique_keys"]:
            unique_key_text = f"  Unique Key: {unique_key['name']} | Columns: {', '.join(unique_key['columns'])} | Description: {unique_key['description']}"
            metadata_texts.append(unique_key_text)
    
    return metadata_texts

metadata_texts = generate_metadata_text(metadata)

metadata_embeddings = model.encode(metadata_texts)

metadata_embeddings = np.array(metadata_embeddings).astype('float32')

index = faiss.IndexFlatL2(metadata_embeddings.shape[1])
index.add(metadata_embeddings)

metadata_texts_map = metadata_texts

def link_entities_to_metadata(entities, model, index, metadata_texts_map):
    linked_entities = {}

    for entity in entities:
        entity_embedding = model.encode([entity])[0].astype('float32').reshape(1, -1)

        D, I = index.search(entity_embedding, k=1)
        
        linked_metadata = metadata_texts_map[I[0][0]]
        
        if "Table:" in linked_metadata:
            table_name = linked_metadata.split(":")[1].split("|")[0].strip()
            linked_entities[entity] = table_name
    
    return linked_entities

input_query = "Which films feature the actor 'Tom Cruise'?"

# TODO: NER MODEL to get entities
entities_metadata = ["actor", "film", "film_actor"]

linked_entities = link_entities_to_metadata(entities_metadata, model, index, metadata_texts_map)

print("Linked Entities to Tables:")
for entity, table in linked_entities.items():
    print(f"{entity} is linked to table: {table}")
