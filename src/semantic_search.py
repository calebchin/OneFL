import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#import fedisca  # Import federated learning module
import torch
import os

# Load SpaCy and pre-trained embedding model
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# All available MedMNIST datasets with full descriptions
medmnist_datasets = {
    "pathmnist": {"image_type": "colon pathology", "disease": "colon pathology"},
    "chestmnist": {"image_type": "chest x-ray", "disease": "thorax diseases"},
    "dermamnist": {"image_type": "dermatoscope", "disease": "dermatology disease"},
    "octmnist": {"image_type": "retinal oct", "disease": "retinal oct"},
    "pneumoniamnist": {"image_type": "chest x-ray", "disease": "pneumonia"},
    "retinamnist": {"image_type": "fundus camera", "disease": "fundus abnormalities"},
    "breastmnist": {"image_type": "breast ultrasound", "disease": "breast cancer"},
    "bloodmnist": {"image_type": "blood cell microscope", "disease": "blood cell abnormalities"},
    "tissuemnist": {"image_type": "kidney cortex microscope", "disease": "tissue abnormalities"},
    "organamnist": {"image_type": "abdominal ct", "disease": "abdominal organs"},
    "organcmnist": {"image_type": "abdominal ct", "disease": "abdominal organs"},
    "organsmnist": {"image_type": "abdominal ct", "disease": "abdominal organs"},
}

# Flattened terms for matching
image_type_terms = list(set([info['image_type'] for info in medmnist_datasets.values()]))
disease_terms = list(set([info['disease'] for info in medmnist_datasets.values()]))

# Embed terms for flexible matching
image_type_embeddings = embedding_model.encode(image_type_terms)
disease_embeddings = embedding_model.encode(disease_terms)
SIMILARITY_THRESHOLD = 0.3

def find_closest_match_with_alternatives(query, terms, term_embeddings, threshold=SIMILARITY_THRESHOLD):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, term_embeddings)[0]
    sorted_indices = similarities.argsort()[::-1]
    top_indices = sorted_indices[:3]
    top_matches = [(terms[idx], similarities[idx]) for idx in top_indices]

    for term, score in top_matches:
        if score > threshold:
            return term, None
    return None, top_matches

def extract_labels_dynamic(query, threshold=SIMILARITY_THRESHOLD):
    detected_image_type = None
    detected_disease = None
    image_alternatives = []
    disease_alternatives = []

    # Search for matches in image types and diseases
    term, alts = find_closest_match_with_alternatives(query, image_type_terms, image_type_embeddings, threshold)
    if term:
        detected_image_type = term
    else:
        image_alternatives.extend(alts or [])

    term, alts = find_closest_match_with_alternatives(query, disease_terms, disease_embeddings, threshold)
    if term:
        detected_disease = term
    else:
        disease_alternatives.extend(alts or [])

    # Match to medmnist datasets
    dataset_match = None
    for key, info in medmnist_datasets.items():
        if detected_image_type == info['image_type'] and detected_disease == info['disease']:
            dataset_match = key
            break

    return detected_image_type, detected_disease, dataset_match, image_alternatives, disease_alternatives

def evaluate_query_matcher(queries, threshold=SIMILARITY_THRESHOLD):
    correct_disease_count = 0
    correct_image_count = 0
    correct_dataset_count = 0
    total_queries = len(queries)

    for query_text, expected_image_type, expected_disease, expected_dataset in queries:
        # Extract detected labels and alternatives
        detected_image_type, detected_disease, dataset_match, image_alts, disease_alts = extract_labels_dynamic(
            query_text, threshold
        )

        # Correct Disease
        is_correct_disease = detected_disease == expected_disease
        correct_disease_count += is_correct_disease

        # Correct Image Type
        is_correct_image = detected_image_type == expected_image_type
        correct_image_count += is_correct_image

        # Correct Dataset
        is_correct_dataset = dataset_match == expected_dataset
        correct_dataset_count += is_correct_dataset

        print(f"Query: {query_text}")
        print(f"Expected Image Type: {expected_image_type}, Detected Image Type: {detected_image_type}")
        if not detected_image_type:
            print("I was unclear about the image type. Here are some possibilities:")
            for alt, score in image_alts:
                print(f"- {alt} (similarity: {score:.2f})")

        print(f"Expected Disease: {expected_disease}, Detected Disease: {detected_disease}")
        if not detected_disease:
            print("I was unclear about the disease. Here are some possibilities:")
            for alt, score in disease_alts:
                print(f"- {alt} (similarity: {score:.2f})")

        print(f"Expected Dataset: {expected_dataset}, Detected Dataset: {dataset_match}")
        print(f"Correct Image Type: {is_correct_image}, Correct Disease: {is_correct_disease}, Correct Dataset: {is_correct_dataset}\n")

    # Calculate metrics
    disease_accuracy = correct_disease_count / total_queries * 100
    image_accuracy = correct_image_count / total_queries * 100
    dataset_accuracy = correct_dataset_count / total_queries * 100

    print(f"Accuracy Metrics:")
    print(f"Disease Accuracy: {disease_accuracy:.2f}%")
    print(f"Image Type Accuracy: {image_accuracy:.2f}%")
    print(f"Dataset Accuracy: {dataset_accuracy:.2f}%")