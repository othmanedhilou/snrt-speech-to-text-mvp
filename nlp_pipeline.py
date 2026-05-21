"""NLP pipeline: topic classification + NER (spaCy) + summarization (Groq)."""
import re
import unicodedata
from typing import Any

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_ENABLED, MIN_WORDS_TO_SAVE

# ── spaCy — load best available French model ──────────────────────────────────
try:
    import spacy
    for _model in ("fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"):
        try:
            _nlp = spacy.load(_model)
            break
        except OSError:
            _nlp = None
    SPACY_AVAILABLE = _nlp is not None
except ImportError:
    _nlp = None
    SPACY_AVAILABLE = False

# ── Topic keywords ────────────────────────────────────────────────────────────
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "politique": [
        "gouvernement", "ministre", "roi", "parlement", "election", "loi",
        "decret", "depute", "president", "royal", "majeste", "trone",
        "politique", "parti", "vote", "senat", "chambre", "reforme",
    ],
    "sport": [
        "football", "raja", "wydad", "match", "but", "can", "far", "frmf",
        "botola", "sport", "equipe", "joueur", "entraineur", "coupe",
        "championnat", "ligue", "score", "victoire", "defaite", "stade",
    ],
    "economie": [
        "dirham", "pib", "economie", "budget", "investissement", "marche",
        "bourse", "banque", "inflation", "croissance", "entreprise", "emploi",
        "chomage", "export", "import", "commerce", "financier", "fiscal",
    ],
    "societe": [
        "education", "sante", "hopital", "ecole", "universite", "medecin",
        "patient", "logement", "transport", "securite", "social", "famille",
        "enfant", "femme", "jeunesse", "citoyen",
    ],
    "international": [
        "onu", "unesco", "union africaine", "europe", "france", "espagne",
        "international", "accord", "traite", "diplomatie", "ambassadeur",
        "sommet", "mondial", "usa", "etats-unis", "arabie",
    ],
    "culture": [
        "festival", "art", "musique", "film", "cinema", "theatre", "livre",
        "patrimoine", "culture", "artiste", "exposition", "spectacle",
    ],
    "meteo": [
        "meteo", "pluie", "temperature", "soleil", "nuage", "vent",
        "chaleur", "froid", "neige", "previsions", "climat",
    ],
    "faits_divers": [
        "accident", "incendie", "crime", "arrestation", "police",
        "gendarmerie", "tribunal", "jugement", "victime", "enquete",
    ],
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def classify_topic(text: str) -> str:
    norm = _normalize(text)
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in norm)
        if score:
            scores[topic] = score
    return max(scores, key=lambda k: scores[k]) if scores else "general"


def extract_entities(text: str) -> list[dict[str, str]]:
    if not SPACY_AVAILABLE or _nlp is None:
        return []
    doc = _nlp(text[:1_000_000])
    seen: set[tuple] = set()
    entities = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen and len(ent.text.strip()) > 1:
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_})
    return entities


def summarize(text: str, language: str = "fr") -> str | None:
    if not GROQ_ENABLED or not text.strip():
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        lang_tag = "en français" if language in ("fr", None) else "en arabe"
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Tu es un journaliste expert. Résume ce segment audio {lang_tag} "
                        "en 2-3 phrases factuelles. Inclus les noms, chiffres et faits clés. "
                        "Réponds uniquement avec le résumé, sans introduction."
                    ),
                },
                {"role": "user", "content": text[:2_500]},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def is_meaningful(text: str) -> bool:
    if len(text.split()) < MIN_WORDS_TO_SAVE:
        return False
    noise = [r"^\s*\[.*\]\s*$", r"^[\s\.\-♪♫]+$"]
    return not any(re.fullmatch(p, text.strip()) for p in noise)


def process(text: str, language: str = "fr") -> dict[str, Any]:
    """Run the full NLP pipeline on a transcribed text chunk.

    Returns a dict with keys: skip, topic, entities, summary.
    """
    if not is_meaningful(text):
        return {"skip": True}

    return {
        "skip": False,
        "topic": classify_topic(text),
        "entities": extract_entities(text),
        "summary": summarize(text, language),
    }
