"""Client HTTP entre Streamlit et FastAPI."""

from __future__ import annotations

from typing import Any

import requests


class APIClientError(RuntimeError):
    """Erreur présentable à l'utilisateur de l'interface."""


def get_health(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIClientError("L'API FastAPI n'est pas joignable.") from exc
    except ValueError as exc:
        raise APIClientError("La réponse de l'API est invalide.") from exc


def request_prediction(
    base_url: str, payload: dict[str, Any], timeout: float = 10.0
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/predict", json=payload, timeout=timeout
        )
        if response.status_code == 422:
            raise APIClientError("Les mesures saisies ne sont pas valides.")
        if response.status_code == 503:
            detail = response.json().get("detail", "Le modèle est indisponible.")
            raise APIClientError(str(detail))
        response.raise_for_status()
        return response.json()
    except APIClientError:
        raise
    except requests.ConnectionError as exc:
        raise APIClientError(
            "FastAPI n'est pas lancé. Démarrez-le avec : uvicorn main:app --reload"
        ) from exc
    except requests.Timeout as exc:
        raise APIClientError("L'API met trop de temps à répondre.") from exc
    except requests.RequestException as exc:
        raise APIClientError("Une erreur est survenue pendant l'appel à l'API.") from exc
    except ValueError as exc:
        raise APIClientError("La réponse de l'API est invalide.") from exc
