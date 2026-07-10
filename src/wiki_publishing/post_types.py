"""LinkedIn post archetypes (ADR-024) — ``post_type`` drives source + prompt.

Each :class:`PostType` maps to a :class:`PostTypeSpec` that selects (1) which wiki
categories the content search draws from and (2) a system-prompt *overlay* layered
over the shared ADR-021 "profesional cercano" base voice, plus a length target and
an attachment policy. The base voice and the no-fabrication / URL-citation rules in
``linkedin_draft._SYSTEM_PROMPT`` are never dropped — overlays only shape structure,
length, audience, and which artifact (if any) the human attaches by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PostType(StrEnum):
    """The supported LinkedIn post archetypes (ADR-024, extended by ADR-044)."""

    REPO_DEEP_DIVE = "repo_deep_dive"
    SHOWCASE = "showcase"
    TUTORIAL = "tutorial"
    INFORMATIVO = "informativo"
    # Build-in-public family (ADR-026) — about the USER'S OWN project, sourced
    # from local commits + ADRs (via ProjectContextSource), first-person.
    PROJECT_LAUNCH = "project_launch"
    PROJECT_FEATURE = "project_feature"
    PROJECT_WEEKLY = "project_weekly"
    # Third-party educational explainer (ADR-044) — teach the audience about a
    # concept/paper/result you did NOT build; structured, evidence-backed.
    EXPLAINER = "explainer"


class Focus(StrEnum):
    """The lens a post takes on its subject (ADR-024 amendment A1) — orthogonal
    to :class:`PostType`. ``use`` = user/value perspective (what it does for you,
    capabilities, benchmarks-as-benefits, when to use it); ``code`` = internals
    (architecture, modules, design)."""

    USE = "use"
    CODE = "code"


_FOCUS_OVERLAYS: dict[Focus, str] = {
    Focus.USE: (
        "ENFOQUE = USO (perspectiva del usuario). Prioriza: qué problema resuelve, "
        "qué puedes HACER con él, sus capacidades y cualquier benchmark COMO BENEFICIO "
        "(no como spec interna), cuándo conviene usarlo y frente a qué alternativas, y "
        "cómo empezar/instalarlo si el material lo indica. MINIMIZA el detalle de "
        "arquitectura interna (módulos, conteo de nodos, bounded contexts, % de "
        "encapsulación) salvo que sea EL diferenciador para quien lo usa."
    ),
    Focus.CODE: (
        "ENFOQUE = CÓDIGO/ARQUITECTURA (para ingenieros que evalúan el interior). "
        "Prioriza cómo está construido: estructura, módulos/contextos clave y cómo se "
        "relacionan, decisiones de diseño y dependencias."
    ),
}


def focus_overlay(focus: Focus) -> str:
    return _FOCUS_OVERLAYS[focus]


def coerce_focus(value: str | Focus | None) -> Focus | None:
    """Map a string/None to a :class:`Focus`; ``None``/unknown -> ``None`` so the
    caller falls back to the post type's default focus."""
    if isinstance(value, Focus):
        return value
    if not value:
        return None
    try:
        return Focus(str(value).strip().lower())
    except ValueError:
        return None


# Attachment policy values (kept as plain strings for a simple, testable contract).
ATTACH_DIAGRAM = "diagram"  # resolve the ADR-023 diagram PNG for manual attach
ATTACH_NONE = "none"  # no image attachment

# Bullet render style (ADR-044) — consumed at PUBLISH time by
# ``linkedin_publish.format_for_linkedin``. ``dot`` keeps today's ``• `` for all
# 7 pre-ADR-044 archetypes (byte-identical); ``arrow`` opts the ``explainer`` into
# a friendlier ``➡️ `` bullet.
BULLET_DOT = "dot"
BULLET_ARROW = "arrow"


@dataclass(frozen=True, slots=True)
class PostTypeSpec:
    """The contract for one archetype."""

    post_type: PostType
    system_overlay: str
    categories: tuple[str, ...]
    """Wiki page-path category segments the search is biased to (``()`` = no filter)."""
    max_chars: int
    attachment: str  # ATTACH_DIAGRAM | ATTACH_NONE
    default_focus: Focus = Focus.USE
    """The lens used when the caller doesn't pass an explicit ``focus`` (ADR-024 A1)."""
    bullet_style: str = BULLET_DOT
    """Publish-time bullet render style (ADR-044): ``dot`` (default, ``• ``) or
    ``arrow`` (``➡️ ``). Opt-in per archetype so no existing post changes shape."""


_DEEP_DIVE = PostTypeSpec(
    post_type=PostType.REPO_DEEP_DIVE,
    system_overlay=(
        "TIPO DE POST: análisis profundo de un repositorio para ingenieros.\n"
        "Estructura: (1) qué es y qué problema resuelve, (2) la arquitectura — sus "
        "bounded contexts / módulos clave y cómo se relacionan, (3) una decisión de "
        "diseño que valga la pena destacar, (4) para quién es útil. Tono técnico pero "
        "claro. Si se adjunta un diagrama, invita a verlo ('te dejo el diagrama de "
        "arquitectura'). Usa SOLO módulos/datos presentes en el material."
    ),
    categories=("sources",),
    max_chars=2400,
    attachment=ATTACH_DIAGRAM,
    default_focus=Focus.CODE,
)

_SHOWCASE = PostTypeSpec(
    post_type=PostType.SHOWCASE,
    system_overlay=(
        "TIPO DE POST: spotlight de una herramienta/repo (tercera persona, 'miren esto').\n"
        "Engancha en la primera línea, di QUÉ HACE y POR QUÉ IMPORTA, y cierra con el "
        "link al repo. Elige el sub-tono según el material:\n"
        "- si el material trae benchmarks/números reales (throughput, latencia, tamaño, "
        "comparativas), DESTÁCALOS — es el gancho (sub-tono specs/benchmark);\n"
        "- si no, cuéntalo como un descubrimiento (sub-tono narrativo).\n"
        "REGLA DURA: cualquier número, benchmark o comparativa debe aparecer VERBATIM en "
        "el material — NUNCA inventes ni redondees cifras (un benchmark falso es mentir "
        "sobre el proyecto de alguien). Si no hay números en el material, no los menciones."
    ),
    categories=("sources",),
    max_chars=2200,
    attachment=ATTACH_DIAGRAM,
)

_TUTORIAL = PostTypeSpec(
    post_type=PostType.TUTORIAL,
    system_overlay=(
        "TIPO DE POST: tutorial práctico ('cómo hacer X') para practicantes.\n"
        "Estructura: el problema en 1 línea → pasos concretos y accionables → un cierre "
        "con el aprendizaje clave.\n"
        "CÓDIGO: incluye un snippet SOLO si aparece en el material; preséntalo en un bloque "
        "de código. NUNCA inventes ni completes código que no esté en el material — si no "
        "hay código, escribe la guía conceptual sin snippet (no lo fabriques)."
    ),
    categories=("concepts", "sources"),
    max_chars=2600,
    attachment=ATTACH_NONE,
)

_INFORMATIVO = PostTypeSpec(
    post_type=PostType.INFORMATIVO,
    system_overlay=(
        "TIPO DE POST: reflexión informativa/opinión sobre un tema o tendencia (NO sobre "
        "un repo concreto). Audiencia amplia.\n"
        "UNA idea fuerte, accesible, breve y con punto de vista propio. Evita jerga "
        "innecesaria y listas largas. No fuerces enlaces ni detalles de implementación. "
        "Usa solo afirmaciones que el material respalde."
    ),
    categories=("concepts", "entities"),
    max_chars=1400,
    attachment=ATTACH_NONE,
)

_PROJECT_LAUNCH = PostTypeSpec(
    post_type=PostType.PROJECT_LAUNCH,
    system_overlay=(
        "TIPO DE POST: LANZAMIENTO de TU propio proyecto (build-in-public, primera "
        "persona). Es la presentación — se hace UNA sola vez. Estructura: qué estás "
        "construyendo y por qué (el problema que te llevó a hacerlo) → qué hace hoy → "
        "para quién → invitación a seguir el avance. Tono: 'estoy construyendo X'. "
        "Usa SOLO lo que el material (README/visión + ADRs) respalda; no exageres el estado."
    ),
    categories=(),
    max_chars=2200,
    attachment=ATTACH_NONE,
)

_PROJECT_FEATURE = PostTypeSpec(
    post_type=PostType.PROJECT_FEATURE,
    system_overlay=(
        "TIPO DE POST: FEATURE/decisión que construiste (build-in-public, primera "
        "persona, estilo war-story). Estructura objetivo:\n"
        "1. GANCHO = el síntoma/problema que el lector reconoce (no 'construí X').\n"
        "2. DIAGNÓSTICO = la causa real.\n"
        "3. LA DECISIÓN/FIX = qué cambiaste y por qué (la decisión del ADR).\n"
        "4. NÚMEROS MEDIDOS (antes → después) si el material los tiene.\n"
        "5. 2-3 lecciones generalizables.\n"
        "6. CTA (repo/link).\n"
        "REGLA DURA: todo número/benchmark/medición debe aparecer VERBATIM en el "
        "material (ADR/commits/benchmark) — NUNCA inventes ni redondees una medición. "
        "Si el material no tiene números, escribe el post SIN números (no fabriques un speedup)."
    ),
    categories=(),
    max_chars=2600,
    attachment=ATTACH_NONE,
)

_PROJECT_WEEKLY = PostTypeSpec(
    post_type=PostType.PROJECT_WEEKLY,
    system_overlay=(
        "TIPO DE POST: BUILD-LOG / resumen de lo que shippeaste en el periodo (primera "
        "persona). Lista breve de lo enviado con su POR QUÉ, a partir de los commits/ADRs "
        "reales del material. Cierra con qué sigue. Tono honesto de 'esta semana construí…'. "
        "Incluye SOLO lo que aparece en los commits/ADRs — nunca inventes features ni avances."
    ),
    categories=(),
    max_chars=2000,
    attachment=ATTACH_NONE,
)

_EXPLAINER = PostTypeSpec(
    post_type=PostType.EXPLAINER,
    system_overlay=(
        "TIPO DE POST: EXPLICADOR educativo de un concepto/paper/resultado de "
        "TERCEROS (no es tuyo). Le ENSEÑAS a la audiencia el qué/por qué, NO haces "
        "spotlight de una herramienta para linkearla. Audiencia: practicantes.\n"
        "Estructura objetivo:\n"
        "1. GANCHO (1 línea): una cifra sorprendente O una promesa de utilidad "
        "('te lo explico en 1 minuto').\n"
        "2. ENCUADRE / DES-HYPE: aterriza la expectativa ('esto NO es X' o 'no es "
        "magia, es…').\n"
        "3. NOMBRA EL SUJETO: di explícitamente de qué concepto/paper/resultado hablas.\n"
        "4. BLOQUES ETIQUETADOS de evidencia/explicación ('Los números:', 'Por qué "
        "importa:', 'Cómo funciona:') con viñetas; cada NÚMERO debe ir emparejado "
        "con un ancla de comparación (vs qué, cuánto mejor/peor).\n"
        "5. TRANSICIÓN de curiosidad/énfasis: una línea con 👇 o un 'ADEMÁS:' en "
        "negrita que abre el insight de fondo.\n"
        "6. EL INSIGHT CON NOMBRE: el modelo mental/idea central nombrada, + una "
        "advertencia/caveat honesta si corresponde.\n"
        "7. CIERRE corto e imperativo (1 línea), opcionalmente con UN emoji de firma.\n"
        "8. Enlace 'para profundizar' si el material lo aporta.\n"
        "REGLA DURA (no-fabricación): TODO número, cifra o comparativa debe aparecer "
        "VERBATIM en el material — NUNCA inventes, redondees ni adornes una cifra, y "
        "siempre emparéjala con su ancla de comparación tal como está en el material. "
        "Si el material no trae un número, no lo menciones. No copies texto verbatim: "
        "sintetiza y atribuye."
    ),
    categories=("concepts", "entities", "sources"),
    max_chars=2400,
    attachment=ATTACH_DIAGRAM,
    default_focus=Focus.USE,
    bullet_style=BULLET_ARROW,
)

_REGISTRY: dict[PostType, PostTypeSpec] = {
    PostType.REPO_DEEP_DIVE: _DEEP_DIVE,
    PostType.SHOWCASE: _SHOWCASE,
    PostType.TUTORIAL: _TUTORIAL,
    PostType.INFORMATIVO: _INFORMATIVO,
    PostType.PROJECT_LAUNCH: _PROJECT_LAUNCH,
    PostType.PROJECT_FEATURE: _PROJECT_FEATURE,
    PostType.PROJECT_WEEKLY: _PROJECT_WEEKLY,
    PostType.EXPLAINER: _EXPLAINER,
}

DEFAULT_POST_TYPE = PostType.REPO_DEEP_DIVE


def coerce_post_type(value: str | PostType | None) -> PostType:
    """Map a string/None to a :class:`PostType`, defaulting to the deep-dive.

    Type *inference* (D2) is intentionally NOT done here — an unknown/empty value
    falls back to the backward-compatible default rather than guessing (ADR-024)."""
    if isinstance(value, PostType):
        return value
    if not value:
        return DEFAULT_POST_TYPE
    try:
        return PostType(str(value).strip().lower())
    except ValueError:
        return DEFAULT_POST_TYPE


def spec_for(value: str | PostType | None) -> PostTypeSpec:
    """Return the :class:`PostTypeSpec` for a post type (coerced, never raises)."""
    return _REGISTRY[coerce_post_type(value)]


# --------------------------------------------------------------------------- #
# Composable CTA modifiers (ADR-044) — orthogonal to post_type and focus.
# --------------------------------------------------------------------------- #
#
# Both are off by default. Their factual values (a subscriber count, a product
# name/pitch/URL) are FACTS ABOUT THE REAL WORLD: the model must NEVER author
# them. They are caller-supplied and rendered DETERMINISTICALLY by the trailer
# builders below — enforced structurally, not just by prompt. A missing required
# field is a caller error (``ValueError``), never a silent model-filled gap.

# Short overlay HINTS appended to the system prompt when a modifier is active.
# They only tell the model to leave room for a closing block — the model writes
# NO CTA copy (the trailer builders render it from the dataclass fields).
LEAVE_ROOM_FOR_CLOSER = (
    "CIERRE: deja espacio para un enlace 'para profundizar' al final; un bloque "
    "de newsletter se añadirá automáticamente DESPUÉS de tu texto. No escribas tú "
    "el conteo de suscriptores ni inventes una cifra de audiencia."
)
LEAVE_ROOM_FOR_PS = (
    "CIERRE: deja espacio para una posdata (P.D.) suave al final; un bloque de "
    "producto se añadirá automáticamente DESPUÉS de tu texto. No escribas tú el "
    "nombre, el pitch ni la URL del producto, ni inventes capacidades o métricas."
)


@dataclass(frozen=True, slots=True)
class NewsletterCTA:
    """A 'go-deeper' newsletter call-to-action (ADR-044 modifier).

    ``url`` is required. ``proof`` (e.g. ``"leído por 4000+ ingenieros"``) and
    ``label`` are optional and rendered VERBATIM or omitted — the model never
    writes or invents a subscriber/reader count. The newsletter URL is promotion,
    NOT a source citation: it is exempt from the source-URL filter/guard."""

    url: str
    proof: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not (self.url and self.url.strip()):
            raise ValueError("NewsletterCTA requires a non-empty url")


@dataclass(frozen=True, slots=True)
class ProductPS:
    """A soft-sell product 'P.D.' (ADR-044 modifier).

    ``name`` and ``pitch`` (one line) are required; ``url`` is optional. All
    fields are rendered VERBATIM — the model never invents a capability, metric,
    or URL. Placed AFTER the body's value, low-pressure (mirrors the reference
    'prueba una demo de 10 s')."""

    name: str
    pitch: str
    url: str | None = None

    def __post_init__(self) -> None:
        if not (self.name and self.name.strip()):
            raise ValueError("ProductPS requires a non-empty name")
        if not (self.pitch and self.pitch.strip()):
            raise ValueError("ProductPS requires a non-empty pitch")


def render_product_ps(ps: ProductPS) -> str:
    """Render the deterministic product P.D. trailer FROM the dataclass fields
    only — the model never authors any part of this. Absent ``url`` → no URL line.

    Returned with a leading blank-line separator so it appends cleanly to a body."""
    name = ps.name.strip()
    pitch = ps.pitch.strip()
    line = f"P.D. {name} — {pitch}"
    if ps.url and ps.url.strip():
        line = f"{line} {ps.url.strip()}"
    return f"\n\n{line}"


def render_newsletter_cta(cta: NewsletterCTA) -> str:
    """Render the deterministic newsletter 'go-deeper' trailer FROM the dataclass
    fields only — the model never authors the proof/count. ``proof`` is rendered
    VERBATIM or omitted. The URL is promotion, not a ``Fuente:`` citation.

    Returned with a leading blank-line separator so it appends cleanly to a body."""
    label = (cta.label or "La versión detallada").strip()
    framing = label
    if cta.proof and cta.proof.strip():
        framing = f"{framing} ({cta.proof.strip()})"
    return f"\n\n{framing}: {cta.url.strip()}"
