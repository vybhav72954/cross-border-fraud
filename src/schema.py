"""
Dataset schema — the single role->column-name map that decouples both the
injection protocol (``src.inject``) and the feature extractors
(``src.models.ssm`` / ``src.models.sequence`` / ...) from any one dataset's
column names.

A ``Schema`` maps the benchmark's abstract roles to a base dataset's actual
columns. Five roles are required (entity / target / time / amount / label) and
are enough to host the ring / velocity / temporal typologies and to run the
per-entity sequence extractors. ``category`` unlocks the category typology; the
full location quad (entity + target lat/long) unlocks geo. ``row_id`` and
``unix_time`` are written through when present but never gate a typology.

``SPARKOV`` is the default adapter and reproduces the original Sparkov-coupled
behaviour exactly. Other base datasets supply their own ``Schema`` (e.g. a thin
PaySim/BankSim adapter returning ``(df, Schema)``); the injectors and extractors
then run unchanged on the native columns.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schema:
    """Maps the benchmark's abstract roles to a base dataset's column names.

    The five required roles (entity / target / time / amount / label) host
    ring / velocity / temporal and drive the per-entity sequence extractors.
    ``category`` unlocks the category typology; the location quad (entity +
    target lat/long) unlocks geo. ``row_id`` and ``unix_time`` are written
    through when present but never gate a typology."""
    entity: str            # who transacts          (Sparkov: cc_num)
    target: str            # what is transacted with (Sparkov: merchant)
    time: str              # event timestamp         (Sparkov: trans_date_trans_time)
    amount: str            # transaction amount      (Sparkov: amt)
    label: str             # fraud flag              (Sparkov: is_fraud)
    category: str | None = None        # Sparkov: category
    entity_lat: str | None = None      # Sparkov: lat
    entity_long: str | None = None     # Sparkov: long
    target_lat: str | None = None      # Sparkov: merch_lat
    target_long: str | None = None     # Sparkov: merch_long
    row_id: str | None = None          # Sparkov: trans_num
    unix_time: str | None = None       # Sparkov: unix_time

    @property
    def has_category(self) -> bool:
        return self.category is not None

    @property
    def has_location(self) -> bool:
        return None not in (self.entity_lat, self.entity_long,
                            self.target_lat, self.target_long)

    def supported_typologies(self) -> list[str]:
        """Typologies this schema can host, in injection order. ring/velocity/
        temporal need only the required roles; category needs a category column;
        geo needs the location quad."""
        typ = ["ring", "velocity", "temporal"]
        if self.has_category:
            typ.append("category")
        if self.has_location:
            typ.append("geo")
        return typ


SPARKOV = Schema(
    entity="cc_num",
    target="merchant",
    time="trans_date_trans_time",
    amount="amt",
    label="is_fraud",
    category="category",
    entity_lat="lat",
    entity_long="long",
    target_lat="merch_lat",
    target_long="merch_long",
    row_id="trans_num",
    unix_time="unix_time",
)
